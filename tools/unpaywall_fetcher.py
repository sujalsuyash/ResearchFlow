"""
tools/unpaywall_fetcher.py

LangChain StructuredTool that resolves a DOI to its Open Access PDF via Unpaywall.

Unlike the four search tools (Semantic Scholar, arXiv, PubMed, OpenAlex), this tool
is a DOI *lookup*, not a keyword search. The Research Agent calls it as a second
pass — once it has a DOI from a search tool, it checks here whether a free legal
PDF exists, allowing the Synthesis Chain to read full text instead of just abstracts.

Design contract with the rest of ResearchFlow
----------------------------------------------
• Input  : a single bare DOI string  (e.g. "10.1038/nature12373")
• Output : a dict with keys: doi, title, is_oa, oa_status, pdf_url, landing_url
• arXiv papers already carry pdf_url — callers should skip this tool for those.
• A 404 from Unpaywall (DOI unknown) returns a graceful "not found" dict; it
  never raises, so the agent pipeline is never blocked by a missing DOI.

Transport & rate-limiting
--------------------------
• _TokenBucket enforces ≤ 1 req/sec (Unpaywall's fair-use recommendation).
• _throttled_get() owns the HTTP call, bucket drain, and retry loop.
• Up to RETRY_ATTEMPTS total attempts on 429; non-429/non-404 errors raise immediately.

Caching
-------
• Redis (TTL = 7 days) with transparent in-process dict fallback.
• Cache key = CACHE_KEY_PREFIX + MD5(doi) — DOI is the natural unique key.

Environment variables
---------------------
REDIS_URL          Redis connection string (default: redis://localhost:6379/0)
UNPAYWALL_EMAIL    Registered email sent with every request (Unpaywall requirement)
                   Default: researchflow@example.com
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional

import httpx
import redis as redis_lib
import certifi
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL             = "https://api.unpaywall.org/v2"
REQUEST_TIMEOUT_SECS = 15

RETRY_ATTEMPTS   = 4    # 1 original + 3 retries on 429
RETRY_BASE_DELAY = 10   # seconds — doubles each attempt
RETRY_MAX_DELAY  = 120  # seconds — hard ceiling

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60    # 7 days
CACHE_KEY_PREFIX  = "researchflow:unpaywall:"

# ---------------------------------------------------------------------------
# Rate limiter — identical _TokenBucket used across all ResearchFlow tools
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Single-token bucket: enforces a minimum interval between requests."""

    def __init__(self, rate: float = 1.0):
        self._rate       = rate           # tokens added per second
        self._tokens     = 1.0            # start with one ready token
        self._last_check = time.monotonic()

    def consume(self) -> float:
        """
        Deduct one token and return how many seconds the caller must sleep.
        Returns 0.0 when a token was available immediately.
        """
        now     = time.monotonic()
        elapsed = now - self._last_check
        self._last_check = now
        self._tokens = min(1.0, self._tokens + elapsed * self._rate)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0

        wait         = (1.0 - self._tokens) / self._rate
        self._tokens = 0.0
        return wait


_bucket = _TokenBucket(rate=1.0)

# ---------------------------------------------------------------------------
# Redis + in-process fallback cache
# ---------------------------------------------------------------------------

_fallback_cache: dict[str, dict] = {}


def _connect_redis() -> Optional[redis_lib.Redis]:
    """
    Connect to Redis and verify with PING.
    Returns None (with a warning) if Redis is unreachable — the tool continues
    with the fallback dict so it never crashes in Redis-free environments.
    """
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis_lib.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=2, ssl_ca_certs=certifi.where()
        )
        client.ping()
        host = urlparse(url).hostname
        logger.info("Redis cache connected: %s", host)

        return client
    except Exception as exc:
        logger.warning(
            "Redis unavailable (%s: %s). Falling back to in-process dict cache.",
            type(exc).__name__, exc,
        )
        return None


_redis: Optional[redis_lib.Redis] = _connect_redis()


def _cache_key(doi: str) -> str:
    """Prefixed MD5 key — DOI is already a natural unique identifier."""
    digest = hashlib.md5(doi.encode()).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class UnpaywallInput(BaseModel):
    doi: str = Field(
        description=(
            "A bare DOI string to look up, e.g. '10.1038/nature12373'. "
            "Do not include 'https://doi.org/' — pass the DOI only."
        )
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _not_found_dict(doi: str) -> dict:
    """Returned when Unpaywall responds with 404 (DOI unknown to them)."""
    return {
        "doi":         doi,
        "title":       None,
        "is_oa":       False,
        "oa_status":   "not_found",
        "pdf_url":     None,
        "landing_url": None,
        "error":       f"DOI not found in Unpaywall: {doi}",
    }


def _parse_response(doi: str, data: dict) -> dict:
    """
    Extract the ResearchFlow-standard fields from a successful Unpaywall response.

    best_oa_location can be None for closed-access papers — guarded with `or {}`.
    """
    best = data.get("best_oa_location") or {}
    return {
        "doi":         doi,
        "title":       data.get("title"),
        "is_oa":       bool(data.get("is_oa", False)),
        "oa_status":   data.get("oa_status"),
        "pdf_url":     best.get("url_for_pdf"),
        "landing_url": best.get("url_for_landing_page"),
    }


# ---------------------------------------------------------------------------
# HTTP layer — rate-limited, with retry on 429
# ---------------------------------------------------------------------------

def _throttled_get(url: str, params: dict) -> httpx.Response:
    """
    Execute a GET request against `url`, enforcing rate-limiting and retrying on 429.

    Behaviour by status:
      200          → return the Response object
      404          → raise HTTPStatusError immediately (caller handles gracefully)
      429          → sleep (honouring Retry-After if present) and retry
      anything else→ raise HTTPStatusError immediately (pipeline-level error)

    The token bucket is drained once before the first attempt. Retry sleeps are
    additional and do not re-drain the bucket.
    """
    wait = _bucket.consume()
    if wait > 0:
        logger.debug("Rate limiting: sleeping %.3fs before Unpaywall request", wait)
        time.sleep(wait)

    last_exc: Optional[Exception] = None
    succeeded = False

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECS) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
            succeeded = True
            break

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code

            if status == 404:
                raise   # caller converts this to a graceful "not found" result

            if status != 429:
                raise   # 5xx, 401, 403, etc. — not retryable, surface immediately

            # --- 429 handling ---
            last_exc = exc
            if attempt == RETRY_ATTEMPTS:
                break   # exhausted — will raise below

            retry_after = exc.response.headers.get("Retry-After")
            if retry_after and str(retry_after).isdigit():
                delay = int(retry_after)
                logger.warning(
                    "Unpaywall 429 (attempt %d/%d). Honouring Retry-After: %ds.",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            else:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                logger.warning(
                    "Unpaywall 429 (attempt %d/%d). Exponential backoff: %ds.",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            time.sleep(delay)

    if not succeeded:
        raise last_exc  # type: ignore[misc]

    return response


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def fetch_unpaywall(doi: str) -> dict:
    """
    Resolve a DOI to its Open Access status and PDF URL via the Unpaywall API.

    Returns a dict with keys: doi, title, is_oa, oa_status, pdf_url, landing_url.
    Never raises — a 404 or empty DOI returns a graceful "not_found" dict.

    Example
    -------
    >>> fetch_unpaywall("10.1038/nature12373")
    {
        "doi":         "10.1038/nature12373",
        "title":       "Quantum entanglement ...",
        "is_oa":       True,
        "oa_status":   "gold",
        "pdf_url":     "https://...",
        "landing_url": "https://...",
    }
    """
    doi = doi.strip()
    if not doi:
        return _not_found_dict(doi)

    key = _cache_key(doi)

    # --- Cache READ ---
    if _redis is not None:
        try:
            cached = _redis.get(key)
            if cached is not None:
                logger.debug("Redis cache hit for DOI=%r", doi)
                return json.loads(cached)
        except Exception as exc:
            logger.warning(
                "Redis GET failed (%s: %s). Proceeding to API call.",
                type(exc).__name__, exc,
            )
    elif key in _fallback_cache:
        logger.debug("Fallback cache hit for DOI=%r", doi)
        return _fallback_cache[key]

    # --- API call ---
    email  = os.environ.get("UNPAYWALL_EMAIL", "yadavsujal2507@gmail.com")
    url    = f"{BASE_URL}/{doi}"
    params = {"email": email}

    logger.debug("Unpaywall API call: DOI=%r email=%s", doi, email)

    try:
        response = _throttled_get(url, params)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.info("Unpaywall: DOI not found — %r", doi)
            return _not_found_dict(doi)
        raise   # propagate non-404 HTTP errors to the caller

    result = _parse_response(doi, response.json())

    # --- Cache WRITE ---
    if _redis is not None:
        try:
            _redis.setex(key, CACHE_TTL_SECONDS, json.dumps(result))
            logger.debug("Wrote DOI=%r to Redis (TTL=%ds)", doi, CACHE_TTL_SECONDS)
        except Exception as exc:
            logger.warning(
                "Redis SET failed (%s: %s). Result not cached.",
                type(exc).__name__, exc,
            )
    else:
        _fallback_cache[key] = result

    logger.info(
        "Unpaywall: DOI=%r is_oa=%s oa_status=%s pdf_url=%s",
        doi, result["is_oa"], result["oa_status"], result["pdf_url"],
    )
    return result


# ---------------------------------------------------------------------------
# LangChain Tool definition
# ---------------------------------------------------------------------------

def _tool_fn(doi: str) -> str:
    """
    Agent-facing wrapper: calls fetch_unpaywall and formats the result as
    a concise Markdown block the LLM can read and cite directly.
    """
    result = fetch_unpaywall(doi)

    # Graceful not-found path
    if result.get("oa_status") == "not_found":
        return (
            f"## Open Access Lookup: `{doi}`\n\n"
            f"❌ **Not found in Unpaywall.** This DOI is either invalid or "
            f"not yet indexed. No free PDF available."
        )

    # Build status badge
    status = (result.get("oa_status") or "unknown").lower()
    badge  = {
        "gold":    "🟡 Gold OA",
        "green":   "🟢 Green OA",
        "hybrid":  "🔵 Hybrid OA",
        "bronze":  "🟤 Bronze OA",
        "closed":  "🔴 Closed Access",
        "unknown": "⚪ Unknown",
    }.get(status, f"⚪ {status.title()}")

    title_line   = f"**Title:** {result['title']}" if result.get("title") else ""
    pdf_line     = f"**PDF:**   {result['pdf_url']}" if result.get("pdf_url") else "**PDF:** Not available"
    landing_line = f"**Landing page:** {result['landing_url']}" if result.get("landing_url") else ""

    lines = [
        f"## Open Access Status: `{doi}`",
        "",
        title_line,
        f"**OA Status:** {badge}",
        pdf_line,
    ]
    if landing_line:
        lines.append(landing_line)

    return "\n".join(line for line in lines if line != "")


unpaywall_tool = StructuredTool.from_function(
    func=_tool_fn,
    name="unpaywall_lookup",
    description=(
        "Look up whether a paper is freely available as an Open Access PDF using its DOI. "
        "Returns the direct PDF download URL if one exists, plus the OA status "
        "(gold, green, hybrid, bronze, or closed). "
        "Use this AFTER a search tool has returned a DOI, to retrieve full-text content "
        "for the Synthesis Chain. Skip arXiv papers — they already carry a pdf_url."
    ),
    args_schema=UnpaywallInput,
    return_direct=False,
)


# ---------------------------------------------------------------------------
# Cache management helpers
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """
    Delete all ResearchFlow Unpaywall cache entries.
    Redis mode uses SCAN (non-blocking); fallback mode clears the dict.
    """
    if _redis is not None:
        try:
            cursor, deleted = 0, 0
            pattern = f"{CACHE_KEY_PREFIX}*"
            while True:
                cursor, keys = _redis.scan(cursor, match=pattern, count=100)
                if keys:
                    _redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(
                "Redis cache cleared: %d key(s) with prefix '%s'.",
                deleted, CACHE_KEY_PREFIX,
            )
        except Exception as exc:
            logger.warning("Redis clear failed (%s: %s).", type(exc).__name__, exc)
    else:
        count = len(_fallback_cache)
        _fallback_cache.clear()
        logger.info("Fallback cache cleared: removed %d entry/entries.", count)


def cache_stats() -> dict:
    """Return backend name, entry count, and TTL info."""
    if _redis is not None:
        try:
            keys: list[str] = []
            cursor = 0
            while True:
                cursor, batch = _redis.scan(cursor, match=f"{CACHE_KEY_PREFIX}*", count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            sample_ttl = _redis.ttl(keys[0]) if keys else None
            return {
                "backend":                  "redis",
                "entries":                  len(keys),
                "key_prefix":               CACHE_KEY_PREFIX,
                "ttl_seconds":              CACHE_TTL_SECONDS,
                "sample_key_ttl_remaining": sample_ttl,
            }
        except Exception as exc:
            return {"backend": "redis", "error": str(exc)}
    else:
        return {
            "backend": "in-process dict (Redis unavailable)",
            "entries": len(_fallback_cache),
            "keys":    list(_fallback_cache.keys()),
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    TEST_DOIS = [
        "10.1038/nature12373",         # well-known Nature paper — usually OA
        "10.1371/journal.pmed.0020124",# PLOS Medicine — gold OA
        "10.9999/this.does.not.exist", # intentional 404
    ]

    for doi in TEST_DOIS:
        print(f"\n{'='*60}")
        result = fetch_unpaywall(doi)
        print(f"DOI     : {result['doi']}")
        print(f"is_oa   : {result['is_oa']}")
        print(f"status  : {result['oa_status']}")
        print(f"pdf_url : {result['pdf_url']}")

    print("\n=== Via LangChain tool (agent string output) ===")
    print(unpaywall_tool.invoke({"doi": "10.1371/journal.pmed.0020124"}))

    print("\n=== Cache hit test ===")
    r1 = fetch_unpaywall("10.1038/nature12373")
    r2 = fetch_unpaywall("10.1038/nature12373")
    assert r1 == r2, "Cache miss!"
    print("Cache hit confirmed ✓")
    print(f"Stats: {cache_stats()}")

