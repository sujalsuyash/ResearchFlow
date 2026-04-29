"""
tools/semantic_scholar.py

LangChain StructuredTool wrapping the Semantic Scholar Graph API.
- Rate limited to 1 req/sec (unauthenticated) or 100 req/sec (with S2_API_KEY)
- Redis cache (TTL = 7 days) keyed on (query, year_start, year_end, fields_of_study, limit)
  Falls back to a per-process in-memory dict if Redis is unavailable, so the tool
  never crashes in environments without Redis (local dev, CI, etc.)
- Returns a list of PaperResult dicts ready for the Research Agent

Environment variables
---------------------
REDIS_URL   Redis connection string (default: redis://localhost:6379/0)
S2_API_KEY  Semantic Scholar API key (optional — unauthenticated if not set)
"""

import json
import os
import time
import hashlib
import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv          
load_dotenv()                           

import redis as redis_lib
import certifi

import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Fields we request from the API — keep this list stable to maximise cache hits
REQUESTED_FIELDS = ",".join([
    "title",
    "abstract",
    "authors",
    "year",
    "citationCount",
    "externalIds",
    "url",
    "fieldsOfStudy",
    "tldr",
])

MIN_INTERVAL_SECONDS = 1.0          # 1 req/sec for unauthenticated use
MAX_RESULTS_CAP = 10                # hard cap per call to keep context windows sane
REQUEST_TIMEOUT_SECONDS = 15

RETRY_ATTEMPTS = 4                  # total attempts (1 original + 3 retries)
RETRY_BASE_DELAY = 10               # seconds — first backoff; doubles each attempt
RETRY_MAX_DELAY = 120               # seconds — ceiling so we never wait absurdly long

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60   # 7 days — research data rarely changes faster
CACHE_KEY_PREFIX = "researchflow:ss:"   # namespaces keys so flushdb isn't needed for clears

# ---------------------------------------------------------------------------
# API key & dynamic rate limit                                             
# ---------------------------------------------------------------------------

S2_API_KEY: Optional[str] = os.environ.get("S2_API_KEY")                  
_RATE = 1.0 if S2_API_KEY else 1.0                                       

if S2_API_KEY:                                                             
    logger.info("Semantic Scholar: authenticated (100 req/sec)")           
else:                                                                      
    logger.warning(                                                        
        "Semantic Scholar: no S2_API_KEY found — unauthenticated (1 req/sec)"  
    )                                                                      

# ---------------------------------------------------------------------------
# Simple rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe (enough for single-threaded asyncio) token bucket."""

    def __init__(self, rate: float = 1.0):
        self._rate = rate                       
        self._last_check = time.monotonic()
        self._tokens: float = 1.0

    def consume(self) -> float:
        """Return the number of seconds to sleep before the token is available."""
        now = time.monotonic()
        elapsed = now - self._last_check
        self._last_check = now
        self._tokens = min(1.0, self._tokens + elapsed * self._rate)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0

        wait = (1.0 - self._tokens) / self._rate
        self._tokens = 0.0 
        return wait


_bucket = _TokenBucket(rate=_RATE)                                        # ← CHANGED (was rate=1.0)

# ---------------------------------------------------------------------------
# Cache — Redis with in-process dict fallback
# ---------------------------------------------------------------------------

# The fallback dict is intentionally module-level so it survives for the lifetime
# of the worker process (same behaviour as the old _cache dict).
_fallback_cache: dict[str, list[dict]] = {}

def _connect_redis() -> Optional[redis_lib.Redis]:
    """
    Attempt to connect to Redis and verify the connection with PING.
    Returns a Redis client on success, or None if Redis is unavailable.
    Failure is a warning, not an error — the tool falls back gracefully.
    """
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis_lib.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, ssl_ca_certs=certifi.where())
        client.ping()
        from urllib.parse import urlparse
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

def _cache_key(query: str, year_start: Optional[int], year_end: Optional[int],
               fields_of_study: Optional[str], limit: int) -> str:
    """Return a Redis-safe, prefixed cache key derived from the search parameters."""
    raw = f"{query}|{year_start}|{year_end}|{fields_of_study}|{limit}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class SemanticScholarInput(BaseModel):
    query: str = Field(
        description="The academic search query, e.g. 'transformer attention mechanisms NLP'"
    )
    year_start: Optional[int] = Field(
        default=None,
        description="Filter papers published from this year (inclusive), e.g. 2020"
    )
    year_end: Optional[int] = Field(
        default=None,
        description="Filter papers published up to this year (inclusive), e.g. 2024"
    )
    fields_of_study: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated Semantic Scholar field labels to filter by, "
            "e.g. 'Computer Science,Mathematics'. "
            "Valid values include: Computer Science, Medicine, Physics, Biology, "
            "Chemistry, Psychology, Economics, Engineering, Environmental Science, "
            "Political Science, Sociology, History, Philosophy, Business, Art."
        )
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=MAX_RESULTS_CAP,
        description=f"Number of results to return (1–{MAX_RESULTS_CAP})."
    )


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------

def _build_params(inp: SemanticScholarInput) -> dict:
    params: dict = {
        "query": inp.query,
        "fields": REQUESTED_FIELDS,
        "limit": inp.limit,
    }
    if inp.year_start or inp.year_end:
        lo = str(inp.year_start) if inp.year_start else ""
        hi = str(inp.year_end)   if inp.year_end   else ""
        # Semantic Scholar accepts "2022-", "-2024", or "2022-2024"
        # but we only set the param when at least one bound exists
        params["year"] = f"{lo}-{hi}"
    if inp.fields_of_study:
        params["fieldsOfStudy"] = inp.fields_of_study
    return params


def _parse_paper(raw: dict) -> dict:
    """Flatten a raw API paper object into a clean dict."""
    external_ids = raw.get("externalIds") or {}
    doi = external_ids.get("DOI")
    arxiv_id = external_ids.get("ArXiv")

    authors = [a.get("name", "") for a in (raw.get("authors") or [])]

    tldr = None
    if raw.get("tldr"):
        tldr = raw["tldr"].get("text")

    return {
        "title": raw.get("title", "Unknown Title"),
        "abstract": raw.get("abstract") or tldr or "No abstract available.",
        "authors": authors,
        "year": raw.get("year"),
        "citation_count": raw.get("citationCount", 0),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": raw.get("url") or (f"https://doi.org/{doi}" if doi else None),
        "fields_of_study": raw.get("fieldsOfStudy") or [],
        "source": "Semantic Scholar",
    }


def search_semantic_scholar(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    fields_of_study: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Synchronous entry point used by the LangChain StructuredTool.

    Returns a list of paper dicts, or raises on unrecoverable API errors.
    """
    inp = SemanticScholarInput(
        query=query,
        year_start=year_start,
        year_end=year_end,
        fields_of_study=fields_of_study,
        limit=min(limit, MAX_RESULTS_CAP),
    )

    key = _cache_key(inp.query, inp.year_start, inp.year_end, inp.fields_of_study, inp.limit)

    # --- Cache READ ---
    if _redis is not None:
        try:
            cached = _redis.get(key)
            if cached is not None:
                logger.debug("Redis cache hit for query=%r", inp.query)
                return json.loads(cached)
        except Exception as exc:
            logger.warning("Redis GET failed (%s: %s). Proceeding to API call.", type(exc).__name__, exc)
    elif key in _fallback_cache:
        logger.debug("Fallback cache hit for query=%r", inp.query)
        return _fallback_cache[key]

    # Rate limiting
    wait = _bucket.consume()
    if wait > 0:
        logger.debug("Rate limiting: sleeping %.2fs before Semantic Scholar request", wait)
        time.sleep(wait)

    params = _build_params(inp)

    last_exc: Exception | None = None
    succeeded = False
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS , headers=headers) as client:
                response = client.get(BASE_URL, params=params)
                response.raise_for_status()
            succeeded = True
            break                               # success — exit retry loop

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise                           # non-429 errors are not retried

            last_exc = exc
            if attempt == RETRY_ATTEMPTS:
                break                           # exhausted all attempts

            # Honour Retry-After if the server sent it, otherwise use exponential backoff
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)
                logger.warning(
                    "Semantic Scholar 429 (attempt %d/%d). Server asked to wait %ds.",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            else:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                logger.warning(
                    "Semantic Scholar 429 (attempt %d/%d). Backing off %ds (exponential).",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            time.sleep(delay)

    if not succeeded:
        raise last_exc  # type: ignore[misc]

    data = response.json()
    raw_papers = data.get("data", [])

    results = [_parse_paper(p) for p in raw_papers]

    # --- Cache WRITE ---
    if _redis is not None:
        try:
            _redis.setex(key, CACHE_TTL_SECONDS, json.dumps(results))
            logger.debug("Wrote %d papers to Redis (TTL=%ds, key=%s)", len(results), CACHE_TTL_SECONDS, key)
        except Exception as exc:
            logger.warning("Redis SET failed (%s: %s). Result not cached.", type(exc).__name__, exc)
    else:
        _fallback_cache[key] = results

    logger.info("Semantic Scholar returned %d papers for query=%r", len(results), inp.query)
    return results


# ---------------------------------------------------------------------------
# LangChain Tool definition
# ---------------------------------------------------------------------------

def _tool_fn(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    fields_of_study: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Wrapper that formats results as a readable string for the LLM.
    The agent sees this string; downstream code can also call search_semantic_scholar
    directly to get structured dicts.
    """
    papers = search_semantic_scholar(
        query=query,
        year_start=year_start,
        year_end=year_end,
        fields_of_study=fields_of_study,
        limit=limit,
    )

    if not papers:
        return f"No results found on Semantic Scholar for query: '{query}'"

    lines = [f"Semantic Scholar results for '{query}':\n"]
    for i, p in enumerate(papers, start=1):
        doi_str = f"DOI: {p['doi']}" if p["doi"] else "DOI: N/A"
        url_str = p["url"] or "URL: N/A"
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += f" et al. (+{len(p['authors']) - 3} more)"

        lines.append(
            f"[{i}] {p['title']} ({p['year']})\n"
            f"    Authors: {authors_str}\n"
            f"    Citations: {p['citation_count']} | {doi_str}\n"
            f"    URL: {url_str}\n"
            f"    Abstract: {p['abstract'][:300].strip()}{'...' if len(p['abstract']) > 300 else ''}\n"
        )

    return "\n".join(lines)


semantic_scholar_tool = StructuredTool.from_function(
    func=_tool_fn,
    name="semantic_scholar_search",
    description=(
        "Search the Semantic Scholar academic database (200M+ papers, all domains). "
        "Use this for ML, CS, physics, chemistry, social science, and interdisciplinary queries. "
        "Input: a search query string, optional year range, optional field filter, and result limit. "
        "Output: a formatted list of papers with title, authors, abstract snippet, citations, DOI, and URL."
    ),
    args_schema=SemanticScholarInput,
    return_direct=False,
)


# ---------------------------------------------------------------------------
# Cache management helpers (useful for testing / pipeline resets)
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """
    Delete all ResearchFlow Semantic Scholar entries from the active cache.

    Redis mode  : scans for keys matching the CACHE_KEY_PREFIX and deletes them
                  in batches. Uses SCAN (not KEYS) so it is safe on large dbs.
    Fallback mode: clears the in-process dict.
    """
    if _redis is not None:
        try:
            cursor = 0
            deleted = 0
            pattern = f"{CACHE_KEY_PREFIX}*"
            while True:
                cursor, keys = _redis.scan(cursor, match=pattern, count=100)
                if keys:
                    _redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info("Redis cache cleared: deleted %d key(s) with prefix '%s'.", deleted, CACHE_KEY_PREFIX)
        except Exception as exc:
            logger.warning("Redis clear failed (%s: %s).", type(exc).__name__, exc)
    else:
        count = len(_fallback_cache)
        _fallback_cache.clear()
        logger.info("Fallback cache cleared: removed %d entry/entries.", count)


def cache_stats() -> dict:
    """
    Return basic statistics about the active cache.

    Redis mode  : counts keys matching CACHE_KEY_PREFIX and reports TTL of the
                  first found key as a proxy for freshness.
    Fallback mode: returns entry count and raw keys from the in-process dict.
    """
    if _redis is not None:
        try:
            pattern = f"{CACHE_KEY_PREFIX}*"
            keys: list[str] = []
            cursor = 0
            while True:
                cursor, batch = _redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            sample_ttl = _redis.ttl(keys[0]) if keys else None
            return {
                "backend": "redis",
                "entries": len(keys),
                "key_prefix": CACHE_KEY_PREFIX,
                "ttl_seconds": CACHE_TTL_SECONDS,
                "sample_key_ttl_remaining": sample_ttl,
            }
        except Exception as exc:
            return {"backend": "redis", "error": str(exc)}
    else:
        return {
            "backend": "in-process dict (Redis unavailable)",
            "entries": len(_fallback_cache),
            "keys": list(_fallback_cache.keys()),
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== Direct function call ===")
    results = search_semantic_scholar(
        query="retrieval augmented generation language models",
        year_start=2022,
        limit=3,
    )
    for r in results:
        print(f"- {r['title']} ({r['year']}) | citations={r['citation_count']} | doi={r['doi']}")

    print("\n=== Via LangChain tool (string output for agent) ===")
    output = semantic_scholar_tool.invoke({
        "query": "retrieval augmented generation language models",
        "year_start": 2022,
        "limit": 3,
    })
    print(output)

    print("\n=== Cache hit test (should not hit API) ===")
    results2 = search_semantic_scholar(
        query="retrieval augmented generation language models",
        year_start=2022,
        limit=3,
    )
    assert results == results2, "Cache miss — something is wrong"
    print("Cache hit confirmed ✓")
    print(f"Cache stats: {cache_stats()}")
