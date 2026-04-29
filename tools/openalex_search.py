"""
tools/openalex_search.py

LangChain StructuredTool wrapping the OpenAlex Works API.

OpenAlex indexes 250 M+ scholarly works across all disciplines and is fully
open — no API key required.  Sending an email address via the `mailto` query
parameter moves requests into the "Polite Pool", which receives higher rate
limits and prioritised routing from OpenAlex's CDN.

Two-step design:
  _throttled_get()  — rate-limited, retrying HTTP layer (same as other tools)
  search_openalex() — cache → HTTP → parse pipeline; returns PaperResult dicts
  _tool_fn()        — formats results as a Markdown string for the LLM agent

Abstract reconstruction:
  OpenAlex stores abstracts as an *inverted index* — a dict mapping each word
  to the list of positions it occupies in the text.  _reconstruct_abstract()
  reverses this into a readable string by sorting (position, word) pairs.

Environment variables
---------------------
OPENALEX_EMAIL  Contact email sent with every request to enter the Polite Pool.
                Example: researcher@university.edu
                Defaults to "researchflow@example.com" — fine for low volume;
                set your real address in production.
REDIS_URL       Redis connection string (default: redis://localhost:6379/0)
"""

from __future__ import annotations

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

BASE_URL = "https://api.openalex.org/works"

# Fields we request via the `select` param — keeps payloads small and cache
# keys stable across API version bumps.
REQUESTED_FIELDS = ",".join([
    "id",
    "title",
    "display_name",
    "abstract_inverted_index",
    "authorships",
    "publication_year",
    "cited_by_count",
    "doi",
    "ids",
    "primary_location",
])

MAX_RESULTS_CAP      = 10
REQUEST_TIMEOUT_SECS = 15
CACHE_TTL_SECONDS    = 7 * 24 * 60 * 60    # 7 days
CACHE_KEY_PREFIX     = "researchflow:openalex:"

RETRY_ATTEMPTS   = 4
RETRY_BASE_DELAY = 10       # seconds; doubles each attempt
RETRY_MAX_DELAY  = 120      # ceiling

# OpenAlex Polite Pool allows ~10 req/sec
_RATE_LIMIT_RPS = 10.0

# Polite Pool identifier — set OPENALEX_EMAIL in env for production use
_OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "researchflow@example.com")

# ---------------------------------------------------------------------------
# Token-bucket rate limiter  (identical design to semantic_scholar.py)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe-enough (single-threaded asyncio) token bucket."""

    def __init__(self, rate: float) -> None:
        self._rate       = rate
        self._tokens     = 1.0
        self._last_check = time.monotonic()

    def consume(self) -> float:
        """Return seconds to sleep before the next request may fire."""
        now              = time.monotonic()
        elapsed          = now - self._last_check
        self._last_check = now
        self._tokens     = min(1.0, self._tokens + elapsed * self._rate)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0

        wait         = (1.0 - self._tokens) / self._rate
        self._tokens = 0.0
        return wait


_bucket = _TokenBucket(rate=_RATE_LIMIT_RPS)

# ---------------------------------------------------------------------------
# Redis cache with in-process dict fallback
# ---------------------------------------------------------------------------

_fallback_cache: dict[str, list[dict]] = {}


def _connect_redis() -> Optional[redis_lib.Redis]:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis_lib.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, ssl_ca_certs=certifi.where())
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


def _cache_key(
    query: str,
    year_start: Optional[int],
    year_end: Optional[int],
    concept: Optional[str],
    limit: int,
) -> str:
    """Deterministic, prefixed cache key derived from all search parameters."""
    raw    = f"{query}|{year_start}|{year_end}|{concept}|{limit}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


def _cache_read(key: str) -> Optional[list[dict]]:
    if _redis is not None:
        try:
            cached = _redis.get(key)
            if cached is not None:
                logger.debug("Redis cache hit (key=%s)", key)
                return json.loads(cached)
        except Exception as exc:
            logger.warning("Redis GET failed (%s: %s). Proceeding to API.", type(exc).__name__, exc)
    elif key in _fallback_cache:
        logger.debug("Fallback cache hit (key=%s)", key)
        return _fallback_cache[key]
    return None


def _cache_write(key: str, results: list[dict]) -> None:
    if _redis is not None:
        try:
            _redis.setex(key, CACHE_TTL_SECONDS, json.dumps(results))
            logger.debug("Wrote %d works to Redis (key=%s)", len(results), key)
        except Exception as exc:
            logger.warning("Redis SET failed (%s: %s). Result not cached.", type(exc).__name__, exc)
    else:
        _fallback_cache[key] = results

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class OpenAlexInput(BaseModel):
    query: str = Field(
        description=(
            "Full-text search query across title, abstract, and full text of papers. "
            "Plain natural language works well, e.g. 'transformer self-attention NLP' "
            "or 'mRNA vaccine immunology clinical trial'."
        )
    )
    year_start: Optional[int] = Field(
        default=None,
        description="Filter works published from this year (inclusive), e.g. 2019.",
    )
    year_end: Optional[int] = Field(
        default=None,
        description="Filter works published up to this year (inclusive), e.g. 2024.",
    )
    concept: Optional[str] = Field(
        default=None,
        description=(
            "OpenAlex concept label to narrow results to a specific field, "
            "e.g. 'machine learning', 'genomics', 'climate change'. "
            "Uses a text-search match on OpenAlex's concept taxonomy — partial "
            "matches are supported. Leave None for broad cross-domain queries."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=MAX_RESULTS_CAP,
        description=f"Number of results to return (1–{MAX_RESULTS_CAP}).",
    )

# ---------------------------------------------------------------------------
# Abstract reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """
    Reconstruct a plain-text abstract from OpenAlex's inverted-index format.

    OpenAlex stores abstracts as {word: [position, position, ...]} rather than
    raw text (to avoid redistributing copyrighted content).  We reverse the
    mapping by flattening to (position, word) pairs, sorting by position, and
    joining the words.

    Returns "No abstract available." for None, empty dicts, or parse errors.
    """
    if not inverted_index:
        return "No abstract available."
    try:
        pos_word: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                pos_word.append((pos, word))
        if not pos_word:
            return "No abstract available."
        pos_word.sort()
        return " ".join(word for _, word in pos_word)
    except Exception as exc:
        logger.warning("Abstract reconstruction failed: %s", exc)
        return "No abstract available."


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_doi(raw_doi: Optional[str]) -> Optional[str]:
    """
    Return a bare DOI string (e.g. '10.1038/s41586-023-05543-x') or None.

    OpenAlex returns DOIs as full resolver URLs ('https://doi.org/10.xxx').
    We strip the prefix so the DOI is in the same format as the other tools,
    enabling cross-tool deduplication by DOI.
    """
    if not raw_doi:
        return None
    if "doi.org/" in raw_doi:
        return raw_doi.split("doi.org/", 1)[-1]
    return raw_doi  # already a bare DOI


def _extract_arxiv_id(raw_arxiv: Optional[str]) -> Optional[str]:
    """
    Return a bare arXiv ID (e.g. '1706.03762') or None.

    OpenAlex stores arXiv IDs as full URLs ('https://arxiv.org/abs/1706.03762').
    """
    if not raw_arxiv:
        return None
    if "arxiv.org/abs/" in raw_arxiv:
        return raw_arxiv.split("arxiv.org/abs/", 1)[-1]
    return raw_arxiv  # assume already bare


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------

def _build_filter(
    year_start: Optional[int],
    year_end: Optional[int],
) -> Optional[str]:
    """
    Compose the OpenAlex `filter` parameter string.
    OpenAlex filters are comma-separated clauses.
    """
    clauses: list[str] = []
    if year_start:
        clauses.append(f"from_publication_date:{year_start}-01-01")
    if year_end:
        clauses.append(f"to_publication_date:{year_end}-12-31")
    return ",".join(clauses) if clauses else None


# ---------------------------------------------------------------------------
# Query / result builders
# ---------------------------------------------------------------------------

def _build_params(inp: OpenAlexInput) -> dict:
    """Build the full dict of query parameters for the OpenAlex works endpoint."""
    # OpenAlex recently removed nested .search filters for concepts/topics.
    # The modern pattern is to append the concept directly to the main text search.
    query = inp.query
    if inp.concept and inp.concept.strip():
        query = f"{query} {inp.concept.strip()}"

    params: dict = {
        "search":   query,
        "select":   REQUESTED_FIELDS,
        "per-page": inp.limit,
        "mailto":   _OPENALEX_EMAIL,    # enters the Polite Pool
    }
    filter_str = _build_filter(inp.year_start, inp.year_end)
    if filter_str:
        params["filter"] = filter_str
    return params

def _parse_work(raw: dict) -> dict:
    """
    Normalise a raw OpenAlex work object into the ResearchFlow PaperResult schema.

    URL preference (highest to lowest):
      1. primary_location.pdf_url       — direct open-access PDF
      2. primary_location.landing_page_url — publisher/preprint landing page
      3. https://doi.org/{doi}          — DOI resolver
      4. https://arxiv.org/abs/{arxiv_id} — arXiv abstract page
      5. work's own OpenAlex canonical URL — always present
    """
    doi      = _extract_doi(raw.get("doi"))
    arxiv_id = _extract_arxiv_id((raw.get("ids") or {}).get("arxiv"))

    authors = [
        authorship.get("author", {}).get("display_name", "")
        for authorship in (raw.get("authorships") or [])
        if authorship.get("author", {}).get("display_name")
    ]

    abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))

    primary  = raw.get("primary_location") or {}
    url = (
        primary.get("pdf_url")
        or primary.get("landing_page_url")
        or (f"https://doi.org/{doi}"             if doi      else None)
        or (f"https://arxiv.org/abs/{arxiv_id}"  if arxiv_id else None)
        or raw.get("id")   # OpenAlex canonical page — absolute last resort
    )

    return {
        # ---- Universal ResearchFlow PaperResult schema ----
        "title":          raw.get("title") or raw.get("display_name") or "Unknown Title",
        "abstract":       abstract,
        "authors":        authors,
        "year":           raw.get("publication_year"),
        "citation_count": raw.get("cited_by_count"),
        "doi":            doi,
        "arxiv_id":       arxiv_id,
        "url":            url,
        "source":         "OpenAlex",
    }


# ---------------------------------------------------------------------------
# HTTP layer — rate-limited with exponential back-off on 429
# ---------------------------------------------------------------------------

def _throttled_get(url: str, params: dict) -> httpx.Response:
    """
    Issue a single rate-limited GET to the OpenAlex API.

    Retries up to RETRY_ATTEMPTS times on HTTP 429 with exponential back-off,
    honouring the Retry-After header when provided.  All other HTTP errors are
    re-raised immediately.
    """
    wait = _bucket.consume()
    if wait > 0:
        logger.debug("Rate limiting: sleeping %.2fs before OpenAlex request", wait)
        time.sleep(wait)

    last_exc: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECS) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
            return response

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise                   # non-429 errors are not retried

            last_exc = exc
            if attempt == RETRY_ATTEMPTS:
                break

            retry_after = exc.response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)
                logger.warning(
                    "OpenAlex 429 (attempt %d/%d). Server asked to wait %ds.",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            else:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                logger.warning(
                    "OpenAlex 429 (attempt %d/%d). Backing off %ds (exponential).",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search_openalex(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    concept: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Synchronous entry point used by the LangChain StructuredTool.

    Returns a list of PaperResult dicts, or raises on unrecoverable API errors.
    Results are cached in Redis (or the fallback dict) for 7 days.
    """
    inp = OpenAlexInput(
        query=query,
        year_start=year_start,
        year_end=year_end,
        concept=concept,
        limit=min(limit, MAX_RESULTS_CAP),
    )

    key = _cache_key(inp.query, inp.year_start, inp.year_end, inp.concept, inp.limit)

    # --- Cache READ ---
    cached = _cache_read(key)
    if cached is not None:
        return cached

    params   = _build_params(inp)
    response = _throttled_get(BASE_URL, params)
    data     = response.json()

    raw_works = data.get("results", [])
    results   = [_parse_work(w) for w in raw_works]
    results   = results[: inp.limit]   # guard against over-return

    logger.info(
        "OpenAlex returned %d works for query=%r",
        len(results), inp.query,
    )

    # --- Cache WRITE ---
    _cache_write(key, results)

    return results


# ---------------------------------------------------------------------------
# LangChain Tool definition
# ---------------------------------------------------------------------------

def _tool_fn(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    concept: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Markdown-formatted wrapper consumed by the ReAct agent's LLM.
    Downstream pipeline code should call search_openalex() for structured dicts.
    """
    papers = search_openalex(
        query=query,
        year_start=year_start,
        year_end=year_end,
        concept=concept,
        limit=limit,
    )

    if not papers:
        return f"No results found on OpenAlex for query: '{query}'"

    lines = [f"OpenAlex results for '{query}':\n"]
    for i, p in enumerate(papers, start=1):
        doi_str    = f"DOI: {p['doi']}"         if p["doi"]      else "DOI: N/A"
        arxiv_str  = f"arXiv: {p['arxiv_id']}"  if p["arxiv_id"] else None
        citations  = p["citation_count"] if p["citation_count"] is not None else "N/A"

        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += f" et al. (+{len(p['authors']) - 3} more)"

        id_parts = [doi_str]
        if arxiv_str:
            id_parts.append(arxiv_str)

        abstract_snippet = p["abstract"][:300].strip()
        if len(p["abstract"]) > 300:
            abstract_snippet += "..."

        lines.append(
            f"[{i}] {p['title']} ({p['year']})\n"
            f"    Authors: {authors_str}\n"
            f"    Citations: {citations} | {' | '.join(id_parts)}\n"
            f"    URL: {p['url'] or 'N/A'}\n"
            f"    Abstract: {abstract_snippet}\n"
        )

    return "\n".join(lines)


openalex_tool = StructuredTool.from_function(
    func=_tool_fn,
    name="openalex_search",
    description=(
        "Search OpenAlex (250M+ scholarly works, all disciplines, fully open). "
        "Best for cross-disciplinary queries, citation counts, open-access PDF discovery, "
        "and any topic not covered by a specialised database. "
        "Supports optional concept filters (e.g. 'machine learning', 'genomics') "
        "and year range filters. "
        "Returns title, authors, abstract, citation count, DOI, arXiv ID, and a "
        "direct PDF URL when available."
    ),
    args_schema=OpenAlexInput,
    return_direct=False,
)


# ---------------------------------------------------------------------------
# Cache management helpers
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """
    Delete all ResearchFlow OpenAlex cache entries.
    Uses SCAN (not KEYS) in Redis mode — safe on large databases.
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
                "Redis cache cleared: deleted %d key(s) with prefix '%s'.",
                deleted, CACHE_KEY_PREFIX,
            )
        except Exception as exc:
            logger.warning("Redis clear failed (%s: %s).", type(exc).__name__, exc)
    else:
        count = len(_fallback_cache)
        _fallback_cache.clear()
        logger.info("Fallback cache cleared: removed %d entry/entries.", count)


def cache_stats() -> dict:
    """Return cache backend info and current entry count."""
    if _redis is not None:
        try:
            keys: list[str] = []
            cursor = 0
            pattern = f"{CACHE_KEY_PREFIX}*"
            while True:
                cursor, batch = _redis.scan(cursor, match=pattern, count=100)
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

    print("=== Direct function call ===")
    results = search_openalex(
        query="retrieval augmented generation language models",
        year_start=2022,
        limit=3,
    )
    for r in results:
        print(
            f"- {r['title']} ({r['year']}) | "
            f"citations={r['citation_count']} | doi={r['doi']} | arxiv={r['arxiv_id']}"
        )

    print("\n=== With concept filter ===")
    concept_results = search_openalex(
        query="protein structure prediction",
        concept="bioinformatics",
        limit=2,
    )
    for r in concept_results:
        print(f"- {r['title']} | url={r['url']}")

    print("\n=== Via LangChain tool (agent string output) ===")
    output = openalex_tool.invoke({
        "query": "retrieval augmented generation language models",
        "year_start": 2022,
        "limit": 3,
    })
    print(output)

    print("\n=== Cache hit test ===")
    results2 = search_openalex(
        query="retrieval augmented generation language models",
        year_start=2022,
        limit=3,
    )
    assert results == results2, "Cache miss — something is wrong"
    print("Cache hit confirmed ✓")
    print(f"Cache stats: {cache_stats()}")
