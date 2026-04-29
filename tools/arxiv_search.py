"""
tools/arxiv_search.py

LangChain StructuredTool wrapping the arXiv API via the `arxiv` Python library.

Transport & rate-limiting:
  The `arxiv.Client` is configured with delay_seconds=3.0 and num_retries=3.
  It owns the HTTP layer, pacing, and retry logic — we do not duplicate those here.
  One module-level client is reused across all calls to preserve its internal state.

Caching:
  Redis (TTL = 7 days) with a transparent in-process dict fallback when Redis
  is unavailable — identical strategy to semantic_scholar.py.

Output schema:
  Matches semantic_scholar.py field-for-field, with two additions:
    - pdf_url   : direct arXiv PDF link (lets the Unpaywall fetcher skip arXiv papers)
    - categories: list of arXiv category codes (e.g. ['cs.LG', 'stat.ML'])
  citation_count is always None — arXiv does not expose citation data.

Environment variables
---------------------
REDIS_URL   Redis connection string (default: redis://localhost:6379/0)
"""

import hashlib
import json
import logging
import os
from typing import Optional

import arxiv
import redis as redis_lib
import certifi
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESULTS_CAP      = 10
CACHE_TTL_SECONDS    = 7 * 24 * 60 * 60    # 7 days
CACHE_KEY_PREFIX     = "researchflow:arxiv:"

# When a year filter is active we over-fetch so post-filtering still
# yields `limit` results. Capped at 100 (one arxiv API page).
YEAR_FILTER_MULTIPLIER = 4
ARXIV_MAX_FETCH        = 100

# ---------------------------------------------------------------------------
# Module-level arxiv client — reused across calls to keep its pacing state
# ---------------------------------------------------------------------------

_arxiv_client = arxiv.Client(
    page_size=ARXIV_MAX_FETCH,
    delay_seconds=3.0,          # 1 req / 3s — arxiv's recommended rate
    num_retries=3,              # library handles transient failures internally
)

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
    categories: Optional[str],
    limit: int,
) -> str:
    """Deterministic, prefixed cache key derived from all search parameters."""
    raw = f"{query}|{year_start}|{year_end}|{categories}|{limit}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class ArxivInput(BaseModel):
    query: str = Field(
        description=(
            "Full-text search query for arXiv. Supports arXiv query syntax, "
            "e.g. 'ti:transformer AND abs:attention' or plain natural language."
        )
    )
    year_start: Optional[int] = Field(
        default=None,
        description="Filter papers first published from this year (inclusive), e.g. 2020.",
    )
    year_end: Optional[int] = Field(
        default=None,
        description="Filter papers first published up to this year (inclusive), e.g. 2024.",
    )
    categories: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated arXiv category codes to filter by, "
            "e.g. 'cs.LG,cs.AI'. "
            "Common codes: cs.LG (machine learning), cs.CL (computation & language), "
            "cs.CV (computer vision), cs.AI (artificial intelligence), "
            "stat.ML (statistics - ML), physics.hep-th, q-bio.GN."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=MAX_RESULTS_CAP,
        description=f"Number of results to return (1–{MAX_RESULTS_CAP}).",
    )


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_query(inp: ArxivInput) -> str:
    """
    Compose the arXiv query string from user inputs.

    Category filtering is injected directly into the query using arXiv's
    native `cat:` field prefix, which is more reliable than the separate
    category parameter (which the library does not expose).

    Multiple categories are OR-combined: cs.LG,cs.AI → (cat:cs.LG OR cat:cs.AI)
    """
    query = inp.query.strip()

    if inp.categories:
        cat_clauses = [
            f"cat:{c.strip()}"
            for c in inp.categories.split(",")
            if c.strip()
        ]
        if cat_clauses:
            cat_expr = " OR ".join(cat_clauses)
            query = f"({query}) AND ({cat_expr})"

    return query


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------

def _extract_doi(result: arxiv.Result) -> Optional[str]:
    """
    Return a bare DOI string (e.g. '10.48550/arXiv.1706.03762') or None.

    The arxiv library stores DOI as a full resolver URL
    ('https://doi.org/10.xxx') or an empty string when absent.
    We strip the resolver prefix so the DOI is in the same format
    as semantic_scholar.py — enabling cross-tool deduplication by DOI.
    """
    raw = result.doi or ""
    if not raw:
        return None
    if "doi.org/" in raw:
        return raw.split("doi.org/", 1)[-1]
    return raw  # already a bare DOI


def _parse_result(result: arxiv.Result) -> dict:
    """Normalise an arxiv.Result into the ResearchFlow paper dict schema."""
    return {
        "title":          result.title,
        "abstract":       result.summary.replace("\n", " ").strip() or "No abstract available.",
        "authors":        [a.name for a in result.authors],
        "year":           result.published.year if result.published else None,
        "citation_count": None,                   # arXiv does not expose citations
        "doi":            _extract_doi(result),
        "arxiv_id":       result.get_short_id(),
        "url":            result.entry_id,         # canonical abstract page URL
        "pdf_url":        result.pdf_url,          # direct PDF — lets Unpaywall skip this paper
        "categories":     result.categories,
        "source":         "arXiv",
    }


# ---------------------------------------------------------------------------
# Year filter (post-fetch, because arXiv API has no clean year param)
# ---------------------------------------------------------------------------

def _passes_year_filter(paper: dict, year_start: Optional[int], year_end: Optional[int]) -> bool:
    if paper["year"] is None:
        return True     # keep papers whose date is unknown rather than silently drop them
    if year_start and paper["year"] < year_start:
        return False
    if year_end and paper["year"] > year_end:
        return False
    return True


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search_arxiv(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    categories: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Synchronous entry point used by the LangChain StructuredTool.

    Returns a list of paper dicts. Never raises on empty results — returns [].
    Raises arxiv.ArxivError (or subclasses) on unrecoverable API errors.
    """
    inp = ArxivInput(
        query=query,
        year_start=year_start,
        year_end=year_end,
        categories=categories,
        limit=min(limit, MAX_RESULTS_CAP),
    )

    key = _cache_key(inp.query, inp.year_start, inp.year_end, inp.categories, inp.limit)

    # --- Cache READ ---
    if _redis is not None:
        try:
            cached = _redis.get(key)
            if cached is not None:
                logger.debug("Redis cache hit for arXiv query=%r", inp.query)
                return json.loads(cached)
        except Exception as exc:
            logger.warning(
                "Redis GET failed (%s: %s). Proceeding to API call.",
                type(exc).__name__, exc,
            )
    elif key in _fallback_cache:
        logger.debug("Fallback cache hit for arXiv query=%r", inp.query)
        return _fallback_cache[key]

    # --- API call ---
    # Over-fetch when a year filter is active so post-filtering still
    # yields `limit` results in the common case.
    needs_year_filter = bool(inp.year_start or inp.year_end)
    fetch_count = min(
        inp.limit * YEAR_FILTER_MULTIPLIER if needs_year_filter else inp.limit,
        ARXIV_MAX_FETCH,
    )

    arxiv_query = _build_query(inp)
    search = arxiv.Search(
        query=arxiv_query,
        max_results=fetch_count,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    logger.debug(
        "arXiv API call: query=%r fetch_count=%d year_filter=%s",
        arxiv_query, fetch_count, needs_year_filter,
    )

    raw_results = list(_arxiv_client.results(search))
    all_papers = [_parse_result(r) for r in raw_results]

    # Post-filter by year if requested
    if needs_year_filter:
        all_papers = [
            p for p in all_papers
            if _passes_year_filter(p, inp.year_start, inp.year_end)
        ]

    results = all_papers[: inp.limit]

    logger.info(
        "arXiv returned %d papers (fetched %d, after year filter) for query=%r",
        len(results), len(raw_results), inp.query,
    )

    # --- Cache WRITE ---
    if _redis is not None:
        try:
            _redis.setex(key, CACHE_TTL_SECONDS, json.dumps(results))
            logger.debug(
                "Wrote %d papers to Redis (TTL=%ds, key=%s)",
                len(results), CACHE_TTL_SECONDS, key,
            )
        except Exception as exc:
            logger.warning(
                "Redis SET failed (%s: %s). Result not cached.",
                type(exc).__name__, exc,
            )
    else:
        _fallback_cache[key] = results

    return results


# ---------------------------------------------------------------------------
# LangChain Tool definition
# ---------------------------------------------------------------------------

def _tool_fn(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    categories: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    String-formatted wrapper seen by the ReAct agent's LLM.
    Downstream pipeline code should call search_arxiv() directly for dicts.
    """
    papers = search_arxiv(
        query=query,
        year_start=year_start,
        year_end=year_end,
        categories=categories,
        limit=limit,
    )

    if not papers:
        return f"No results found on arXiv for query: '{query}'"

    lines = [f"arXiv results for '{query}':\n"]
    for i, p in enumerate(papers, start=1):
        doi_str = f"DOI: {p['doi']}" if p["doi"] else "DOI: N/A"
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += f" et al. (+{len(p['authors']) - 3} more)"
        cats_str = ", ".join(p["categories"][:4])
        abstract_snippet = p["abstract"][:300].strip()
        if len(p["abstract"]) > 300:
            abstract_snippet += "..."

        lines.append(
            f"[{i}] {p['title']} ({p['year']})\n"
            f"    Authors: {authors_str}\n"
            f"    arXiv ID: {p['arxiv_id']} | {doi_str}\n"
            f"    Categories: {cats_str}\n"
            f"    URL: {p['url']}\n"
            f"    PDF: {p['pdf_url'] or 'N/A'}\n"
            f"    Abstract: {abstract_snippet}\n"
        )

    return "\n".join(lines)


arxiv_tool = StructuredTool.from_function(
    func=_tool_fn,
    name="arxiv_search",
    description=(
        "Search the arXiv preprint server (2M+ papers in CS, physics, math, biology, "
        "economics, and statistics). Best for cutting-edge ML/AI, NLP, computer vision, "
        "and physics research where the latest preprints matter. "
        "Supports optional category codes (e.g. 'cs.LG', 'cs.CL') and year range filters. "
        "Returns title, authors, abstract, arXiv ID, DOI (if published), PDF URL, and categories."
    ),
    args_schema=ArxivInput,
    return_direct=False,
)


# ---------------------------------------------------------------------------
# Cache management helpers
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """
    Delete all ResearchFlow arXiv cache entries.
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
    import logging
    logging.basicConfig(level=logging.DEBUG)

    print("=== Direct function call ===")
    results = search_arxiv(
        query="retrieval augmented generation language models",
        year_start=2022,
        limit=3,
    )
    for r in results:
        print(
            f"- {r['title']} ({r['year']}) | "
            f"arxiv_id={r['arxiv_id']} | doi={r['doi']} | pdf={r['pdf_url']}"
        )

    print("\n=== With category filter ===")
    cs_results = search_arxiv(
        query="graph neural network knowledge graph",
        categories="cs.LG,cs.AI",
        limit=2,
    )
    for r in cs_results:
        print(f"- {r['title']} | cats={r['categories'][:2]}")

    print("\n=== Via LangChain tool (agent string output) ===")
    output = arxiv_tool.invoke({
        "query": "retrieval augmented generation language models",
        "year_start": 2022,
        "limit": 3,
    })
    print(output)

    print("\n=== Cache hit test ===")
    results2 = search_arxiv(
        query="retrieval augmented generation language models",
        year_start=2022,
        limit=3,
    )
    assert results == results2, "Cache miss — something is wrong"
    print("Cache hit confirmed ✓")
    print(f"Cache stats: {cache_stats()}")
