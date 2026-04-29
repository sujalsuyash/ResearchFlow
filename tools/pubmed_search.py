"""
tools/pubmed_search.py

LangChain StructuredTool wrapping the NCBI PubMed E-utilities API.

Two-step fetch strategy (mirrors NCBI's recommended approach):
  1. esearch  — POST query → receive a ranked list of PMIDs
  2. efetch   — POST PMID batch → receive full XML records

Rate limiting:
  NCBI enforces 3 req/sec without an API key, 10 req/sec with one.
  A _TokenBucket (identical to semantic_scholar.py) is configured to the correct
  rate automatically based on whether NCBI_API_KEY is set.

Caching:
  Redis (TTL = 7 days) with a transparent in-process dict fallback when Redis
  is unavailable — identical strategy to semantic_scholar.py and arxiv_search.py.

Output schema:
  Matches the ResearchFlow universal PaperResult schema field-for-field, plus two
  PubMed-specific extras that are useful downstream (mesh_terms, journal):
    title, abstract, authors, year, citation_count (None), doi, arxiv_id (None),
    url, source, pmid, mesh_terms, journal

Environment variables
---------------------
NCBI_API_KEY   NCBI API key — raises the rate limit from 3 to 10 req/sec and
               unlocks slightly higher result caps. Free to obtain at:
               https://www.ncbi.nlm.nih.gov/account/
               Optional — tool works without it (at the lower rate).
NCBI_EMAIL     Contact email sent with every NCBI request. NCBI policy requires
               this for any automated tool; they use it to reach developers if
               their servers are overloaded. Example: researcher@university.edu
               Defaults to "researchflow@example.com" if not set (sufficient for
               low-volume use but you should set your real address in production).
REDIS_URL      Redis connection string (default: redis://localhost:6379/0)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
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

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

MAX_RESULTS_CAP       = 10               # hard cap per call — keeps LLM context sane
REQUEST_TIMEOUT_SECS  = 20              # efetch XML can be large; give it room
CACHE_TTL_SECONDS     = 7 * 24 * 60 * 60   # 7 days
CACHE_KEY_PREFIX      = "researchflow:pubmed:"

RETRY_ATTEMPTS   = 4
RETRY_BASE_DELAY = 10       # seconds; doubles each attempt (exponential backoff)
RETRY_MAX_DELAY  = 120      # ceiling so we never wait absurdly long

# NCBI rate limits: 3/s without key, 10/s with key
_NCBI_API_KEY         = os.environ.get("NCBI_API_KEY", "")
_RATE_LIMIT_RPS       = 10.0 if _NCBI_API_KEY else 3.0
_MIN_INTERVAL_SECONDS = 1.0 / _RATE_LIMIT_RPS

# NCBI policy: all automated tools must identify themselves via tool + email.
# https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requirements
_NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "researchflow@example.com")

# Base params appended to every NCBI call
_BASE_PARAMS: dict[str, str] = {
    "db":      "pubmed",
    "retmode": "json",
    "tool":    "ResearchFlow",   # identifies this application to NCBI
    "email":   _NCBI_EMAIL,      # contact address; set NCBI_EMAIL in env for production
}
if _NCBI_API_KEY:
    _BASE_PARAMS["api_key"] = _NCBI_API_KEY

# ---------------------------------------------------------------------------
# Token-bucket rate limiter  (identical design to semantic_scholar.py)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe-enough (single-threaded asyncio) token bucket."""

    def __init__(self, rate: float) -> None:
        self._rate      = rate          # tokens per second
        self._tokens    = 1.0
        self._last_check = time.monotonic()

    def consume(self) -> float:
        """Return seconds to sleep *before* the next request may fire."""
        now     = time.monotonic()
        elapsed = now - self._last_check
        self._last_check = now
        self._tokens = min(1.0, self._tokens + elapsed * self._rate)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0

        wait = (1.0 - self._tokens) / self._rate
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
    mesh_terms: Optional[str],
    limit: int,
) -> str:
    """Deterministic, prefixed cache key derived from all search parameters."""
    raw    = f"{query}|{year_start}|{year_end}|{mesh_terms}|{limit}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


def _cache_read(key: str) -> Optional[list[dict]]:
    if _redis is not None:
        try:
            cached = _redis.get(key)
            if cached is not None:
                return json.loads(cached)
        except Exception as exc:
            logger.warning("Redis GET failed (%s: %s). Proceeding to API.", type(exc).__name__, exc)
    elif key in _fallback_cache:
        return _fallback_cache[key]
    return None


def _cache_write(key: str, results: list[dict]) -> None:
    if _redis is not None:
        try:
            _redis.setex(key, CACHE_TTL_SECONDS, json.dumps(results))
            logger.debug("Wrote %d papers to Redis (key=%s)", len(results), key)
        except Exception as exc:
            logger.warning("Redis SET failed (%s: %s). Result not cached.", type(exc).__name__, exc)
    else:
        _fallback_cache[key] = results


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class PubMedInput(BaseModel):
    query: str = Field(
        description=(
            "PubMed search query. Supports full PubMed query syntax, including field tags "
            "such as '[Title]', '[MeSH Terms]', '[Author]', and Boolean operators "
            "(AND, OR, NOT). Plain natural language queries also work. "
            "Example: 'CRISPR gene editing cancer therapy' or "
            "'\"machine learning\"[Title] AND diabetes[MeSH Terms]'"
        )
    )
    year_start: Optional[int] = Field(
        default=None,
        description="Filter papers published from this year (inclusive), e.g. 2018.",
    )
    year_end: Optional[int] = Field(
        default=None,
        description="Filter papers published up to this year (inclusive), e.g. 2024.",
    )
    mesh_terms: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated MeSH (Medical Subject Heading) terms to AND-inject into "
            "the query, narrowing results to specific biomedical concepts. "
            "Example: 'Neoplasms,Immunotherapy' filters to papers indexed under both. "
            "Leave None for broad queries that should not be MeSH-restricted."
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

def _build_query(inp: PubMedInput) -> str:
    """
    Compose the final PubMed query string.

    MeSH terms are AND-appended using the '[MeSH Terms]' field tag so they act
    as a controlled-vocabulary filter on top of the free-text query.
    Multiple MeSH terms are AND-combined — a paper must be indexed under *all*
    provided headings, which is the correct semantics for domain narrowing.
    """
    query = inp.query.strip()

    if inp.mesh_terms:
        mesh_clauses = [
            f'"{m.strip()}"[MeSH Terms]'
            for m in inp.mesh_terms.split(",")
            if m.strip()
        ]
        if mesh_clauses:
            query = f"({query}) AND {' AND '.join(mesh_clauses)}"

    return query


def _build_date_params(year_start: Optional[int], year_end: Optional[int]) -> dict:
    """
    Convert optional year bounds into NCBI datetype/mindate/maxdate parameters.
    NCBI expects dates in YYYY/MM/DD format; we use /01/01 and /12/31 as bounds.
    """
    params: dict = {}
    if year_start or year_end:
        params["datetype"] = "pdat"          # publication date
        params["mindate"]  = f"{year_start}/01/01" if year_start else "1000/01/01"
        params["maxdate"]  = f"{year_end}/12/31"   if year_end   else "3000/12/31"
    return params


# ---------------------------------------------------------------------------
# HTTP layer — rate-limited with exponential back-off on 429
# ---------------------------------------------------------------------------

def _throttled_get(url: str, params: dict) -> httpx.Response:
    """
    Issue a single rate-limited GET to an NCBI endpoint.
    Retries up to RETRY_ATTEMPTS times on HTTP 429 with exponential back-off,
    honouring the Retry-After header when the server provides it.
    """
    wait = _bucket.consume()
    if wait > 0:
        logger.debug("Rate limiting: sleeping %.2fs before PubMed request", wait)
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
                raise                       # non-429 errors are not retried

            last_exc = exc
            if attempt == RETRY_ATTEMPTS:
                break

            retry_after = exc.response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)
                logger.warning(
                    "PubMed 429 (attempt %d/%d). Server asked to wait %ds.",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            else:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                logger.warning(
                    "PubMed 429 (attempt %d/%d). Backing off %ds (exponential).",
                    attempt, RETRY_ATTEMPTS, delay,
                )
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Step 1 — esearch: query → PMIDs
# ---------------------------------------------------------------------------

def _esearch(query: str, limit: int,
             year_start: Optional[int], year_end: Optional[int]) -> list[str]:
    """
    Call the NCBI esearch endpoint and return a list of PMIDs (strings).
    Returns at most `limit` IDs — NCBI already ranks by relevance.
    """
    params: dict = {
        **_BASE_PARAMS,
        "term":    query,
        "retmax":  str(limit),
        "retmode": "json",
        "usehistory": "n",              # no server-side history needed — PMIDs are small
        **_build_date_params(year_start, year_end),
    }

    response = _throttled_get(ESEARCH_URL, params)
    data     = response.json()
    pmids    = data.get("esearchresult", {}).get("idlist", [])

    logger.debug("esearch returned %d PMIDs for query=%r", len(pmids), query)
    return pmids


# ---------------------------------------------------------------------------
# Step 2 — efetch: PMIDs → XML → PaperResult dicts
# ---------------------------------------------------------------------------

def _efetch(pmids: list[str]) -> list[dict]:
    """
    Fetch full PubMed XML records for a batch of PMIDs and parse them
    into ResearchFlow PaperResult dicts.
    """
    params: dict = {
        **_BASE_PARAMS,
        "id":      ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    # efetch returns XML, not JSON — override the base retmode
    params["retmode"] = "xml"

    response = _throttled_get(EFETCH_URL, params)
    return _parse_xml(response.text)


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

def _text(element: Optional[ET.Element]) -> str:
    """Safely extract text from an Element, stripping whitespace."""
    if element is None:
        return ""
    return (element.text or "").strip()


def _parse_abstract(article: ET.Element) -> str:
    """
    Handle both flat and structured (BACKGROUND / METHODS / RESULTS / …)
    PubMed abstracts. Structured abstracts have multiple <AbstractText> nodes
    each with a Label attribute; we concatenate them with a label prefix.
    """
    abstract_el = article.find("Abstract")
    if abstract_el is None:
        return "No abstract available."

    parts: list[str] = []
    for node in abstract_el.findall("AbstractText"):
        label = node.get("Label", "")
        text  = (node.text or "").strip()
        if text:
            parts.append(f"{label}: {text}" if label else text)

    return " ".join(parts) if parts else "No abstract available."


def _parse_year(article: ET.Element) -> Optional[int]:
    """
    Extract publication year, trying three progressively fallback locations
    that NCBI uses for different article types:
      1. <PubDate><Year>                   (most journals)
      2. <PubDate><MedlineDate>            (older records, e.g. "2003 Jan-Feb")
      3. <ArticleDate DateType="Electronic"><Year>  (epub-ahead-of-print)
    """
    pub_date = article.find(".//PubDate")
    if pub_date is not None:
        year_el = pub_date.find("Year")
        if year_el is not None and year_el.text:
            try:
                return int(year_el.text.strip())
            except ValueError:
                pass

        medline = pub_date.find("MedlineDate")
        if medline is not None and medline.text:
            # e.g. "2003 Jan-Feb" or "2003 Spring"
            first_token = medline.text.strip().split()[0]
            try:
                return int(first_token)
            except ValueError:
                pass

    article_date = article.find(".//ArticleDate[@DateType='Electronic']")
    if article_date is not None:
        year_el = article_date.find("Year")
        if year_el is not None and year_el.text:
            try:
                return int(year_el.text.strip())
            except ValueError:
                pass

    return None


def _parse_authors(article: ET.Element) -> list[str]:
    """Return a list of "ForeName LastName" strings (or just LastName if no ForeName)."""
    authors: list[str] = []
    for author in article.findall(".//Author"):
        last  = _text(author.find("LastName"))
        first = _text(author.find("ForeName"))
        if last:
            authors.append(f"{first} {last}".strip() if first else last)
        else:
            # CollectiveName (e.g. consortia)
            collective = _text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
    return authors


def _parse_xml(xml_text: str) -> list[dict]:
    """
    Parse a PubMedArticleSet XML response into a list of PaperResult dicts.
    Silently skips malformed article nodes rather than aborting the entire batch.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse PubMed XML: %s", exc)
        return []

    papers: list[dict] = []

    for article_node in root.findall(".//PubmedArticle"):
        try:
            medline  = article_node.find("MedlineCitation")
            if medline is None:
                continue

            pmid_el  = medline.find("PMID")
            pmid     = _text(pmid_el) if pmid_el is not None else None

            article  = medline.find("Article")
            if article is None:
                continue

            title    = _text(article.find("ArticleTitle"))
            abstract = _parse_abstract(article)
            authors  = _parse_authors(article)
            year     = _parse_year(article)

            # Journal
            journal_el = article.find("Journal/Title")
            journal    = _text(journal_el) if journal_el is not None else None

            # MeSH terms  (only present after MEDLINE indexing, usually 4–8 weeks post-pub)
            mesh_terms: list[str] = [
                _text(mh.find("DescriptorName"))
                for mh in medline.findall(".//MeshHeading")
                if mh.find("DescriptorName") is not None
            ]

            # DOI and PMC ID from PubmedData
            doi   = None
            pmc   = None
            for id_el in article_node.findall(".//ArticleId"):
                id_type = id_el.get("IdType", "")
                val     = (id_el.text or "").strip()
                if id_type == "doi"  and val:
                    doi = val
                elif id_type == "pmc" and val:
                    pmc = val

            # Canonical URL — prefer free PMC full-text, fall back to PubMed abstract
            if pmc:
                url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/"
            elif pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            else:
                url = f"https://doi.org/{doi}" if doi else None

            papers.append({
                # ---- Universal ResearchFlow schema ----
                "title":          title or "Unknown Title",
                "abstract":       abstract,
                "authors":        authors,
                "year":           year,
                "citation_count": None,     # PubMed does not expose citation counts
                "doi":            doi,
                "arxiv_id":       None,     # PubMed papers are not arXiv preprints
                "url":            url,
                "source":         "PubMed",
                # ---- PubMed-specific extras ----
                "pmid":           pmid,
                "mesh_terms":     mesh_terms,
                "journal":        journal,
            })

        except Exception as exc:   # noqa: BLE001
            logger.warning("Skipping malformed PubMed article node: %s", exc)
            continue

    return papers


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search_pubmed(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    mesh_terms: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Synchronous entry point used by the LangChain StructuredTool.

    Executes a two-step NCBI E-utilities call (esearch → efetch) and
    returns a list of PaperResult dicts, or raises on unrecoverable errors.
    Results are cached in Redis (or the fallback dict) to avoid redundant API hits.
    """
    inp = PubMedInput(
        query=query,
        year_start=year_start,
        year_end=year_end,
        mesh_terms=mesh_terms,
        limit=min(limit, MAX_RESULTS_CAP),
    )

    key = _cache_key(inp.query, inp.year_start, inp.year_end, inp.mesh_terms, inp.limit)

    # --- Cache READ ---
    cached = _cache_read(key)
    if cached is not None:
        logger.debug("Cache hit for PubMed query=%r", inp.query)
        return cached

    # --- Build the final query string ---
    final_query = _build_query(inp)
    logger.debug("PubMed final query: %r", final_query)

    # --- Step 1: esearch → PMIDs ---
    pmids = _esearch(final_query, inp.limit, inp.year_start, inp.year_end)

    if not pmids:
        logger.info("PubMed esearch returned 0 PMIDs for query=%r", inp.query)
        return []

    # --- Step 2: efetch → full records ---
    results = _efetch(pmids)

    # Trim to requested limit (efetch may return slightly different counts)
    results = results[: inp.limit]

    logger.info(
        "PubMed returned %d papers (fetched %d PMIDs) for query=%r",
        len(results), len(pmids), inp.query,
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
    mesh_terms: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    String-formatted wrapper seen by the ReAct agent's LLM.
    Downstream pipeline code should call search_pubmed() directly for dicts.
    """
    papers = search_pubmed(
        query=query,
        year_start=year_start,
        year_end=year_end,
        mesh_terms=mesh_terms,
        limit=limit,
    )

    if not papers:
        return f"No results found on PubMed for query: '{query}'"

    lines = [f"PubMed results for '{query}':\n"]
    for i, p in enumerate(papers, start=1):
        doi_str = f"DOI: {p['doi']}" if p["doi"] else "DOI: N/A"
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += f" et al. (+{len(p['authors']) - 3} more)"

        mesh_str = ", ".join(p["mesh_terms"][:5]) if p["mesh_terms"] else "N/A"
        abstract_snippet = p["abstract"][:300].strip()
        if len(p["abstract"]) > 300:
            abstract_snippet += "..."

        lines.append(
            f"[{i}] {p['title']} ({p['year']})\n"
            f"    Authors: {authors_str}\n"
            f"    Journal: {p['journal'] or 'N/A'} | PMID: {p['pmid'] or 'N/A'} | {doi_str}\n"
            f"    MeSH: {mesh_str}\n"
            f"    URL: {p['url'] or 'N/A'}\n"
            f"    Abstract: {abstract_snippet}\n"
        )

    return "\n".join(lines)


pubmed_tool = StructuredTool.from_function(
    func=_tool_fn,
    name="pubmed_search",
    description=(
        "Search PubMed/MEDLINE (35M+ biomedical and life sciences articles). "
        "Best for clinical medicine, pharmacology, genomics, neuroscience, public health, "
        "and any query involving human disease, biology, or healthcare. "
        "Supports optional MeSH term filters for controlled-vocabulary narrowing, "
        "and date range filters. "
        "Returns title, authors, abstract, journal, PMID, DOI, MeSH terms, and URL."
    ),
    args_schema=PubMedInput,
    return_direct=False,
)


# ---------------------------------------------------------------------------
# Cache management helpers
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """
    Delete all ResearchFlow PubMed cache entries.
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

    print("=== Direct function call (plain query) ===")
    results = search_pubmed(
        query="CRISPR gene editing cancer immunotherapy",
        year_start=2021,
        limit=3,
    )
    for r in results:
        print(
            f"- {r['title']} ({r['year']}) | "
            f"pmid={r['pmid']} | doi={r['doi']} | journal={r['journal']}"
        )

    print("\n=== With MeSH term filter ===")
    mesh_results = search_pubmed(
        query="machine learning diagnosis",
        mesh_terms="Neoplasms,Deep Learning",
        limit=2,
    )
    for r in mesh_results:
        print(f"- {r['title']} | mesh={r['mesh_terms'][:3]}")

    print("\n=== Via LangChain tool (agent string output) ===")
    output = pubmed_tool.invoke({
        "query": "CRISPR gene editing cancer immunotherapy",
        "year_start": 2021,
        "limit": 3,
    })
    print(output)

    print("\n=== Cache hit test (should not hit API) ===")
    results2 = search_pubmed(
        query="CRISPR gene editing cancer immunotherapy",
        year_start=2021,
        limit=3,
    )
    assert results == results2, "Cache miss — something is wrong"
    print("Cache hit confirmed ✓")
    print(f"Cache stats: {cache_stats()}")
