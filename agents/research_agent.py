"""
agents/research_agent.py
────────────────────────
ResearchFlow's autonomous research orchestrator.

Two public classes:

  ResilientLLM
      A LangChain-compatible Runnable that tries up to three Groq API keys in
      sequence, catching 429 / quota errors and immediately falling back to the
      next key.  If all Groq keys are exhausted it falls over to Gemini
      (gemini-2.0-flash) with all safety filters disabled so academic content
      is never blocked.

  ResearchAgent
      Async orchestrator.  Calls QueryPlanner → parallel tool fan-out →
      global deduplication → Unpaywall PDF enrichment → SynthesisChain.

Domain detection
────────────────
Domain classification (cs / biomedical / general) is handled entirely by
``core.filter.detect_domain_semantic`` using the shared embedding model.
No keyword arrays are maintained here.

Environment variables
─────────────────────
  GROQ_KEY_1, GROQ_KEY_2, GROQ_KEY_3  — Groq keys tried in this order
  GOOGLE_API_KEY                        — Gemini fallback
  S2_API_KEY                            — Semantic Scholar Graph API key (optional;
                                          unauthenticated access is rate-limited to
                                          1 req/s; an API key raises this to 10 req/s)

Usage (programmatic)
────────────────────
  from agents.research_agent import build_from_env

  agent = build_from_env()
  report = asyncio.run(agent.research("What are the latest advances in RAG?"))
  print(report)

Usage (CLI)
───────────
  python -m agents.research_agent "What are the latest advances in RAG?"
"""

from __future__ import annotations

import asyncio
import logging
import os
from difflib import SequenceMatcher
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig

# Domain detection — single shared function, no keyword arrays in this file
from core.filter import detect_domain_semantic  # type: ignore[import]

# Chains
from chains.query_planner import QueryPlanner, ResearchPlan, SearchStep  # type: ignore[import]
from chains.synthesizer import SynthesisChain

logger = logging.getLogger(__name__)

def _sanitise_query(query: str) -> str:
    """
    Normalise special characters in a query string before sending to an LLM.
    Groq's structured output JSON serialiser chokes on apostrophes, smart
    quotes, and other non-ASCII punctuation when they appear inside
    generated field values. Replacing them here prevents the LLM from
    echoing them into search_query strings.
    """
    replacements = {
        "\u2019": "",   # right single quotation mark (Parkinson's → Parkinsons)
        "\u2018": "",   # left single quotation mark
        "\u0060": "",   # grave accent
        "\u00b4": "",   # acute accent
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",# ellipsis
        "'": "",        # plain ASCII apostrophe in possessives
    }
    for char, replacement in replacements.items():
        query = query.replace(char, replacement)
    return query


# ──────────────────────────────────────────────────────────────────────────────
# Rate-limit / quota error detection
# ──────────────────────────────────────────────────────────────────────────────

_RATE_LIMIT_SIGNALS: tuple[str, ...] = (
    "rate limit",
    "ratelimit",
    "429",
    "too many requests",
    "quota",
    "resource_exhausted",
    "resource exhausted",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "tool_use_failed" in msg or "failed_generation" in msg:
        return True
    
    try:
        import groq
        if isinstance(exc, groq.RateLimitError):
            return True
        if isinstance(exc, groq.APIStatusError) and getattr(exc, "status_code", None) == 429:
            return True
    except ImportError:
        pass

    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            return True
    except ImportError:
        pass

    msg = str(exc).lower()
    return any(signal in msg for signal in _RATE_LIMIT_SIGNALS)


# ──────────────────────────────────────────────────────────────────────────────
# ResilientLLM
# ──────────────────────────────────────────────────────────────────────────────

class ResilientLLM(Runnable):
    """
    LangChain-compatible Runnable with ordered multi-provider fallback.

    Priority order (left → right):
        Groq key 1  →  Groq key 2  →  Groq key 3  →  Gemini


    A 429 / quota error on any provider causes an immediate retry on the
    next provider in the chain. All other errors are re-raised to the caller
    so bugs surface clearly.

    Parameters
    ----------
    groq_keys:
        Ordered list of Groq API keys. Pass 1–3 keys.
    gemini_key:
        Google AI Studio key. Used only if every Groq key is exhausted.
    groq_model:
        Groq model name (default: llama-3.3-70b-versatile).
    gemini_model:
        Gemini model name (default: gemini-2.0-flash).
    temperature:
        Shared temperature applied to every provider.
    """

    def __init__(
        self,
        groq_keys: list[str],
        gemini_key: str = "",
        groq_model: str = "llama-3.3-70b-versatile",
        gemini_model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
    ) -> None:
        if not groq_keys:
            raise ValueError("At least one Groq API key is required.")
        self._groq_keys    = list(groq_keys)
        self._gemini_key   = gemini_key
        self._groq_model   = groq_model
        self._gemini_model = gemini_model
        self._temperature  = temperature
        self._schema: Any  = None

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "ResilientLLM":
        """Mirror BaseChatModel.with_structured_output for transparent drop-in use."""
        clone = ResilientLLM(
            groq_keys    = self._groq_keys,
            gemini_key   = self._gemini_key,
            groq_model   = self._groq_model,
            gemini_model = self._gemini_model,
            temperature  = self._temperature,
        )
        clone._schema = schema
        return clone

    def _make_groq(self, api_key: str) -> Runnable:
        from langchain_groq import ChatGroq  # type: ignore[import]
        return ChatGroq(
            api_key     = api_key,
            model       = self._groq_model,
            temperature = self._temperature,
        )

    def _make_gemini(self) -> Runnable:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=self._gemini_key,
            model=self._gemini_model,
            temperature=self._temperature,
            safety_settings=_build_gemini_safety_settings(),
            convert_system_message_to_human=True,
            max_retries=0,
    )

    def _iter_providers(self):
        """
        Yield (label, llm) pairs in priority order using a generator so each
        provider is constructed only when it is about to be tried.
        Order: Groq[1] → Groq[2] → Groq[3] → Gemini
        All Groq keys are tried first before falling back to Gemini.
        Groq keys are a last-resort safety net.
        """
        def _wrap(llm: Runnable) -> Runnable:
            return llm.with_structured_output(self._schema) if self._schema is not None else llm

        for i, key in enumerate(self._groq_keys, start=1):
            yield f"groq[{i}]", _wrap(self._make_groq(key))

        if self._gemini_key:
            yield "gemini", _wrap(self._make_gemini())

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        last_rate_exc: BaseException | None = None
        for label, llm in self._iter_providers():
            try:
                logger.debug("ResilientLLM.invoke: trying %s", label)
                result = llm.invoke(input, config, **kwargs)
                logger.debug("ResilientLLM.invoke: success with %s", label)
                return result
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    logger.warning("ResilientLLM: %s rate-limited — falling back", label)
                    last_rate_exc = exc
                    continue
                raise
        raise RuntimeError("ResilientLLM: all providers exhausted by rate limits.") from last_rate_exc

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        last_rate_exc: BaseException | None = None
        for label, llm in self._iter_providers():
            try:
                logger.debug("ResilientLLM.ainvoke: trying %s", label)
                result = await llm.ainvoke(input, config, **kwargs)
                logger.debug("ResilientLLM.ainvoke: success with %s", label)
                return result
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    logger.warning("ResilientLLM: %s rate-limited — falling back", label)
                    last_rate_exc = exc
                    continue
                raise
        raise RuntimeError("ResilientLLM: all providers exhausted by rate limits.") from last_rate_exc


def _build_gemini_safety_settings() -> dict:
    return {
        "HARM_CATEGORY_HARASSMENT":        "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH":       "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool routing
# ──────────────────────────────────────────────────────────────────────────────

def _tools_for_domain(
    domain: str,
    arxiv: Any,
    semantic_scholar: Any,
    pubmed: Any,
    openalex: Any,
) -> list[tuple[str, Any]]:
    """
    Return an ordered [(name, tool)] list for domain.

    Domain classification comes from detect_domain_semantic() which returns
    "general" for cross-domain queries — those get full four-tool fan-out.

    ┌──────────────────┬──────────────────────────────────────────────┐
    │ Domain           │ Tools                                         │
    ├──────────────────┼──────────────────────────────────────────────┤
    │ cs               │ arXiv, Semantic Scholar, OpenAlex             │
    │ biomedical       │ PubMed, Semantic Scholar, OpenAlex            │
    │ general          │ arXiv, Semantic Scholar, PubMed, OpenAlex    │
    └──────────────────┴──────────────────────────────────────────────┘

    "general" covers both explicitly general queries AND cross-domain queries
    (e.g. "neuro-symbolic AI in medical field") that detect_domain_semantic
    classifies as "general" due to the similarity gap being below threshold.
    """
    if domain == "biomedical":
        return [
            ("pubmed",            pubmed),
            ("semantic_scholar",  semantic_scholar),
            ("openalex",          openalex),
        ]
    if domain == "cs":
        return [
            ("arxiv",             arxiv),
            ("semantic_scholar",  semantic_scholar),
            ("openalex",          openalex),
        ]
    # "general" — full fan-out (cross-domain or unknown)
    return [
        ("arxiv",             arxiv),
        ("semantic_scholar",  semantic_scholar),
        ("pubmed",            pubmed),
        ("openalex",          openalex),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────────────────────────

def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not None. Avoids Python's `0 or x` pitfall."""
    for v in values:
        if v is not None:
            return v
    return None


def _titles_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Return True when two title strings are likely the same paper."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= threshold


def _dedup_key(paper: dict[str, Any]) -> str:
    """
    Canonical identity key for a paper.

    Priority:
      1. arXiv ID  — stable cross-source identifier linking preprint + published.
      2. Normalised non-arXiv DOI.
      3. title: + stripped lowercase title — fallback when no DOI exists.
    """
    import re

    # ── Step 1: extract arXiv ID ───────────────────────────────────────────────
    arxiv_id: str = ""

    raw_arxiv = (paper.get("arxiv_id") or "").strip().lower()
    if raw_arxiv:
        arxiv_id = raw_arxiv

    if not arxiv_id:
        doi_str = (paper.get("doi") or "").strip().lower()
        m = re.search(r"10\.48550/arxiv\.(\S+)", doi_str)
        if m:
            arxiv_id = m.group(1).rstrip(".")

    if not arxiv_id:
        url_str = (paper.get("url") or "").strip().lower()
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)", url_str)
        if m:
            arxiv_id = m.group(1)

    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    # ── Step 2: normalised non-arXiv DOI ──────────────────────────────────────
    doi = (paper.get("doi") or "").strip().lower()
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    if doi and not doi.startswith("10.48550"):
        return doi

    # ── Step 3: normalised title fallback ─────────────────────────────────────
    title = (paper.get("title") or "").strip().lower()
    title = " ".join(title.split())
    return f"title:{title}"


def _merge_papers(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """
    Choose which copy of a duplicate paper to keep.
    Prefers higher citation count, then longer abstract.
    """
    existing_cit = existing.get("citation_count") or 0
    incoming_cit = incoming.get("citation_count") or 0

    if incoming_cit > existing_cit:
        return incoming
    if incoming_cit == existing_cit:
        if len(incoming.get("abstract") or "") > len(existing.get("abstract") or ""):
            return incoming
    return existing


def deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge papers from multiple API sources into a deduplicated list.

    Two-pass strategy
    ─────────────────
    Pass 1 — exact key match (DOI / arXiv ID / normalised title).
              Fast O(n) dict lookup. Handles most duplicates.

    Pass 2 — fuzzy title similarity (SequenceMatcher ratio > 0.85).
              Catches cases where the same paper arrives from two sources
              with different metadata completeness (one has a DOI, one
              doesn't) and would survive Pass 1 with different keys.
    """
    # ── Pass 1: exact key dedup ───────────────────────────────────────────────
    seen: dict[str, dict[str, Any]] = {}
    for paper in papers:
        key = _dedup_key(paper)
        if key not in seen:
            seen[key] = paper
        else:
            seen[key] = _merge_papers(seen[key], paper)

    # ── Pass 2: fuzzy title dedup ─────────────────────────────────────────────
    # Iterates the pass-1 survivors and collapses near-identical titles.
    # O(n²) but n is at most ~50 papers so the cost is negligible (<1 ms).
    unique: list[dict[str, Any]] = []
    for candidate in seen.values():
        c_title = (candidate.get("title") or "").lower().strip()
        matched = False
        for i, kept in enumerate(unique):
            k_title = (kept.get("title") or "").lower().strip()
            if _titles_similar(c_title, k_title):
                unique[i] = _merge_papers(kept, candidate)
                matched = True
                break
        if not matched:
            unique.append(candidate)

    return unique


# ──────────────────────────────────────────────────────────────────────────────
# Tool-output normaliser — helpers
# ──────────────────────────────────────────────────────────────────────────────

def _reconstruct_inverted_index(index: dict) -> str:
    """
    Reassemble an OpenAlex abstract_inverted_index into a plain string.

    OpenAlex stores abstracts as {"word": [position, ...], ...}.
    This reverses that mapping and joins words in position order.
    """
    if not index or not isinstance(index, dict):
        return ""

    max_pos: int = -1
    for positions in index.values():
        if isinstance(positions, list):
            for p in positions:
                if isinstance(p, int) and p > max_pos:
                    max_pos = p

    if max_pos < 0:
        return ""

    slots: list[str] = [""] * (max_pos + 1)
    for word, positions in index.items():
        if isinstance(positions, list):
            for p in positions:
                if isinstance(p, int) and 0 <= p <= max_pos:
                    slots[p] = str(word)

    return " ".join(w for w in slots if w)


def _normalise_dict(raw: dict) -> list[dict[str, Any]]:
    """
    Normalise a single paper dict from any API into the standard schema.

    Uses _first_not_none() for all multi-key lookups to avoid the Python
    truthiness pitfall where citation_count=0 would be treated as "absent".
    """
    # ── title ─────────────────────────────────────────────────────────────────
    title: str = (
        raw.get("title")
        or raw.get("display_name")
        or raw.get("name")
        or ""
    )

    # ── abstract ──────────────────────────────────────────────────────────────
    abstract: str = raw.get("abstract") or raw.get("summary") or ""
    if not abstract:
        aii = raw.get("abstract_inverted_index")
        if isinstance(aii, dict):
            abstract = _reconstruct_inverted_index(aii)

    if not title and not abstract:
        return []

    # ── authors ───────────────────────────────────────────────────────────────
    raw_authors = raw.get("authors") or raw.get("authorships") or []
    authors: list[str] = []
    for a in raw_authors:
        if isinstance(a, str):
            authors.append(a)
        elif isinstance(a, dict):
            name = (
                a.get("name")
                or (a.get("author") or {}).get("display_name")
                or a.get("display_name")
            )
            if name:
                authors.append(str(name))

    # ── year ──────────────────────────────────────────────────────────────────
    year: int | None = _first_not_none(
        raw.get("year"),
        raw.get("publication_year"),
    )
    if year is None:
        published = raw.get("published") or raw.get("publishedDate") or ""
        if published and str(published)[:4].isdigit():
            year = int(str(published)[:4])
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    # ── citation count ────────────────────────────────────────────────────────
    # Uses _first_not_none so that citation_count=0 is preserved correctly.
    # The original `raw.get("a") or raw.get("b")` pattern loses 0 values
    # because `0 or x` evaluates to x in Python.
    citation_count: int | None = _first_not_none(
        raw.get("citation_count"),
        raw.get("cited_by_count"),
        raw.get("citationCount"),
    )

    # ── doi ───────────────────────────────────────────────────────────────────
    doi: str | None = raw.get("doi") or raw.get("DOI")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # ── url ───────────────────────────────────────────────────────────────────
    url: str = (
        raw.get("url")
        or raw.get("URL")
        or raw.get("entry_id")
        or raw.get("landing_page_url")
        or (f"https://doi.org/{doi}" if doi else "")
    )

    primary_location = raw.get("primary_location")
    if isinstance(primary_location, dict) and not url:
        url = primary_location.get("landing_page_url") or ""

    entry: dict[str, Any] = {"title": title, "abstract": abstract}
    if authors:
        entry["authors"] = authors
    if year is not None:
        entry["year"] = year
    if citation_count is not None:
        entry["citation_count"] = citation_count
    if doi:
        entry["doi"] = doi
    if url:
        entry["url"] = url

    for extra in ("source", "arxiv_id", "fields_of_study", "tldr", "pdf_url"):
        if raw.get(extra) is not None:
            entry[extra] = raw[extra]

    return [entry]


def _parse_formatted_string(text: str) -> list[dict[str, Any]]:
    """
    Parse a human-readable multi-paper string returned by a LangChain tool's
    .run() method back into a list of normalised paper dicts.

    Handles Format A (numbered [N] blocks) and Format B (Key: value blocks).
    """
    papers: list[dict[str, Any]] = []
    text = text.strip()

    import re

    blocks_a = re.split(r"\n(?=\[\d+\])", text)

    if len(blocks_a) > 1 or re.match(r"^\[\d+\]", blocks_a[0].strip()):
        for block in blocks_a:
            block = block.strip()
            if not block or not re.match(r"^\[\d+\]", block):
                continue

            lines = block.splitlines()
            first_line = re.sub(r"^\[\d+\]\s*", "", lines[0]).strip()

            year_match = re.search(r"\((\d{4})\)\s*$", first_line)
            year: int | None = int(year_match.group(1)) if year_match else None
            title = re.sub(r"\s*\(\d{4}\)\s*$", "", first_line).strip()

            entry: dict[str, Any] = {"title": title or "Untitled"}
            if year:
                entry["year"] = year

            for line in lines[1:]:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                if "abstract" in entry and not re.match(r"^[A-Za-z ]+:", line_stripped):
                    entry["abstract"] = entry["abstract"] + " " + line_stripped
                    continue

                if line_stripped.lower().startswith("abstract:"):
                    val = line_stripped[len("abstract:"):].strip()
                    if val.endswith("..."):
                        val = val[:-3].strip()
                    entry["abstract"] = val

                elif line_stripped.lower().startswith("authors:"):
                    raw_authors = line_stripped[len("authors:"):].strip()
                    raw_authors = re.sub(r"\s+et al\.\s*\(\+\d+ more\)", "", raw_authors)
                    entry["authors"] = [a.strip() for a in raw_authors.split(",") if a.strip()]

                elif line_stripped.lower().startswith("url:"):
                    entry["url"] = line_stripped[4:].strip()

                elif "doi:" in line_stripped.lower():
                    doi_match = re.search(r"DOI:\s*(\S+)", line_stripped, re.IGNORECASE)
                    if doi_match and doi_match.group(1).upper() != "N/A":
                        entry["doi"] = doi_match.group(1)

                elif "citations:" in line_stripped.lower():
                    cit_match = re.search(r"Citations:\s*(\d+)", line_stripped, re.IGNORECASE)
                    if cit_match:
                        entry["citation_count"] = int(cit_match.group(1))

            if entry.get("title") or entry.get("abstract"):
                papers.append(entry)

        if papers:
            return papers

    blocks_b = re.split(r"\n{2,}", text)
    for block in blocks_b:
        block = block.strip()
        if not block:
            continue

        entry = {}
        current_key: str | None = None
        current_val: list[str] = []

        def _flush() -> None:
            if current_key and current_val:
                entry[current_key] = " ".join(current_val).strip()

        for line in block.splitlines():
            kv_match = re.match(r"^([A-Za-z_][A-Za-z_ ]*):\s*(.*)", line)
            if kv_match:
                _flush()
                current_key = kv_match.group(1).strip().lower().replace(" ", "_")
                current_val = [kv_match.group(2).strip()]
            else:
                current_val.append(line.strip())

        _flush()

        if not entry:
            continue

        title   = entry.get("title") or entry.get("display_name") or ""
        abstract = entry.get("abstract") or entry.get("summary") or ""
        if not title and not abstract:
            continue

        paper: dict[str, Any] = {"title": title, "abstract": abstract}
        if entry.get("authors"):
            paper["authors"] = [a.strip() for a in entry["authors"].split(",") if a.strip()]
        if entry.get("year") and str(entry["year"]).isdigit():
            paper["year"] = int(entry["year"])
        if entry.get("doi") and entry["doi"].upper() != "N/A":
            paper["doi"] = entry["doi"]
        if entry.get("url") and entry["url"].upper() not in ("N/A", "URL: N/A"):
            paper["url"] = entry["url"]
        if entry.get("citation_count") and str(entry["citation_count"]).isdigit():
            paper["citation_count"] = int(entry["citation_count"])

        papers.append(paper)

    return papers


# ──────────────────────────────────────────────────────────────────────────────
# Tool-output normaliser
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_tool_output(raw: Any) -> list[dict[str, Any]]:
    """
    Aggressively coerce any tool return value into a list[dict] using the
    standard paper schema (title, abstract, authors, year, doi, url,
    citation_count).
    """
    if not raw:
        return []

    if isinstance(raw, list):
        results: list[dict[str, Any]] = []
        for item in raw:
            results.extend(_normalise_tool_output(item))
        return results

    if isinstance(raw, dict):
        return _normalise_dict(raw)

    if hasattr(raw, "page_content"):
        meta: dict = getattr(raw, "metadata", {}) or {}
        abstract = getattr(raw, "page_content", "") or ""
        if not abstract:
            aii = meta.get("abstract_inverted_index")
            if isinstance(aii, dict):
                abstract = _reconstruct_inverted_index(aii)

        entry: dict[str, Any] = {
            "title": (
                meta.get("Title") or meta.get("title")
                or meta.get("display_name") or "Untitled"
            ),
            "abstract": abstract,
            "authors": meta.get("Authors") or meta.get("authors") or [],
            "year": _first_not_none(
                meta.get("Published"), meta.get("year"), meta.get("publication_year")
            ),
            "doi":  meta.get("DOI") or meta.get("doi"),
            "url":  meta.get("URL") or meta.get("url") or meta.get("entry_id") or "",
            "citation_count": _first_not_none(
                meta.get("citationCount"),
                meta.get("citation_count"),
                meta.get("cited_by_count"),
            ),
        }
        entry = {k: v for k, v in entry.items() if v is not None}
        if entry.get("abstract") or entry.get("title", "Untitled") != "Untitled":
            return [entry]
        return []

    if hasattr(raw, "model_dump"):
        try:
            return _normalise_tool_output(raw.model_dump())
        except Exception:
            pass

    if hasattr(raw, "dict") and callable(getattr(raw, "dict")):
        try:
            return _normalise_tool_output(raw.dict())
        except Exception:
            pass

    if isinstance(raw, str):
        import json

        stripped = raw.strip()
        if not stripped:
            return []

        if stripped.startswith(("[", "{")):
            try:
                return _normalise_tool_output(json.loads(stripped))
            except (json.JSONDecodeError, ValueError):
                pass

        parsed = _parse_formatted_string(stripped)
        if parsed:
            return parsed

        if len(stripped) >= 30:
            snippet = stripped[:50].replace("\n", " ").strip()
            return [{"title": f"Raw: {snippet}…", "abstract": stripped}]
        return []

    try:
        return _normalise_tool_output(vars(raw))
    except TypeError:
        pass

    return []


# ──────────────────────────────────────────────────────────────────────────────
# ResearchAgent
# ──────────────────────────────────────────────────────────────────────────────

class ResearchAgent:
    """
    Full async research orchestrator.

    Five-stage pipeline
    ───────────────────
    1. Plan       — QueryPlanner decomposes the query into SearchStep objects.
    2. Execute    — Tools called concurrently (asyncio.gather) per step.
                    Domain routing uses detect_domain_semantic() — no keyword arrays.
    3. Dedup      — Two-pass deduplication (exact key + fuzzy title similarity).
    4. Enrich     — Unpaywall queried concurrently for open-access PDF URLs.
    5. Synthesise — SynthesisChain produces a cited Markdown report.
    """

    def __init__(
        self,
        llm: Runnable,
        semantic_scholar: Any = None,
        arxiv: Any = None,
        pubmed: Any = None,
        openalex: Any = None,
        unpaywall: Any = None,
        max_papers_per_step: int = 10,
        max_total_papers: int = 40,
        tool_timeout: float = 120.0,
    ) -> None:
        self._llm          = llm
        self._planner      = QueryPlanner(llm=llm)
        self._synthesiser  = SynthesisChain(llm=llm)

        self._semantic_scholar = semantic_scholar or _lazy_import_tool("semantic_scholar")
        self._arxiv            = arxiv            or _lazy_import_tool("arxiv_search")
        self._pubmed           = pubmed           or _lazy_import_tool("pubmed_search")
        self._openalex         = openalex         or _lazy_import_tool("openalex_search")
        self._unpaywall        = unpaywall        or _lazy_import_tool("unpaywall_fetcher")

        self._max_papers_per_step = max_papers_per_step
        self._max_total_papers    = max_total_papers
        self._tool_timeout        = tool_timeout

    # ── Public API ────────────────────────────────────────────────────────────

    async def research(self, query: str) -> str:
        """Run the full five-stage pipeline. Returns a Markdown report string."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        logger.info("ResearchAgent: starting — %r", query)

        plan: ResearchPlan = await self._aplan(query)
        logger.info("ResearchAgent: %d search step(s) planned", len(plan.steps))

        all_papers: list[dict[str, Any]] = []
        for idx, step in enumerate(plan.steps, start=1):
            label = (
                getattr(step, "sub_question", None)
                or getattr(step, "search_query", None)
                or f"step {idx}"
            )
            logger.info("ResearchAgent: step %d/%d — %r", idx, len(plan.steps), label)
            step_papers = await self._execute_step(step)
            all_papers.extend(step_papers)
            logger.info("ResearchAgent: step %d → %d paper(s)", idx, len(step_papers))

        unique = deduplicate(all_papers)

        if not unique:
            logger.warning("ResearchAgent: no papers found across all tools.")
            return (
                "### Research Summary\n\n"
                "No academic papers were found for your query. "
                "Try broadening your search terms or checking tool credentials."
            )
        logger.info(
            "ResearchAgent: %d unique paper(s) (from %d raw)", len(unique), len(all_papers)
        )
        unique = sorted(
            unique, key=lambda p: p.get("citation_count") or 0, reverse=True
        )[: self._max_total_papers]

        unique = await self._enrich_with_pdfs(unique)

        result = await self._synthesiser.arun(query, unique)
        logger.info("ResearchAgent: done — %d paper(s) cited", result.paper_count)
        return result.report

    # ── Stage helpers ─────────────────────────────────────────────────────────

    async def _aplan(self, query: str) -> ResearchPlan:
        query = _sanitise_query(query)
        if hasattr(self._planner, "aplan"):
            return await self._planner.aplan(query)
        return await asyncio.to_thread(self._planner.plan, query)

    async def _execute_step(self, step: Any) -> list[dict[str, Any]]:
        """
        Fan out to domain-selected tools concurrently.

        Domain detection uses detect_domain_semantic() imported from
        core.filter — the same function used by RelevanceFilter — so
        retrieval and filtering always agree on domain classification.

        Stagger: 0.5 s between tools (down from 1.5 s) — sufficient to
        avoid simultaneous bursts while cutting dead time from ~4.5 s to
        ~1.5 s per step.
        """
        sub_question: str = (
            getattr(step, "sub_question", None)
            or getattr(step, "search_query", None)
            or ""
        )

        # detect_domain_semantic returns "general" for cross-domain queries,
        # which maps to full four-tool fan-out in _tools_for_domain.
        domain = getattr(step, "domain", None) or detect_domain_semantic(sub_question)

        selected = _tools_for_domain(
            domain,
            arxiv            = self._arxiv,
            semantic_scholar = self._semantic_scholar,
            pubmed           = self._pubmed,
            openalex         = self._openalex,
        )

        query_text: str = (
            getattr(step, "keywords_string", None)
            or getattr(step, "keywords", None)
            or sub_question
        )
        if isinstance(query_text, list):
            query_text = " ".join(query_text)

        async def _call_tool(name: str, tool: Any, delay: float) -> list[dict[str, Any]]:
            if tool is None:
                return []
            try:
                if delay > 0:
                    await asyncio.sleep(delay)
                raw = await asyncio.wait_for(
                    asyncio.to_thread(tool.run, query_text),
                    timeout=self._tool_timeout,
                )
                papers = _normalise_tool_output(raw)
                capped = papers[: self._max_papers_per_step]
                logger.debug("tool %s → %d paper(s)", name, len(capped))
                return capped
            except asyncio.TimeoutError:
                logger.warning(
                    "ResearchAgent: tool %r timed out (%.0f s) — skipping",
                    name, self._tool_timeout,
                )
                return []
            except Exception as exc:
                logger.warning(
                    "ResearchAgent: tool %r failed — %s: %s",
                    name, type(exc).__name__, exc,
                )
                return []

        # Stagger reduced from 1.5 s → 0.5 s: still polite, saves ~3 s per step
        tasks = [
            _call_tool(name, tool, idx * 0.5)
            for idx, (name, tool) in enumerate(selected)
        ]
        batches: list[list[dict[str, Any]]] = await asyncio.gather(*tasks)

        merged: list[dict[str, Any]] = []
        for batch in batches:
            merged.extend(batch)

        logger.info("ResearchAgent: step merged → %d paper(s)", len(merged))
        return merged

    async def _enrich_with_pdfs(
        self, papers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Concurrently call Unpaywall for every paper that has a DOI."""

        async def _fetch(paper: dict[str, Any]) -> dict[str, Any]:
            doi = (paper.get("doi") or "").strip()
            if not doi:
                return paper
            try:
                pdf_url = await asyncio.to_thread(self._unpaywall.run, doi)
                if pdf_url and isinstance(pdf_url, str):
                    return {**paper, "pdf_url": pdf_url}
            except Exception as exc:
                logger.debug("Unpaywall skip doi=%r: %s", doi, exc)
            return paper

        return list(await asyncio.gather(*[_fetch(p) for p in papers]))


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _lazy_import_tool(module_name: str) -> Any:
    """
    Import and return a runnable tool from tools.<module_name>.

    Resolution order (stops at first hit):
    1. Module-level BaseTool instances
    2. Concrete BaseTool subclasses defined in this module
    3. Any class with a callable .run() defined in this module
    4. TitleCase name convention (arxiv_search → ArxivSearch)

    Returns None with a debug log if nothing runnable is found.
    """
    import importlib
    import inspect

    _SKIP_NAMES: frozenset[str] = frozenset(
        {"BaseTool", "StructuredTool", "Tool", "BaseModel", "BaseSettings",
         "RunnableSerializable", "Runnable"}
    )

    try:
        mod = importlib.import_module(f"tools.{module_name}")
    except ImportError:
        logger.debug("_lazy_import_tool: module %r not available", module_name)
        return None
    except Exception as exc:
        logger.warning("_lazy_import_tool: failed to import %r — %s", module_name, exc)
        return None

    try:
        from langchain_core.tools import BaseTool
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            obj = getattr(mod, attr_name, None)
            if isinstance(obj, BaseTool):
                logger.debug("_lazy_import_tool[%s]: found instance %r", module_name, attr_name)
                return obj
    except ImportError:
        BaseTool = None  # type: ignore[assignment]

    if BaseTool is not None:
        for attr_name, cls in inspect.getmembers(mod, inspect.isclass):
            if attr_name in _SKIP_NAMES or cls.__module__ != mod.__name__:
                continue
            if issubclass(cls, BaseTool) and cls is not BaseTool:
                try:
                    instance = cls()
                    logger.debug("_lazy_import_tool[%s]: instantiated %r", module_name, attr_name)
                    return instance
                except Exception as exc:
                    logger.debug("_lazy_import_tool[%s]: %r() raised %s", module_name, attr_name, exc)

    for attr_name, cls in inspect.getmembers(mod, inspect.isclass):
        if attr_name in _SKIP_NAMES or cls.__module__ != mod.__name__:
            continue
        if callable(getattr(cls, "run", None)):
            try:
                instance = cls()
                logger.debug("_lazy_import_tool[%s]: instantiated .run()-capable %r", module_name, attr_name)
                return instance
            except Exception:
                continue

    class_name = "".join(part.title() for part in module_name.split("_"))
    cls = getattr(mod, class_name, None)
    if cls is not None and inspect.isclass(cls) and class_name not in _SKIP_NAMES:
        try:
            return cls()
        except Exception as exc:
            logger.debug("_lazy_import_tool[%s]: TitleCase %r() raised %s", module_name, class_name, exc)

    logger.warning("_lazy_import_tool[%s]: no runnable tool found in module", module_name)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_from_env(
    *,
    max_papers_per_step: int   = 10,
    max_total_papers:    int   = 40,
    tool_timeout:        float = 120.0,
) -> ResearchAgent:
    """
    Build a production ResearchAgent from environment variables.

    Required: at least one of GROQ_KEY_1 / GROQ_KEY_2 / GROQ_KEY_3.
    Optional: GOOGLE_API_KEY (Gemini fallback), S2_API_KEY (Semantic Scholar).

    Raises EnvironmentError if no Groq keys are configured.
    """
    groq_keys = [
        k for k in (
            os.getenv("GROQ_KEY_1", ""),
            os.getenv("GROQ_KEY_2", ""),
            os.getenv("GROQ_KEY_3", ""),
        )
        if k
    ]
    if not groq_keys:
        raise EnvironmentError(
            "Set at least one of GROQ_KEY_1 / GROQ_KEY_2 / GROQ_KEY_3."
        )

    gemini_key = os.getenv("GOOGLE_API_KEY", "")
    llm = ResilientLLM(groq_keys=groq_keys, gemini_key=gemini_key)

    return ResearchAgent(
        llm                 = llm,
        max_papers_per_step = max_papers_per_step,
        max_total_papers    = max_total_papers,
        tool_timeout        = tool_timeout,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

async def _amain(query: str) -> None:
    agent = build_from_env()
    report = await agent.research(query)
    print(report)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog        = "researchflow",
        description = "ResearchFlow — autonomous academic research agent",
    )
    parser.add_argument("query", nargs="?", help="Research question or topic")
    parser.add_argument("--max-papers", type=int, default=40, metavar="N")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.query:
        parser.print_help()
        raise SystemExit(1)

    asyncio.run(_amain(args.query))


if __name__ == "__main__":
    main()