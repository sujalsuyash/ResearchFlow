"""
agents/research_agent.py
────────────────────────
ResearchFlow's autonomous research orchestrator.

Two public classes:

  ResilientLLM
      A LangChain-compatible Runnable that tries up to three Groq API keys in
      sequence, catching 429 / quota errors and immediately falling back to the
      next key.  If all Groq keys are exhausted it falls over to Gemini
      (gemini-1.5-flash) with all safety filters disabled so academic content
      is never blocked.

  ResearchAgent
      Async orchestrator.  Calls QueryPlanner → parallel tool fan-out →
      global deduplication → Unpaywall PDF enrichment → SynthesisChain.

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
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig

# Chains
from chains.query_planner import QueryPlanner, ResearchPlan, SearchStep  # type: ignore[import]
from chains.synthesizer import SynthesisChain

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Rate-limit / quota error detection
# ──────────────────────────────────────────────────────────────────────────────

_RATE_LIMIT_SIGNALS: tuple[str, ...] = (
    "rate limit",
    "ratelimit",
    "429",
    "too many requests",
    "quota",
    "resource_exhausted",      # gRPC / Gemini spelling
    "resource exhausted",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Return ``True`` when *exc* represents a 429 / quota-exceeded condition.

    Checks:
      1. Groq SDK native ``RateLimitError`` / ``APIStatusError(status=429)``
      2. httpx ``HTTPStatusError`` with status 429
      3. Any exception whose string representation contains a known signal
    """
    # 1 — Groq SDK
    try:
        import groq  # optional dep — only present when langchain-groq is installed

        if isinstance(exc, groq.RateLimitError):
            return True
        if isinstance(exc, groq.APIStatusError) and getattr(exc, "status_code", None) == 429:
            return True
    except ImportError:
        pass

    # 2 — httpx (tools layer, could bubble up)
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            return True
    except ImportError:
        pass

    # 3 — String heuristic (catches LangChain wrapping, Gemini gRPC errors, etc.)
    msg = str(exc).lower()
    return any(signal in msg for signal in _RATE_LIMIT_SIGNALS)


# ──────────────────────────────────────────────────────────────────────────────
# ResilientLLM
# ──────────────────────────────────────────────────────────────────────────────

class ResilientLLM(Runnable):
    """
    LangChain-compatible ``Runnable`` with ordered multi-provider fallback.

    Priority order (left → right):
        Groq key 1  →  Groq key 2  →  Groq key 3  →  Gemini

    A 429 / quota error on any provider causes an *immediate* retry on the
    next provider in the chain.  All other errors are re-raised to the caller
    so bugs surface clearly.

    Parameters
    ----------
    groq_keys:
        Ordered list of Groq API keys.  Pass 1–3 keys; the class tries them
        strictly left-to-right.
    gemini_key:
        Google AI Studio key.  Used only if every Groq key is exhausted.
        Pass ``""`` to disable the Gemini fallback (``RuntimeError`` is raised
        when all Groq keys fail in that case).
    groq_model:
        Groq model name (default: ``llama-3.3-70b-versatile``).
    gemini_model:
        Gemini model name (default: ``gemini-1.5-flash``).
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
        self._groq_keys = list(groq_keys)
        self._gemini_key = gemini_key
        self._groq_model = groq_model
        self._gemini_model = gemini_model
        self._temperature = temperature
        # Set by with_structured_output(); None means plain chat mode.
        self._schema: Any = None

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "ResilientLLM":
        """
        Mirror the LangChain ``BaseChatModel.with_structured_output`` interface.

        Returns a **new** ``ResilientLLM`` that applies ``.with_structured_output``
        to each concrete provider LLM at call time.  This lets the QueryPlanner
        (which calls ``llm.with_structured_output(ResearchPlan)``) work with
        ``ResilientLLM`` as a transparent drop-in.
        """
        clone = ResilientLLM(
            groq_keys=self._groq_keys,
            gemini_key=self._gemini_key,
            groq_model=self._groq_model,
            gemini_model=self._gemini_model,
            temperature=self._temperature,
        )
        clone._schema = schema
        return clone

    # ── Private factory helpers ───────────────────────────────────────────────

    def _make_groq(self, api_key: str) -> Runnable:
        """Construct a ``ChatGroq`` instance for *api_key*."""
        from langchain_groq import ChatGroq  # type: ignore[import]

        return ChatGroq(
            api_key=api_key,
            model=self._groq_model,
            temperature=self._temperature,
        )

    def _make_gemini(self) -> Runnable:
        """
        Construct a ``ChatGoogleGenerativeAI`` instance with all safety
        filters set to ``BLOCK_NONE`` so academic/technical content is never
        incorrectly rejected.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]

        safety_settings = _build_gemini_safety_settings()

        return ChatGoogleGenerativeAI(
            google_api_key=self._gemini_key,
            model=self._gemini_model,
            temperature=self._temperature,
            safety_settings=safety_settings,
            # convert_system_message_to_human keeps Gemini compatible with
            # ChatPromptTemplates that use a ("system", "…") message.
            convert_system_message_to_human=True,
        )

    def _iter_providers(self):
        """
        Yield ``(label, llm)`` pairs one at a time in priority order.

        Using a **generator** (not a list) is deliberate: each concrete LLM is
        constructed only at the moment it is about to be tried.  This means:

        • ``_make_gemini()`` is never called when a Groq key succeeds.
        • Test mocks with finite ``side_effect`` lists are consumed one call per
          actual attempt, not all at once during provider-list construction.
        • When ``_schema`` is set via ``with_structured_output``, the schema is
          applied to each concrete LLM right before it is yielded.

        Provider order: Groq[1] → Gemini → Groq[2] → Groq[3]

        Why Gemini is second (not last):
        When multiple Groq keys belong to the same organisation they share a
        single token-per-day quota.  Once Groq[1] hits a 429, Groq[2] and
        Groq[3] will immediately fail with the same error — trying them first
        wastes time.  Gemini (1 M tokens/day free tier) is placed immediately
        after the first Groq failure so it absorbs the overflow.  The remaining
        Groq keys are kept as a last-resort safety net for Gemini outages.
        """
        def _wrap(llm: Runnable) -> Runnable:
            return llm.with_structured_output(self._schema) if self._schema is not None else llm

        # Groq[1] — always the first attempt (fastest, free tier)
        if self._groq_keys:
            yield "groq[1]", _wrap(self._make_groq(self._groq_keys[0]))

        # Gemini — second, so it catches org-wide Groq quota exhaustion early
        if self._gemini_key:
            yield "gemini", _wrap(self._make_gemini())

        # Remaining Groq keys — final safety net for Gemini outages
        for i, key in enumerate(self._groq_keys[1:], start=2):
            yield f"groq[{i}]", _wrap(self._make_groq(key))

    # ── Runnable interface ────────────────────────────────────────────────────

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Synchronous invoke with automatic provider fallback.

        Mirrors the ``BaseChatModel.invoke`` signature so this class is a
        drop-in replacement anywhere LangChain expects an LLM runnable.
        """
        last_rate_exc: BaseException | None = None

        for label, llm in self._iter_providers():
            try:
                logger.debug("ResilientLLM.invoke: trying %s", label)
                result = llm.invoke(input, config, **kwargs)
                logger.debug("ResilientLLM.invoke: success with %s", label)
                return result
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    logger.warning(
                        "ResilientLLM: %s rate-limited (%s) — falling back",
                        label,
                        exc,
                    )
                    last_rate_exc = exc
                    continue
                # Non-rate-limit errors (auth failures, bad requests, etc.)
                # are re-raised immediately — they indicate a configuration
                # bug, not a capacity issue.
                raise

        raise RuntimeError(
            "ResilientLLM: all providers exhausted by rate limits."
        ) from last_rate_exc

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async variant of :meth:`invoke`."""
        last_rate_exc: BaseException | None = None

        for label, llm in self._iter_providers():
            try:
                logger.debug("ResilientLLM.ainvoke: trying %s", label)
                result = await llm.ainvoke(input, config, **kwargs)
                logger.debug("ResilientLLM.ainvoke: success with %s", label)
                return result
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    logger.warning(
                        "ResilientLLM: %s rate-limited (%s) — falling back",
                        label,
                        exc,
                    )
                    last_rate_exc = exc
                    continue
                raise

        raise RuntimeError(
            "ResilientLLM: all providers exhausted by rate limits."
        ) from last_rate_exc


def _build_gemini_safety_settings() -> dict:
    """
    Return a safety-settings dict that disables all Gemini content filters
    using plain strings supported by the latest langchain-google-genai.
    """
    return {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

# ──────────────────────────────────────────────────────────────────────────────
# Domain-routing helpers
# ──────────────────────────────────────────────────────────────────────────────

# Keyword sets for lightweight domain classification when the QueryPlanner
# doesn't emit an explicit `domain` field on a SearchStep.
_BIO_KEYWORDS: frozenset[str] = frozenset(
    {
        "drug", "clinical", "patient", "disease", "therapy", "cancer",
        "gene", "protein", "cell", "biology", "medical", "health", "pharma",
        "virus", "bacteria", "neuron", "brain", "genomic", "dna", "rna",
        "vaccine", "epidemiology", "pharmacology", "pathogen", "mutation",
    }
)
_CS_KEYWORDS: frozenset[str] = frozenset(
    {
        "machine learning", "deep learning", "neural network", "transformer",
        "language model", "llm", "reinforcement learning", "computer vision",
        "nlp", "natural language", "artificial intelligence", "optimization",
        "graph network", "autoencoder", "diffusion model", "retrieval",
    }
)


def _detect_domain(text: str) -> str:
    """Return ``'biomedical'``, ``'cs'``, or ``'general'`` from *text*."""
    lower = text.lower()
    bio_hits = sum(1 for kw in _BIO_KEYWORDS if kw in lower)
    # Multi-word CS keywords need a different check
    cs_hits = sum(1 for kw in _CS_KEYWORDS if kw in lower)
    if bio_hits > cs_hits:
        return "biomedical"
    if cs_hits > bio_hits:
        return "cs"
    return "general"


def _tools_for_domain(
    domain: str,
    arxiv: Any,
    semantic_scholar: Any,
    pubmed: Any,
    openalex: Any,
) -> list[tuple[str, Any]]:
    """
    Return an ordered ``[(name, tool)]`` list for *domain*.

    ┌──────────────────┬──────────────────────────────────────────────┐
    │ Domain           │ Tools                                         │
    ├──────────────────┼──────────────────────────────────────────────┤
    │ cs / ml          │ arXiv, Semantic Scholar, OpenAlex             │
    │ biomedical       │ PubMed, Semantic Scholar, OpenAlex            │
    │ general (fanout) │ arXiv, Semantic Scholar, PubMed, OpenAlex    │
    └──────────────────┴──────────────────────────────────────────────┘
    """
    d = domain.lower()
    if any(token in d for token in ("bio", "med", "pharma", "clinic")):
        return [
            ("pubmed", pubmed),
            ("semantic_scholar", semantic_scholar),
            ("openalex", openalex),
        ]
    if any(token in d for token in ("cs", "comput", "ml", "ai", "nlp")):
        return [
            ("arxiv", arxiv),
            ("semantic_scholar", semantic_scholar),
            ("openalex", openalex),
        ]
    # Cross-disciplinary — full fan-out
    return [
        ("arxiv", arxiv),
        ("semantic_scholar", semantic_scholar),
        ("pubmed", pubmed),
        ("openalex", openalex),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_key(paper: dict[str, Any]) -> str:
    """
    Canonical identity key for a paper.

    Priority:
      1. arXiv ID — extracted from either the doi field (10.48550/arxiv.XXXX)
         or the dedicated arxiv_id field. Used first because the same paper
         often appears as both an arXiv preprint and a published version with
         a completely different DOI — the arXiv ID is the stable cross-source
         identifier that links them.
      2. Normalised non-arXiv DOI — strip whitespace, lowercase, remove
         https://doi.org/ prefix.
      3. title: + stripped lowercase title — final fallback when no DOI exists.
    """
    import re

    # ── Step 1: extract arXiv ID from any available field ─────────────────────
    # Semantic Scholar returns it as arxiv_id; arXiv tool returns it in the DOI
    # as "10.48550/arxiv.2009.02902" or in the url as "arxiv.org/abs/2009.02902"
    arxiv_id: str = ""

    raw_arxiv = (paper.get("arxiv_id") or "").strip().lower()
    if raw_arxiv:
        arxiv_id = raw_arxiv

    if not arxiv_id:
        doi_str = (paper.get("doi") or "").strip().lower()
        # arXiv DOIs follow the pattern 10.48550/arxiv.XXXXXXXXX
        arxiv_doi_match = re.search(r"10\.48550/arxiv\.(\S+)", doi_str)
        if arxiv_doi_match:
            arxiv_id = arxiv_doi_match.group(1).rstrip(".")

    if not arxiv_id:
        url_str = (paper.get("url") or "").strip().lower()
        # arXiv URLs: arxiv.org/abs/2009.02902 or arxiv.org/pdf/2009.02902
        arxiv_url_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)", url_str)
        if arxiv_url_match:
            arxiv_id = arxiv_url_match.group(1)

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

def deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge papers from multiple API sources into a deduplicated list.
    When two entries share the same key:
      - The one with the higher citation count is kept, since the published
        version typically has more citations than the preprint and carries
        richer metadata.
      - Falls back to longer abstract when citation counts are equal or absent,
        so the Synthesiser always gets the richest available context.
    """
    seen: dict[str, dict[str, Any]] = {}
    for paper in papers:
        key = _dedup_key(paper)
        if key not in seen:
            seen[key] = paper
        else:
            existing = seen[key]
            existing_cit = existing.get("citation_count") or 0
            incoming_cit = paper.get("citation_count") or 0

            if incoming_cit > existing_cit:
                # Published version usually has more citations — prefer it
                seen[key] = paper
            elif incoming_cit == existing_cit:
                # Same citation count — keep whichever has the longer abstract
                existing_len = len(existing.get("abstract") or "")
                incoming_len = len(paper.get("abstract") or "")
                if incoming_len > existing_len:
                    seen[key] = paper

    return list(seen.values())

# ──────────────────────────────────────────────────────────────────────────────
# Tool-output normaliser — helpers
# ──────────────────────────────────────────────────────────────────────────────

def _reconstruct_inverted_index(index: dict) -> str:
    """
    Reassemble an OpenAlex ``abstract_inverted_index`` into a plain string.

    OpenAlex stores abstracts as an inverted index mapping each word to the
    list of positions it occupies, e.g.::

        {"We": [0], "propose": [1], "a": [2, 7], "method": [3], ...}

    This function reverses that mapping: it places each word at every position
    it claims, then joins the result in order.

    Returns an empty string if *index* is falsy, not a dict, or has no entries.
    """
    if not index or not isinstance(index, dict):
        return ""

    # Find the total length needed so we can pre-allocate the slot list
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

    Key aliases handled
    ───────────────────
    title        ← title (standard) | display_name (OpenAlex) | name
    abstract     ← abstract (standard) | summary (arXiv) |
                   abstract_inverted_index (OpenAlex inverted map)
    authors      ← authors | authorships (OpenAlex list of dicts)
    year         ← year | publication_year (OpenAlex) | publishedDate (truncated)
    citation_count ← citation_count | cited_by_count (OpenAlex) | citationCount (SS)
    doi          ← doi | DOI
    url          ← url | URL | entry_id (arXiv) | landing_page_url
    """
    # ── title ─────────────────────────────────────────────────────────────────
    title: str = (
        raw.get("title")
        or raw.get("display_name")   # OpenAlex
        or raw.get("name")
        or ""
    )

    # ── abstract ──────────────────────────────────────────────────────────────
    abstract: str = raw.get("abstract") or raw.get("summary") or ""  # summary = arXiv
    if not abstract:
        aii = raw.get("abstract_inverted_index")  # OpenAlex raw dict
        if isinstance(aii, dict):
            abstract = _reconstruct_inverted_index(aii)

    # Bail early if there is nothing useful in this dict
    if not title and not abstract:
        return []

    # ── authors ───────────────────────────────────────────────────────────────
    raw_authors = raw.get("authors") or raw.get("authorships") or []
    authors: list[str] = []
    for a in raw_authors:
        if isinstance(a, str):
            authors.append(a)
        elif isinstance(a, dict):
            # OpenAlex authorship object: {"author": {"display_name": "..."}}
            # Semantic Scholar author object: {"name": "..."}
            name = (
                a.get("name")                                    # Semantic Scholar
                or (a.get("author") or {}).get("display_name")  # OpenAlex
                or a.get("display_name")
            )
            if name:
                authors.append(str(name))

    # ── year ──────────────────────────────────────────────────────────────────
    year: int | None = (
        raw.get("year")
        or raw.get("publication_year")   # OpenAlex
    )
    if year is None:
        # arXiv "published": "2023-05-12T..." — take the first 4 chars
        published = raw.get("published") or raw.get("publishedDate") or ""
        if published and str(published)[:4].isdigit():
            year = int(str(published)[:4])
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    # ── citation count ────────────────────────────────────────────────────────
    citation_count: int | None = (
        raw.get("citation_count")
        or raw.get("cited_by_count")   # OpenAlex
        or raw.get("citationCount")    # Semantic Scholar (camelCase)
    )

    # ── doi ───────────────────────────────────────────────────────────────────
    doi: str | None = raw.get("doi") or raw.get("DOI")
    # OpenAlex doi field is already a full URL like "https://doi.org/10...."
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # ── url ───────────────────────────────────────────────────────────────────
    url: str = (
        raw.get("url")
        or raw.get("URL")
        or raw.get("entry_id")              # arXiv
        or raw.get("landing_page_url")      # OpenAlex primary_location
        or (f"https://doi.org/{doi}" if doi else "")
    )

    # ── primary_location unwrapping (OpenAlex) ────────────────────────────────
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

    # Carry through any extra keys the synthesiser might find useful
    for extra in ("source", "arxiv_id", "fields_of_study", "tldr", "pdf_url"):
        if raw.get(extra) is not None:
            entry[extra] = raw[extra]

    return [entry]


def _parse_formatted_string(text: str) -> list[dict[str, Any]]:
    """
    Parse a human-readable multi-paper string returned by a LangChain tool's
    ``.run()`` method back into a list of normalised paper dicts.

    Handles two distinct formats that our tools actually emit:

    Format A — numbered blocks (Semantic Scholar, arXiv, OpenAlex tool strings)
    ───────────────────────────────────────────────────────────────────────────
        [1] Title of the paper (2023)
            Authors: Smith, J., Doe, A.
            Citations: 42 | DOI: 10.1234/x
            URL: https://...
            Abstract: The paper proposes ...

        [2] Another Title (2024)
            ...

    Format B — key-prefixed blocks (PubMed / concatenated LangChain Documents)
    ───────────────────────────────────────────────────────────────────────────
        Title: Some title
        Abstract: Some abstract text
        Authors: ...
        Year: 2022

        Title: Another title
        Abstract: ...
    """
    papers: list[dict[str, Any]] = []
    text = text.strip()

    # ── Format A detection: text contains "[N]" numbered entries ─────────────
    import re

    # Split on lines that start with "[<digits>]" — these are paper boundaries
    # Use a lookahead so the delimiter is kept with the following block.
    blocks_a = re.split(r"\n(?=\[\d+\])", text)

    # If we found more than one block, or the first block starts with "[N]",
    # treat this as Format A.
    if len(blocks_a) > 1 or re.match(r"^\[\d+\]", blocks_a[0].strip()):
        for block in blocks_a:
            block = block.strip()
            if not block or not re.match(r"^\[\d+\]", block):
                continue  # skip header lines like "Semantic Scholar results for..."

            lines = block.splitlines()
            # Line 0: "[N] Title (year)"  — strip the "[N] " prefix
            first_line = re.sub(r"^\[\d+\]\s*", "", lines[0]).strip()

            # Extract year from trailing "(YYYY)" if present
            year_match = re.search(r"\((\d{4})\)\s*$", first_line)
            year: int | None = int(year_match.group(1)) if year_match else None
            title = re.sub(r"\s*\(\d{4}\)\s*$", "", first_line).strip()

            entry: dict[str, Any] = {"title": title or "Untitled"}
            if year:
                entry["year"] = year

            # Parse labelled sub-lines: "    Key: value"
            for line in lines[1:]:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # Multi-line abstract: continuation lines have no "Key:" prefix
                if "abstract" in entry and not re.match(r"^[A-Za-z ]+:", line_stripped):
                    entry["abstract"] = entry["abstract"] + " " + line_stripped
                    continue

                if line_stripped.lower().startswith("abstract:"):
                    val = line_stripped[len("abstract:"):].strip()
                    # Strip trailing "..." added by the tool's truncation
                    if val.endswith("..."):
                        val = val[:-3].strip()
                    entry["abstract"] = val

                elif line_stripped.lower().startswith("authors:"):
                    raw_authors = line_stripped[len("authors:"):].strip()
                    # "Smith, J., Doe, A. et al. (+3 more)"  — drop the "et al." suffix
                    raw_authors = re.sub(r"\s+et al\.\s*\(\+\d+ more\)", "", raw_authors)
                    entry["authors"] = [a.strip() for a in raw_authors.split(",") if a.strip()]

                elif line_stripped.lower().startswith("url:"):
                    entry["url"] = line_stripped[4:].strip()

                elif "doi:" in line_stripped.lower():
                    # "Citations: 42 | DOI: 10.1234/x" or "DOI: 10.1234/x"
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
        # Fall through to Format B if no papers were extracted

    # ── Format B detection: blank-line separated blocks with "Key: value" lines ─
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

        # Map Format B keys to standard schema
        title = entry.get("title") or entry.get("display_name") or ""
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
    Aggressively coerce any tool return value into a ``list[dict]`` using the
    standard paper schema (title, abstract, authors, year, doi, url,
    citation_count).

    Resolution order
    ────────────────
    1. ``None`` / falsy                → ``[]``
    2. ``list``                        → recurse each element, flatten
    3. ``dict``                        → ``_normalise_dict`` (handles all key aliases
                                         and OpenAlex abstract_inverted_index)
    4. LangChain ``Document``          → mine ``.page_content`` + ``.metadata``
    5. Pydantic v2 model               → ``.model_dump()`` then recurse
    6. Pydantic v1 model               → ``.dict()`` then recurse
    7. JSON string (starts ``[`` /``{``) → parse then recurse
    8. Structured plain string         → ``_parse_formatted_string``
                                         (handles ``[N] Title`` and ``Title:/Abstract:``
                                         multi-paper formats emitted by tool ``.run()``)
    9. Unparsable string (≥ 30 chars)  → unique-titled fallback so deduplication
                                         never collapses multiple blobs into one
    10. Other objects                  → ``vars()`` then give up
    """
    if not raw:
        return []

    # ── 1. list ───────────────────────────────────────────────────────────────
    if isinstance(raw, list):
        results: list[dict[str, Any]] = []
        for item in raw:
            results.extend(_normalise_tool_output(item))
        return results

    # ── 2. dict ───────────────────────────────────────────────────────────────
    if isinstance(raw, dict):
        return _normalise_dict(raw)

    # ── 3. LangChain Document ─────────────────────────────────────────────────
    if hasattr(raw, "page_content"):
        meta: dict = getattr(raw, "metadata", {}) or {}
        # Reconstruct OpenAlex inverted index if it landed in metadata
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
            "year": (
                meta.get("Published") or meta.get("year")
                or meta.get("publication_year")
            ),
            "doi": meta.get("DOI") or meta.get("doi"),
            "url": (
                meta.get("URL") or meta.get("url") or meta.get("entry_id") or ""
            ),
            "citation_count": (
                meta.get("citationCount") or meta.get("citation_count")
                or meta.get("cited_by_count")
            ),
        }
        entry = {k: v for k, v in entry.items() if v is not None}
        if entry.get("abstract") or entry.get("title", "Untitled") != "Untitled":
            return [entry]
        return []

    # ── 4. Pydantic v2 ────────────────────────────────────────────────────────
    if hasattr(raw, "model_dump"):
        try:
            return _normalise_tool_output(raw.model_dump())
        except Exception:
            pass

    # ── 5. Pydantic v1 ────────────────────────────────────────────────────────
    if hasattr(raw, "dict") and callable(getattr(raw, "dict")):
        try:
            return _normalise_tool_output(raw.dict())
        except Exception:
            pass

    # ── 6. String ─────────────────────────────────────────────────────────────
    if isinstance(raw, str):
        import json

        stripped = raw.strip()
        if not stripped:
            return []

        # 6a. JSON
        if stripped.startswith(("[", "{")):
            try:
                return _normalise_tool_output(json.loads(stripped))
            except (json.JSONDecodeError, ValueError):
                pass

        # 6b. Structured multi-paper string (the common case for .run() output)
        parsed = _parse_formatted_string(stripped)
        if parsed:
            return parsed

        # 6c. Last-resort fallback: keep the blob but give it a UNIQUE title
        # derived from a snippet of the text so the deduplicator never merges
        # multiple unparsed blobs into a single "Research Paper" entry.
        if len(stripped) >= 30:
            snippet = stripped[:50].replace("\n", " ").strip()
            return [{"title": f"Raw: {snippet}…", "abstract": stripped}]
        return []

    # ── 7. vars() last resort ─────────────────────────────────────────────────
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
    1. **Plan**       — ``QueryPlanner`` decomposes the query into ``SearchStep`` objects.
    2. **Execute**    — Tools are called **concurrently** (``asyncio.gather``) per step.
                        Domain routing selects the relevant subset of APIs.
    3. **Dedup**      — Papers are merged globally by DOI (title fallback).
                        The top *max_total_papers* by citation count are kept.
    4. **Enrich**     — Unpaywall is queried concurrently for every paper with a DOI,
                        appending a ``pdf_url`` when an open-access copy is available.
    5. **Synthesise** — ``SynthesisChain`` produces a cited Markdown report.

    Parameters
    ----------
    llm:
        Any LangChain ``Runnable`` (typically a ``ResilientLLM``).  Injected
        into both the planner and the synthesiser.
    *_tool:
        Pre-built tool instances.  If omitted, default instances are created.
        Override in tests to inject mocks without patching.
    max_papers_per_step:
        Hard cap on papers returned by each individual tool call.
    max_total_papers:
        Hard cap on papers passed to the Synthesiser (highest citation counts
        are kept when the pool exceeds this limit).
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
        self._llm = llm
        self._planner = QueryPlanner(llm=llm)
        self._synthesiser = SynthesisChain(llm=llm)

        # Lazy imports so the module still loads even if a tool package is
        # absent (useful for testing individual components in isolation).
        self._semantic_scholar = semantic_scholar or _lazy_import_tool("semantic_scholar")
        self._arxiv = arxiv or _lazy_import_tool("arxiv_search")
        self._pubmed = pubmed or _lazy_import_tool("pubmed_search")
        self._openalex = openalex or _lazy_import_tool("openalex_search")
        self._unpaywall = unpaywall or _lazy_import_tool("unpaywall_fetcher")

        self._max_papers_per_step = max_papers_per_step
        self._max_total_papers = max_total_papers
        # Per-tool timeout in seconds.  Must be long enough to accommodate a
        # tool's full internal retry + backoff sequence.  Semantic Scholar's
        # exponential backoff reaches 10 s + 20 s + 40 s = 70 s before its
        # 4th attempt, so the default of 120 s gives it comfortable headroom.
        self._tool_timeout = tool_timeout

    # ── Public API ────────────────────────────────────────────────────────────

    async def research(self, query: str) -> str:
        """
        Run the full five-stage pipeline for *query*.

        Returns
        -------
        str
            The final Markdown research report with inline citations and
            a numbered bibliography.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        logger.info("ResearchAgent: starting — %r", query)

        # ── 1. Plan ───────────────────────────────────────────────────────────
        plan: ResearchPlan = await self._aplan(query)
        logger.info("ResearchAgent: %d search step(s) planned", len(plan.steps))

        # ── 2. Execute (concurrent tool fan-out per step) ─────────────────────
        all_papers: list[dict[str, Any]] = []
        for idx, step in enumerate(plan.steps, start=1):
            # Support both sub_question (original schema) and search_query
            label = (
                getattr(step, "sub_question", None)
                or getattr(step, "search_query", None)
                or f"step {idx}"
            )
            logger.info("ResearchAgent: step %d/%d — %r", idx, len(plan.steps), label)
            step_papers = await self._execute_step(step)
            all_papers.extend(step_papers)
            logger.info("ResearchAgent: step %d → %d paper(s)", idx, len(step_papers))

        # ── 3. Global deduplication + cap ─────────────────────────────────────
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
        # Sort by citation count descending so the most-cited work is cited first
        unique = sorted(
            unique, key=lambda p: p.get("citation_count") or 0, reverse=True
        )[: self._max_total_papers]

        # ── 4. PDF enrichment via Unpaywall ───────────────────────────────────
        unique = await self._enrich_with_pdfs(unique)

        # ── 5. Synthesis ──────────────────────────────────────────────────────
        result = await self._synthesiser.arun(query, unique)
        logger.info("ResearchAgent: done — %d paper(s) cited", result.paper_count)
        return result.report

    # ── Stage helpers ─────────────────────────────────────────────────────────

    async def _aplan(self, query: str) -> ResearchPlan:
        """
        Call the QueryPlanner, preferring its async interface.

        Falls back to ``asyncio.to_thread`` when the planner only exposes a
        synchronous ``plan()`` method (forward-compatibility shim).
        """
        if hasattr(self._planner, "aplan"):
            return await self._planner.aplan(query)
        # Synchronous fallback
        return await asyncio.to_thread(self._planner.plan, query)

    async def _execute_step(self, step: Any) -> list[dict[str, Any]]:
        """
        Fan out to domain-selected tools concurrently with politeness controls.

        Politeness features
        ───────────────────
        • **Staggered start** — each tool waits ``tool_index × 1.5 s`` before
          firing so all four APIs never receive simultaneous bursts.
        • **Timeout circuit-breaker** — ``asyncio.wait_for`` kills any tool call
          that blocks for more than 25 s, logging a warning and returning ``[]``.
        • **Failure isolation** — any exception (network, auth, parse) is caught
          per-tool so one broken API never aborts the whole step.

        The ``step`` object may carry either a ``sub_question`` (original schema)
        or ``search_query`` (alternative QueryPlanner field name); both are
        supported via ``getattr`` fallback.
        """
        # ── Resolve the sub-question text ─────────────────────────────────────
        sub_question: str = (
            getattr(step, "sub_question", None)
            or getattr(step, "search_query", None)
            or ""
        )

        # ── Domain routing (Semantic Scholar is now live everywhere) ──────────
        domain = getattr(step, "domain", None) or _detect_domain(sub_question)
        selected = _tools_for_domain(
            domain,
            arxiv=self._arxiv,
            semantic_scholar=self._semantic_scholar,
            pubmed=self._pubmed,
            openalex=self._openalex,
        )

        # ── Build the query string sent to every tool ─────────────────────────
        # Prefer a pre-built keyword string from the planner when available.
        query_text: str = (
            getattr(step, "keywords_string", None)
            or getattr(step, "keywords", None)
            or sub_question
        )
        if isinstance(query_text, list):
            query_text = " ".join(query_text)

        # ── Per-tool coroutine with stagger + timeout ─────────────────────────
        async def _call_tool(
            name: str, tool: Any, delay: float
        ) -> list[dict[str, Any]]:
            if tool is None:
                return []
            try:
                # Staggered start: tool 0 fires immediately, tool 1 after 1.5 s, …
                if delay > 0:
                    await asyncio.sleep(delay)

                # Timeout circuit-breaker: abort tools that block longer than
                # _tool_timeout seconds.  The default (120 s) covers Semantic
                # Scholar's full 4-attempt backoff (10 + 20 + 40 s).
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

        # ── Dispatch all tools concurrently ───────────────────────────────────
        tasks = [
            _call_tool(name, tool, idx * 1.5)
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
        """
        Concurrently call Unpaywall for every paper that has a DOI.
        Papers without a DOI pass through unchanged.
        A ``pdf_url`` key is added when an open-access PDF is found.
        """

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
    Import and return a runnable tool from ``tools.<module_name>``.

    Resolution order (stops at the first hit):

    1. **Module-level ``BaseTool`` instances** — the most reliable signal.
       Covers tools exported as singletons (e.g. ``arxiv_tool = ArxivTool()``).

    2. **Concrete ``BaseTool`` subclasses defined in this module** — skips
       imported base classes (``BaseTool``, ``StructuredTool``, ``Tool``) and
       any class whose ``__module__`` isn't this module.

    3. **Any class with a callable ``.run()`` defined in this module** — safety
       net for tools that don't inherit ``BaseTool`` explicitly.

    4. **TitleCase name convention** — final fallback for the ``ArxivSearch``
       naming pattern (``arxiv_search`` → ``ArxivSearch``).

    Returns ``None`` (with a debug log) if nothing runnable is found so the
    agent still loads in test environments where optional tool deps are absent.
    """
    import importlib
    import inspect

    # Generic base classes that should never be instantiated directly
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

    # ── Pass 1: module-level BaseTool instances ───────────────────────────────
    try:
        from langchain_core.tools import BaseTool

        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            obj = getattr(mod, attr_name, None)
            if isinstance(obj, BaseTool):
                logger.debug(
                    "_lazy_import_tool[%s]: found instance %r", module_name, attr_name
                )
                return obj
    except ImportError:
        BaseTool = None  # type: ignore[assignment]

    # ── Pass 2: concrete BaseTool subclasses defined in this module ───────────
    if BaseTool is not None:
        for attr_name, cls in inspect.getmembers(mod, inspect.isclass):
            if attr_name in _SKIP_NAMES:
                continue
            if cls.__module__ != mod.__name__:
                continue  # imported symbol, not defined here
            if issubclass(cls, BaseTool) and cls is not BaseTool:
                try:
                    instance = cls()
                    logger.debug(
                        "_lazy_import_tool[%s]: instantiated BaseTool subclass %r",
                        module_name, attr_name,
                    )
                    return instance
                except Exception as exc:
                    logger.debug(
                        "_lazy_import_tool[%s]: %r() raised %s", module_name, attr_name, exc
                    )

    # ── Pass 3: any class with .run() defined in this module ─────────────────
    for attr_name, cls in inspect.getmembers(mod, inspect.isclass):
        if attr_name in _SKIP_NAMES:
            continue
        if cls.__module__ != mod.__name__:
            continue
        if callable(getattr(cls, "run", None)):
            try:
                instance = cls()
                logger.debug(
                    "_lazy_import_tool[%s]: instantiated .run()-capable class %r",
                    module_name, attr_name,
                )
                return instance
            except Exception:
                continue

    # ── Pass 4: TitleCase convention (ArxivSearch, PubmedSearch, …) ──────────
    class_name = "".join(part.title() for part in module_name.split("_"))
    cls = getattr(mod, class_name, None)
    if cls is not None and inspect.isclass(cls) and class_name not in _SKIP_NAMES:
        try:
            return cls()
        except Exception as exc:
            logger.debug(
                "_lazy_import_tool[%s]: TitleCase %r() raised %s",
                module_name, class_name, exc,
            )

    logger.warning(
        "_lazy_import_tool[%s]: no runnable tool found in module", module_name
    )
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_from_env(
    *,
    max_papers_per_step: int = 10,
    max_total_papers: int = 40,
    tool_timeout: float = 120.0,
) -> ResearchAgent:
    """
    Build a production ``ResearchAgent`` from environment variables.

    Required
    --------
    At least one of ``GROQ_KEY_1`` / ``GROQ_KEY_2`` / ``GROQ_KEY_3``.

    Optional
    --------
    ``GOOGLE_API_KEY``           — enables Gemini as the final LLM fallback.
    ``S2_API_KEY``               — read by ``tools/semantic_scholar.py`` at
                                   instantiation time; raises the free rate limit
                                   from 1 req/s to 10 req/s.  The agent does not
                                   need to pass it explicitly — the tool reads it
                                   directly from the environment.

    Parameters
    ----------
    tool_timeout:
        Per-tool timeout in seconds (default 120).  Must cover the tool's full
        internal retry + backoff sequence.  Semantic Scholar backs off
        10 s → 20 s → 40 s across 4 attempts, so anything below ~90 s will
        kill it before it can succeed.

    Raises
    ------
    EnvironmentError
        If no Groq keys are configured.
    """
    groq_keys = [
        k
        for k in (
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
        llm=llm,
        max_papers_per_step=max_papers_per_step,
        max_total_papers=max_total_papers,
        tool_timeout=tool_timeout,
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
        prog="researchflow",
        description="ResearchFlow — autonomous academic research agent",
    )
    parser.add_argument("query", nargs="?", help="Research question or topic to investigate")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=40,
        metavar="N",
        help="Maximum papers to pass to the synthesiser (default: 40)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
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