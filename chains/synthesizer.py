"""
chains/synthesizer.py
─────────────────────
Synthesis chain for ResearchFlow.

Takes the original user query and a list of PaperResult dicts gathered by the
Research Agent and produces a structured Markdown report with inline citations
([1], [2], …) and a numbered bibliography.

Design mirrors query_planner.py:
  • LLM is injected at construction time (LLM-agnostic).
  • Uses LCEL: ChatPromptTemplate | llm | StrOutputParser.
  • A thin Pydantic model (SynthesisInput) validates the incoming payload.
  • No real API calls at this layer — the chain is pure text-in / text-out.
"""

from __future__ import annotations

import textwrap
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator
import logging
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

class PaperResult(BaseModel):
    """Represents a single paper retrieved by any search tool."""

    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str = ""
    doi: str | None = None
    citation_count: int | None = None

    @field_validator("authors", mode="before")
    @classmethod
    def coerce_authors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            # Academic names often look like "Smith, J., Doe, A." where the
            # comma is both a name separator AND part of "Lastname, Initials".
            # Prefer semicolons when present; otherwise fall back to splitting
            # on the pattern "<initial/word>, <Capital>" which marks a new name.
            if not v.strip():
                return []
            if ";" in v:
                return [a.strip() for a in v.split(";") if a.strip()]
            # Simple heuristic: split on ", " only where the next token starts
            # with a capital letter AND the preceding token is a short initial
            # or a single word (i.e. not "Lastname").
            import re
            # Replace name-boundary commas (", Uppercase") with a sentinel,
            # then split on that. This handles "Vaswani, A., Shazeer, N."
            # → ["Vaswani, A.", "Shazeer, N."]
            sentinel = "\x00"
            # A name boundary: comma-space followed by a capital letter that
            # is NOT immediately preceded by a single letter (i.e. not "A, B")
            normalised = re.sub(r",\s+(?=[A-Z])", sentinel, v)
            parts = [p.strip() for p in normalised.split(sentinel) if p.strip()]
            return parts if parts else [v.strip()]
        return v or []


class SynthesisInput(BaseModel):
    """Validated input for the SynthesisChain."""

    user_query: str = Field(..., min_length=1)
    papers: list[PaperResult] = Field(..., min_length=1)

    @classmethod
    def from_dicts(
        cls,
        user_query: str,
        papers: list[dict[str, Any]],
    ) -> "SynthesisInput":
        """Convenience constructor — accepts raw dicts from tool outputs."""
        return cls(
            user_query=user_query,
            papers=[PaperResult(**p) for p in papers],
        )


class SynthesisResult(BaseModel):
    """Structured output returned by SynthesisChain.run()."""

    report: str
    """Full Markdown report including inline citations and bibliography."""

    bibliography: list[str]
    """Ordered list of formatted reference strings (index 0 → [1], etc.)."""

    paper_count: int


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are ResearchFlow's synthesis engine — a senior research analyst who
    distills academic literature into clear, authoritative reports.

    Your reports follow this exact structure:
    ─────────────────────────────────────────────
    # Research Summary: {user_query}

    ## Overview
    <2–3 sentence executive summary of the collective findings>

    ## Key Themes & Findings

    ### <Theme 1 title>
    <Findings for this theme, citing papers inline as [N]>

    ### <Theme 2 title>
    …

    (add as many themes as the literature warrants, minimum 2)

    ## Gaps & Open Questions
    <What the literature does NOT answer; directions for future work>

    ## References
    [1] <Author(s)> (<Year>). *<Title>*. <URL or DOI if available>
    [2] …
    ─────────────────────────────────────────────

    Rules you MUST follow:
    • Every factual claim must carry at least one inline citation [N].
    • Citation numbers must match the References section exactly.
    • Do NOT invent papers or facts not present in the provided abstracts.
    • Do NOT include a paper in References unless it was cited in the body.
    • Before citing any paper, ask: does this paper directly address the user's research query? If a paper is tangentially related or clearly from an unrelated domain, do NOT cite it and do NOT include it in References. 
    • It is better to cite 15 highly relevant papers than 40 loosely related ones.
    • Write in clear, formal academic English.
    • Use **bold** for key terms on first use.
    • Keep each theme section focused — 2-5 sentences is usually right.
""")

_HUMAN_PROMPT = textwrap.dedent("""\
    User research query:
    {user_query}

    Source papers ({paper_count} total):
    {formatted_papers}

    Write the full Markdown report now.
""")

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ]
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_papers_for_prompt(papers: list[PaperResult]) -> str:
    """
    Render each paper as a numbered block the LLM can refer to via [N].
    The index here is 1-based and becomes the citation number in the report.
    """
    blocks: list[str] = []
    for i, p in enumerate(papers, start=1):
        author_str = (
            ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            if p.authors
            else "Unknown authors"
        )
        year_str = str(p.year) if p.year else "n.d."
        lines = [
            f"[{i}] {p.title}",
            f"    Authors : {author_str} ({year_str})",
        ]
        if p.abstract:
            # Trim very long abstracts to keep prompt size manageable
            abstract = p.abstract[:800] + "…" if len(p.abstract) > 800 else p.abstract
            lines.append(f"    Abstract: {abstract}")
        if p.url:
            lines.append(f"    URL     : {p.url}")
        if p.doi:
            lines.append(f"    DOI     : {p.doi}")
        if p.citation_count is not None:
            lines.append(f"    Cited by: {p.citation_count}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_bibliography(papers: list[PaperResult]) -> list[str]:
    """
    Returns one formatted reference string per paper (same 1-based order).
    Used to populate SynthesisResult.bibliography independently of LLM output.
    """
    refs: list[str] = []
    for p in papers:
        author_str = (
            ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            if p.authors
            else "Unknown authors"
        )
        year_str = str(p.year) if p.year else "n.d."
        loc = p.doi or p.url or "no link available"
        refs.append(f"{author_str} ({year_str}). *{p.title}*. {loc}")
    return refs


def _extract_bibliography_from_report(report: str) -> list[str]:
    """
    Pull the References section out of the LLM-generated report.
    Falls back to an empty list if the section is absent.
    """
    marker = "## References"
    if marker not in report:
        return []
    ref_block = report.split(marker, maxsplit=1)[1].strip()
    lines = [ln.strip() for ln in ref_block.splitlines() if ln.strip()]
    # Keep only lines that look like numbered references: [N] …
    return [ln for ln in lines if ln.startswith("[")]


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class SynthesisChain:
    """
    Wraps the LCEL synthesis pipeline.

    Usage
    ─────
        from langchain_groq import ChatGroq

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        chain = SynthesisChain(llm=llm)

        result = chain.run(
            user_query="What are recent advances in protein folding?",
            papers=[
                {
                    "title": "AlphaFold2 …",
                    "abstract": "…",
                    "authors": ["Jumper, J.", "Evans, R."],
                    "year": 2021,
                    "url": "https://…",
                },
                …
            ],
        )
        print(result.report)
    """

    def __init__(self, llm: Runnable) -> None:
        self._llm = llm
        self._chain: Runnable = _PROMPT | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_query: str,
        papers: list[dict[str, Any]],
    ) -> SynthesisResult:
        """
        Synchronous synthesis.

        Parameters
        ----------
        user_query:
            The original research question from the user.
        papers:
            List of PaperResult-compatible dicts (title, abstract, authors,
            year, url, doi, citation_count).  Extra keys are ignored.

        Returns
        -------
        SynthesisResult
            .report        – full Markdown string
            .bibliography  – ordered list of reference strings
            .paper_count   – number of source papers
        """
        synth_input = SynthesisInput.from_dicts(user_query, papers)
        prompt_vars = self._build_prompt_vars(synth_input)

        report: str = self._chain.invoke(prompt_vars)

        # Build canonical bibliography from the structured paper list so
        # callers always get machine-readable refs even if the LLM omits them.
        bibliography = _build_bibliography(synth_input.papers)

        return SynthesisResult(
            report=report,
            bibliography=bibliography,
            paper_count=len(synth_input.papers),
        )

    async def arun(
        self,
        user_query: str,
        papers: list[dict[str, Any]],
    ) -> SynthesisResult:
        """Async variant of :meth:`run`."""
        synth_input = SynthesisInput.from_dicts(user_query, papers)
        prompt_vars = self._build_prompt_vars(synth_input)

        report: str = await self._chain.ainvoke(prompt_vars)

        bibliography = _build_bibliography(synth_input.papers)
        return SynthesisResult(
            report=report,
            bibliography=bibliography,
            paper_count=len(synth_input.papers),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt_vars(synth_input: SynthesisInput) -> dict[str, Any]:
        return {
            "user_query": synth_input.user_query,
            "paper_count": len(synth_input.papers),
            "formatted_papers": _format_papers_for_prompt(synth_input.papers),
        }
    
# ---------------------------------------------------------------------------
# Manual Testing Block (Groq Edition)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from langchain_groq import ChatGroq

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print("Booting up Synthesizer (Powered by Groq/Llama3.3)...")

    groq_key = os.getenv("GROQ_KEY_1") or os.getenv("GROQ_KEY_2") or os.getenv("GROQ_KEY_3")
    if not groq_key:
        raise EnvironmentError("Set at least one of GROQ_KEY_1 / GROQ_KEY_2 / GROQ_KEY_3")

    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_key,
        temperature=0.2
    )

    # Initialize the Synthesizer
    synth = SynthesisChain(llm=llm)

    # Simulated data from our search tools
    test_query = "What are the latest advances in CRISPR for genetic blindness?"
    test_papers = [
        {
            "title": "Base Editing for Leber Congenital Amaurosis",
            "abstract": "We demonstrated that adenine base editors can restore visual function in mice by correcting the RPE65 mutation without double-stranded breaks.",
            "authors": ["Liu, D.", "Newby, J.", "Koblan, L."],
            "year": 2024,
            "url": "https://nature.com/articles/example1"
        },
        {
            "title": "Prime Editing in Human Retinal Organoids",
            "abstract": "Prime editing offers a versatile approach to correcting diverse mutations. Our study shows 15% efficiency in correcting USH2A mutations associated with Usher syndrome.",
            "authors": ["Busskamp, V.", "Sanjurjo-Soriano, C."],
            "year": 2023,
            "url": "https://cell.com/example2"
        },
        {
            "title": "CRISPR-Cas9 Clinical Trial Update for LCA10",
            "abstract": "Phase 1/2 clinical trials of EDIT-101 show safety in adults but limited efficacy in improving visual acuity across all cohorts.",
            "authors": ["Cideciyan, A.", "Pierce, E."],
            "year": 2022,
            "url": "https://nejm.org/example3"
        }
    ]

    print(f"\nGenerating Report for: {test_query}\n")
    
    try:
        # Run the synthesis
        result = synth.run(test_query, test_papers)
        
        print("\n=== FINAL RESEARCH REPORT ===")
        print(result.report)
        print("\n=== BIBLIOGRAPHY DATA (Extracted) ===")
        for ref in result.bibliography:
            print(f"- {ref}")
            
    except Exception as e:
        print(f"Error occurred: {e}")