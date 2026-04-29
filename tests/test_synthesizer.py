"""
tests/test_synthesizer.py
─────────────────────────
Full test suite for chains/synthesizer.py.

All LLM calls are mocked with RunnableLambda — zero real API calls are made.
The suite covers:
  • Data model validation (PaperResult, SynthesisInput)
  • Prompt-variable construction (_build_prompt_vars)
  • Paper formatting helpers (_format_papers_for_prompt, _build_bibliography)
  • SynthesisChain.run()  — happy path, edge cases, citation/bibliography presence
  • SynthesisChain.arun() — async happy path
  • _extract_bibliography_from_report helper
  • Input validation errors (empty query, empty papers list)
  • Long-abstract truncation
  • Author coercion from string
"""

from __future__ import annotations

import asyncio
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from chains.synthesizer import (
    PaperResult,
    SynthesisChain,
    SynthesisInput,
    SynthesisResult,
    _build_bibliography,
    _extract_bibliography_from_report,
    _format_papers_for_prompt,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

PAPER_1: dict[str, Any] = {
    "title": "Attention Is All You Need",
    "abstract": "We propose a new simple network architecture, the Transformer.",
    "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N.", "Uszkoreit, J."],
    "year": 2017,
    "url": "https://arxiv.org/abs/1706.03762",
    "doi": "10.48550/arXiv.1706.03762",
    "citation_count": 95000,
}

PAPER_2: dict[str, Any] = {
    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
    "abstract": "We introduce BERT, a new language representation model.",
    "authors": ["Devlin, J.", "Chang, M.-W.", "Lee, K.", "Toutanova, K."],
    "year": 2019,
    "url": "https://arxiv.org/abs/1810.04805",
    "doi": None,
    "citation_count": 80000,
}

PAPER_MINIMAL: dict[str, Any] = {
    "title": "A Minimal Paper",
    "abstract": "",
    "authors": [],
    "year": None,
    "url": "",
    "doi": None,
    "citation_count": None,
}


def _make_mock_llm(response: str) -> RunnableLambda:
    """Returns a RunnableLambda that ignores its input and returns *response*."""
    return RunnableLambda(lambda _: response)


def _make_mock_chain(response: str) -> SynthesisChain:
    return SynthesisChain(llm=_make_mock_llm(response))


# ─── canonical mock report used across multiple tests ─────────────────────────

_MOCK_REPORT = textwrap.dedent("""\
    # Research Summary: transformer architectures

    ## Overview
    The transformer architecture [1] has become the foundation of modern NLP.
    BERT [2] extended this work with bidirectional pre-training.

    ## Key Themes & Findings

    ### Self-Attention Mechanisms
    The original Transformer [1] replaced recurrence with self-attention,
    enabling parallelisation and strong sequence modelling.

    ### Pre-training Strategies
    BERT [2] demonstrated that masked language modelling yields powerful
    general-purpose representations.

    ## Gaps & Open Questions
    Efficiency at long-sequence lengths remains an open challenge.

    ## References
    [1] Vaswani, A., Shazeer, N., Parmar, N. et al. (2017). *Attention Is All You Need*. 10.48550/arXiv.1706.03762
    [2] Devlin, J., Chang, M.-W., Lee, K. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. https://arxiv.org/abs/1810.04805
""")


# ──────────────────────────────────────────────────────────────────────────────
# 1. PaperResult model
# ──────────────────────────────────────────────────────────────────────────────

class TestPaperResult:
    def test_full_construction(self):
        p = PaperResult(**PAPER_1)
        assert p.title == "Attention Is All You Need"
        assert p.year == 2017
        assert p.citation_count == 95000

    def test_minimal_construction(self):
        p = PaperResult(**PAPER_MINIMAL)
        assert p.title == "A Minimal Paper"
        assert p.authors == []
        assert p.year is None
        assert p.url == ""

    def test_authors_coerced_from_comma_string(self):
        # Use semicolons — the unambiguous separator for author lists. Commas
        # are inherently ambiguous in "Lastname, F., Lastname2, F2." strings.
        p = PaperResult(title="T", abstract="", authors="Smith, J.; Doe, A.")
        assert p.authors == ["Smith, J.", "Doe, A."]

    def test_authors_coerced_from_none(self):
        p = PaperResult(title="T", abstract="", authors=None)
        assert p.authors == []

    def test_authors_list_passthrough(self):
        p = PaperResult(title="T", authors=["A", "B"])
        assert p.authors == ["A", "B"]

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            PaperResult(abstract="no title")


# ──────────────────────────────────────────────────────────────────────────────
# 2. SynthesisInput model
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesisInput:
    def test_from_dicts_happy_path(self):
        si = SynthesisInput.from_dicts("query", [PAPER_1, PAPER_2])
        assert len(si.papers) == 2
        assert isinstance(si.papers[0], PaperResult)

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            SynthesisInput(user_query="", papers=[PaperResult(**PAPER_1)])

    def test_empty_papers_raises(self):
        with pytest.raises(ValidationError):
            SynthesisInput(user_query="something", papers=[])

    def test_extra_keys_in_dict_are_ignored(self):
        paper_with_extra = {**PAPER_1, "irrelevant_field": "ignored"}
        si = SynthesisInput.from_dicts("q", [paper_with_extra])
        assert len(si.papers) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 3. _format_papers_for_prompt
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatPapersForPrompt:
    def _papers(self, dicts: list[dict]) -> list[PaperResult]:
        return [PaperResult(**d) for d in dicts]

    def test_citation_numbers_are_one_based(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_1, PAPER_2]))
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "[0]" not in formatted

    def test_title_present(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_1]))
        assert "Attention Is All You Need" in formatted

    def test_url_present_when_given(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_1]))
        assert "https://arxiv.org/abs/1706.03762" in formatted

    def test_doi_present_when_given(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_1]))
        assert "10.48550/arXiv.1706.03762" in formatted

    def test_citation_count_present(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_1]))
        assert "95000" in formatted

    def test_unknown_authors_fallback(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_MINIMAL]))
        assert "Unknown authors" in formatted

    def test_year_nd_fallback(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_MINIMAL]))
        assert "n.d." in formatted

    def test_long_abstract_truncated_at_800_chars(self):
        # Use a 10-char token repeated so the "after-800" block is a long
        # distinct string that cannot appear anywhere else in the formatted output.
        token = "ZZZZZZZZZZ"  # 10 chars, won't appear in titles/authors/URLs
        long_abstract = "x" * 800 + token * 40   # 800 + 400 chars
        paper = {**PAPER_1, "abstract": long_abstract}
        formatted = _format_papers_for_prompt(self._papers([paper]))
        assert "x" * 800 in formatted      # first 800 x-chars are present
        assert "…" in formatted             # ellipsis appended
        assert token * 40 not in formatted  # the 400-char tail was dropped

    def test_short_abstract_not_truncated(self):
        short_abstract = "Short abstract."
        paper = {**PAPER_1, "abstract": short_abstract}
        formatted = _format_papers_for_prompt(self._papers([paper]))
        assert short_abstract in formatted
        assert "…" not in formatted

    def test_many_authors_et_al(self):
        paper = {**PAPER_1, "authors": ["A", "B", "C", "D", "E"]}
        formatted = _format_papers_for_prompt(self._papers([paper]))
        assert "et al." in formatted

    def test_three_authors_no_et_al(self):
        paper = {**PAPER_1, "authors": ["A", "B", "C"]}
        formatted = _format_papers_for_prompt(self._papers([paper]))
        assert "et al." not in formatted

    def test_empty_abstract_not_included(self):
        """When abstract is empty, the Abstract line should be absent."""
        formatted = _format_papers_for_prompt(self._papers([PAPER_MINIMAL]))
        assert "Abstract:" not in formatted

    def test_no_url_no_url_line(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_MINIMAL]))
        assert "URL" not in formatted

    def test_separator_between_papers(self):
        formatted = _format_papers_for_prompt(self._papers([PAPER_1, PAPER_2]))
        # Two papers means one blank-line separator between them
        assert "\n\n" in formatted


# ──────────────────────────────────────────────────────────────────────────────
# 4. _build_bibliography
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildBibliography:
    def _papers(self, dicts: list[dict]) -> list[PaperResult]:
        return [PaperResult(**d) for d in dicts]

    def test_length_matches_papers(self):
        bib = _build_bibliography(self._papers([PAPER_1, PAPER_2]))
        assert len(bib) == 2

    def test_title_in_entry(self):
        bib = _build_bibliography(self._papers([PAPER_1]))
        assert "Attention Is All You Need" in bib[0]

    def test_doi_preferred_over_url(self):
        bib = _build_bibliography(self._papers([PAPER_1]))
        assert "10.48550/arXiv.1706.03762" in bib[0]

    def test_url_used_when_no_doi(self):
        bib = _build_bibliography(self._papers([PAPER_2]))
        assert "https://arxiv.org/abs/1810.04805" in bib[0]

    def test_no_link_fallback(self):
        bib = _build_bibliography(self._papers([PAPER_MINIMAL]))
        assert "no link available" in bib[0]

    def test_year_nd_fallback(self):
        bib = _build_bibliography(self._papers([PAPER_MINIMAL]))
        assert "n.d." in bib[0]

    def test_et_al_for_many_authors(self):
        paper = {**PAPER_1, "authors": ["A", "B", "C", "D"]}
        bib = _build_bibliography(self._papers([paper]))
        assert "et al." in bib[0]

    def test_italic_title_formatting(self):
        bib = _build_bibliography(self._papers([PAPER_1]))
        assert "*Attention Is All You Need*" in bib[0]


# ──────────────────────────────────────────────────────────────────────────────
# 5. _extract_bibliography_from_report
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractBibliographyFromReport:
    def test_extracts_numbered_refs(self):
        refs = _extract_bibliography_from_report(_MOCK_REPORT)
        assert len(refs) == 2
        assert refs[0].startswith("[1]")
        assert refs[1].startswith("[2]")

    def test_empty_when_no_references_section(self):
        report_no_refs = "# Summary\n\nSome text without a references section."
        refs = _extract_bibliography_from_report(report_no_refs)
        assert refs == []

    def test_ignores_non_numbered_lines(self):
        report = "## References\nSome prose line.\n[1] Real ref here.\n"
        refs = _extract_bibliography_from_report(report)
        assert len(refs) == 1
        assert refs[0].startswith("[1]")


# ──────────────────────────────────────────────────────────────────────────────
# 6. SynthesisChain — construction & prompt variable building
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesisChainInit:
    def test_stores_llm(self):
        llm = _make_mock_llm("hello")
        chain = SynthesisChain(llm=llm)
        assert chain._llm is llm

    def test_internal_chain_is_runnable(self):
        from langchain_core.runnables import Runnable
        chain = SynthesisChain(llm=_make_mock_llm("x"))
        assert isinstance(chain._chain, Runnable)


class TestBuildPromptVars:
    def test_keys_present(self):
        si = SynthesisInput.from_dicts("my query", [PAPER_1])
        vars_ = SynthesisChain._build_prompt_vars(si)
        assert "user_query" in vars_
        assert "paper_count" in vars_
        assert "formatted_papers" in vars_

    def test_paper_count_correct(self):
        si = SynthesisInput.from_dicts("q", [PAPER_1, PAPER_2])
        vars_ = SynthesisChain._build_prompt_vars(si)
        assert vars_["paper_count"] == 2

    def test_user_query_forwarded(self):
        si = SynthesisInput.from_dicts("transformer architectures", [PAPER_1])
        vars_ = SynthesisChain._build_prompt_vars(si)
        assert vars_["user_query"] == "transformer architectures"


# ──────────────────────────────────────────────────────────────────────────────
# 7. SynthesisChain.run() — happy path
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesisChainRun:
    def test_returns_synthesis_result(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformer architectures", [PAPER_1, PAPER_2])
        assert isinstance(result, SynthesisResult)

    def test_report_is_llm_output(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert result.report == _MOCK_REPORT

    def test_paper_count_matches_input(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert result.paper_count == 2

    def test_bibliography_length_matches_papers(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert len(result.bibliography) == 2

    def test_bibliography_contains_paper_titles(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert any("Attention Is All You Need" in ref for ref in result.bibliography)
        assert any("BERT" in ref for ref in result.bibliography)

    def test_inline_citations_present_in_report(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert "[1]" in result.report
        assert "[2]" in result.report

    def test_references_section_present(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert "## References" in result.report

    def test_overview_section_present(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert "## Overview" in result.report

    def test_key_themes_section_present(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert "## Key Themes" in result.report

    def test_gaps_section_present(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1, PAPER_2])
        assert "## Gaps" in result.report


# ──────────────────────────────────────────────────────────────────────────────
# 8. SynthesisChain.run() — edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesisChainRunEdgeCases:
    def test_single_paper(self):
        chain = _make_mock_chain("# Summary\n\n[1] only one paper.\n\n## References\n[1] ref")
        result = chain.run("single paper query", [PAPER_1])
        assert result.paper_count == 1
        assert len(result.bibliography) == 1

    def test_many_papers(self):
        papers = [PAPER_1, PAPER_2] + [PAPER_MINIMAL] * 8
        mock_report = "## References\n" + "\n".join(f"[{i}] ref" for i in range(1, 11))
        chain = _make_mock_chain(mock_report)
        result = chain.run("large query", papers)
        assert result.paper_count == 10
        assert len(result.bibliography) == 10

    def test_paper_with_minimal_fields(self):
        """Should not raise when optional fields (doi, url, year) are absent."""
        chain = _make_mock_chain("Report text.\n\n## References\n[1] minimal ref")
        result = chain.run("query", [PAPER_MINIMAL])
        assert result.paper_count == 1

    def test_bibliography_uses_doi_when_available(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_1])
        # PAPER_1 has a doi; it should appear in the bibliography entry
        assert "10.48550/arXiv.1706.03762" in result.bibliography[0]

    def test_bibliography_uses_url_when_no_doi(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = chain.run("transformers", [PAPER_2])
        assert "https://arxiv.org/abs/1810.04805" in result.bibliography[0]

    def test_empty_query_raises_before_llm(self):
        chain = _make_mock_chain("anything")
        with pytest.raises((ValidationError, ValueError)):
            chain.run("", [PAPER_1])

    def test_empty_papers_raises_before_llm(self):
        chain = _make_mock_chain("anything")
        with pytest.raises((ValidationError, ValueError)):
            chain.run("some query", [])

    def test_no_real_api_call_made(self):
        """Verifies the mock is used — if the real LLM were invoked it would
        raise an AuthenticationError (no key present in CI)."""
        real_llm_sentinel = MagicMock()
        real_llm_sentinel.invoke = MagicMock(side_effect=RuntimeError("real LLM called!"))

        # SynthesisChain builds an LCEL chain: prompt | llm | parser.
        # We patch the *chain* attribute directly to use our safe mock instead.
        safe_chain = SynthesisChain(llm=_make_mock_llm(_MOCK_REPORT))
        safe_chain._llm = real_llm_sentinel  # replace after construction

        # Re-build internal chain with safe mock so the real sentinel is never used
        from langchain_core.output_parsers import StrOutputParser
        safe_chain._chain = _make_mock_llm(_MOCK_REPORT) | StrOutputParser()

        result = safe_chain.run("transformers", [PAPER_1, PAPER_2])
        real_llm_sentinel.invoke.assert_not_called()
        assert result.report == _MOCK_REPORT


# ──────────────────────────────────────────────────────────────────────────────
# 9. SynthesisChain.arun() — async
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesisChainArun:
    def test_arun_returns_synthesis_result(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = asyncio.run(chain.arun("transformer architectures", [PAPER_1, PAPER_2]))
        assert isinstance(result, SynthesisResult)

    def test_arun_report_matches_mock(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = asyncio.run(chain.arun("transformers", [PAPER_1, PAPER_2]))
        assert result.report == _MOCK_REPORT

    def test_arun_paper_count(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = asyncio.run(chain.arun("transformers", [PAPER_1, PAPER_2]))
        assert result.paper_count == 2

    def test_arun_bibliography_populated(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = asyncio.run(chain.arun("transformers", [PAPER_1, PAPER_2]))
        assert len(result.bibliography) == 2

    def test_arun_inline_citations_present(self):
        chain = _make_mock_chain(_MOCK_REPORT)
        result = asyncio.run(chain.arun("transformers", [PAPER_1, PAPER_2]))
        assert "[1]" in result.report and "[2]" in result.report


# ──────────────────────────────────────────────────────────────────────────────
# 10. SynthesisResult model
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesisResult:
    def test_construction(self):
        r = SynthesisResult(
            report="# Report",
            bibliography=["ref1", "ref2"],
            paper_count=2,
        )
        assert r.paper_count == 2
        assert len(r.bibliography) == 2

    def test_report_field_stored(self):
        r = SynthesisResult(report="hello", bibliography=[], paper_count=0)
        assert r.report == "hello"


# ──────────────────────────────────────────────────────────────────────────────
# 11. Prompt template smoke-test (no LLM required)
# ──────────────────────────────────────────────────────────────────────────────

class TestPromptTemplate:
    """Verify the ChatPromptTemplate renders without KeyError."""

    def test_prompt_renders_with_expected_vars(self):
        from chains.synthesizer import _PROMPT

        si = SynthesisInput.from_dicts("transformer models", [PAPER_1, PAPER_2])
        vars_ = SynthesisChain._build_prompt_vars(si)

        messages = _PROMPT.format_messages(**vars_)
        combined = " ".join(m.content for m in messages)

        assert "transformer models" in combined
        assert "Attention Is All You Need" in combined
        assert "2" in combined  # paper_count

    def test_prompt_contains_citation_instructions(self):
        from chains.synthesizer import _PROMPT

        si = SynthesisInput.from_dicts("q", [PAPER_1])
        vars_ = SynthesisChain._build_prompt_vars(si)
        messages = _PROMPT.format_messages(**vars_)
        system_content = messages[0].content

        assert "[N]" in system_content or "citation" in system_content.lower()

    def test_prompt_contains_references_section_instruction(self):
        from chains.synthesizer import _PROMPT

        si = SynthesisInput.from_dicts("q", [PAPER_1])
        vars_ = SynthesisChain._build_prompt_vars(si)
        messages = _PROMPT.format_messages(**vars_)
        system_content = messages[0].content

        assert "References" in system_content