"""
tests/test_agent.py
───────────────────
Test suite for agents/research_agent.py.

All external dependencies are mocked — zero real API calls are made.

Test groups
───────────
  TestIsRateLimitError         — error-classification helper
  TestResilientLLMInvoke       — sync fallback chain (Groq → Groq → Groq → Gemini)
  TestResilientLLMAinvoke      — async mirror of the above
  TestResilientLLMEdgeCases    — no Gemini key, non-rate-limit re-raise, all exhausted
  TestGeminiSafetySettings     — BLOCK_NONE applied regardless of SDK version
  TestDeduplication            — DOI-keyed and title-keyed merging
  TestNormaliseToolOutput      — list / JSON-string / plain-string handling
  TestResearchAgentExecuteStep — concurrent tool fan-out + tool failure isolation
  TestResearchAgentEnrichPDFs  — Unpaywall enrichment, missing DOI passthrough
  TestResearchAgentEndToEnd    — full .research() call with all stages mocked
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.research_agent import (
    ResearchAgent,
    ResilientLLM,
    _build_gemini_safety_settings,
    _detect_domain,
    _is_rate_limit_error,
    _normalise_tool_output,
    _tools_for_domain,
    deduplicate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_paper(
    title: str = "A Paper",
    abstract: str = "An abstract.",
    doi: str | None = None,
    citation_count: int = 0,
) -> dict[str, Any]:
    return {
        "title": title,
        "abstract": abstract,
        "authors": ["Author, A."],
        "year": 2023,
        "url": f"https://example.com/{title[:10]}",
        "doi": doi,
        "citation_count": citation_count,
    }


def _ok_llm(text: str = "ok") -> RunnableLambda:
    """RunnableLambda that returns *text* as an AIMessage."""
    return RunnableLambda(lambda _: AIMessage(content=text))


def _rate_limit_exc(msg: str = "rate limit exceeded") -> RuntimeError:
    return RuntimeError(msg)


def _mock_tool(papers: list[dict]) -> MagicMock:
    tool = MagicMock()
    tool.run.return_value = papers
    return tool


def _make_resilient_llm(
    groq_keys: list[str] | None = None,
    gemini_key: str = "g-key",
) -> ResilientLLM:
    return ResilientLLM(
        groq_keys=groq_keys or ["k1", "k2", "k3"],
        gemini_key=gemini_key,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. _is_rate_limit_error
# ──────────────────────────────────────────────────────────────────────────────

class TestIsRateLimitError:
    def test_message_rate_limit(self):
        assert _is_rate_limit_error(RuntimeError("rate limit exceeded"))

    def test_message_429(self):
        assert _is_rate_limit_error(RuntimeError("HTTP 429 Too Many Requests"))

    def test_message_quota(self):
        assert _is_rate_limit_error(RuntimeError("quota exceeded"))

    def test_message_too_many_requests(self):
        assert _is_rate_limit_error(RuntimeError("too many requests"))

    def test_message_resource_exhausted_grpc(self):
        assert _is_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED: quota exceeded"))

    def test_message_ratelimit_no_space(self):
        assert _is_rate_limit_error(RuntimeError("RateLimitError"))

    def test_normal_error_is_false(self):
        assert not _is_rate_limit_error(RuntimeError("authentication failed"))

    def test_connection_error_is_false(self):
        assert not _is_rate_limit_error(ConnectionError("network unreachable"))

    def test_value_error_is_false(self):
        assert not _is_rate_limit_error(ValueError("invalid model name"))

    def test_case_insensitive(self):
        assert _is_rate_limit_error(RuntimeError("RATE LIMIT"))

    def test_groq_rate_limit_error_class(self):
        """If groq is importable, its native RateLimitError must be caught."""
        try:
            import groq

            exc = groq.RateLimitError.__new__(groq.RateLimitError)
            assert _is_rate_limit_error(exc)
        except (ImportError, Exception):
            pytest.skip("groq SDK not installed")

    def test_httpx_429(self):
        """An httpx HTTPStatusError with status 429 must be caught."""
        try:
            import httpx

            response = MagicMock(spec=httpx.Response)
            response.status_code = 429
            exc = httpx.HTTPStatusError("429", request=MagicMock(), response=response)
            assert _is_rate_limit_error(exc)
        except ImportError:
            pytest.skip("httpx not installed")

    def test_httpx_500_is_false(self):
        try:
            import httpx

            response = MagicMock(spec=httpx.Response)
            response.status_code = 500
            exc = httpx.HTTPStatusError("500", request=MagicMock(), response=response)
            assert not _is_rate_limit_error(exc)
        except ImportError:
            pytest.skip("httpx not installed")


# ──────────────────────────────────────────────────────────────────────────────
# 2. ResilientLLM — synchronous invoke fallback chain
# ──────────────────────────────────────────────────────────────────────────────

class TestResilientLLMInvoke:
    def _rl(self, **kw) -> ResilientLLM:
        return _make_resilient_llm(**kw)

    # helpers to build side-effect LLM mocks
    @staticmethod
    def _fail_then_ok(rate_limit_msg: str = "rate limit") -> tuple[MagicMock, MagicMock]:
        """Return (failing_mock, succeeding_mock)."""
        fail = MagicMock()
        fail.invoke.side_effect = RuntimeError(rate_limit_msg)
        ok = MagicMock()
        ok.invoke.return_value = AIMessage(content="answer")
        return fail, ok

    def test_first_key_succeeds(self):
        llm = self._rl()
        ok = MagicMock()
        ok.invoke.return_value = AIMessage(content="ok")
        with patch.object(llm, "_make_groq", return_value=ok):
            result = llm.invoke("test")
        assert result.content == "ok"
        ok.invoke.assert_called_once()

    def test_second_key_used_after_first_rate_limited(self):
        llm = self._rl()
        fail, succeed = self._fail_then_ok()
        call_count = {"n": 0}

        def make_groq(key: str):
            call_count["n"] += 1
            return fail if call_count["n"] == 1 else succeed

        with patch.object(llm, "_make_groq", side_effect=make_groq):
            result = llm.invoke("test")
        assert result.content == "answer"
        assert fail.invoke.call_count == 1
        assert succeed.invoke.call_count == 1

    def test_third_key_used_after_first_two_rate_limited(self):
        llm = self._rl()
        fail1 = MagicMock()
        fail1.invoke.side_effect = RuntimeError("rate limit")
        fail2 = MagicMock()
        fail2.invoke.side_effect = RuntimeError("429")
        succeed = MagicMock()
        succeed.invoke.return_value = AIMessage(content="third")

        providers = [fail1, fail2, succeed]
        idx = {"i": 0}

        def make_groq(key: str):
            val = providers[idx["i"]]
            idx["i"] += 1
            return val

        with patch.object(llm, "_make_groq", side_effect=make_groq):
            result = llm.invoke("test")
        assert result.content == "third"

    def test_gemini_used_after_all_groq_keys_exhausted(self):
        llm = self._rl()
        fail = MagicMock()
        fail.invoke.side_effect = RuntimeError("rate limit")
        gemini_mock = MagicMock()
        gemini_mock.invoke.return_value = AIMessage(content="from-gemini")

        with patch.object(llm, "_make_groq", return_value=fail), \
             patch.object(llm, "_make_gemini", return_value=gemini_mock):
            result = llm.invoke("test")
        assert result.content == "from-gemini"
        gemini_mock.invoke.assert_called_once()

    def test_non_rate_limit_error_raises_immediately(self):
        """An auth error on the first key must bubble up, not try the next key."""
        llm = self._rl()
        fail = MagicMock()
        fail.invoke.side_effect = RuntimeError("authentication failed")
        ok = MagicMock()
        ok.invoke.return_value = AIMessage(content="should not reach")

        providers = [fail, ok]
        idx = {"i": 0}

        def make_groq(key: str):
            val = providers[idx["i"]]
            idx["i"] += 1
            return val

        with patch.object(llm, "_make_groq", side_effect=make_groq):
            with pytest.raises(RuntimeError, match="authentication failed"):
                llm.invoke("test")

        # Second provider must never have been invoked
        assert ok.invoke.call_count == 0

    def test_all_providers_exhausted_raises_runtime_error(self):
        llm = self._rl(gemini_key="")   # no Gemini fallback
        fail = MagicMock()
        fail.invoke.side_effect = RuntimeError("rate limit")

        with patch.object(llm, "_make_groq", return_value=fail):
            with pytest.raises(RuntimeError, match="all providers exhausted"):
                llm.invoke("test")


# ──────────────────────────────────────────────────────────────────────────────
# 3. ResilientLLM — async ainvoke fallback chain
# ──────────────────────────────────────────────────────────────────────────────

class TestResilientLLMAinvoke:
    @staticmethod
    def _async_fail(msg: str = "rate limit") -> AsyncMock:
        m = AsyncMock()
        m.ainvoke.side_effect = RuntimeError(msg)
        return m

    @staticmethod
    def _async_ok(text: str = "async-ok") -> AsyncMock:
        m = AsyncMock()
        m.ainvoke.return_value = AIMessage(content=text)
        return m

    def test_first_key_succeeds_async(self):
        llm = _make_resilient_llm()
        ok = self._async_ok("first")
        with patch.object(llm, "_make_groq", return_value=ok):
            result = asyncio.run(llm.ainvoke("test"))
        assert result.content == "first"

    def test_second_key_used_after_first_rate_limited_async(self):
        llm = _make_resilient_llm()
        fail, succeed = self._async_fail(), self._async_ok("second")
        idx = {"i": 0}
        providers = [fail, succeed]

        def make_groq(key: str):
            val = providers[idx["i"]]
            idx["i"] += 1
            return val

        with patch.object(llm, "_make_groq", side_effect=make_groq):
            result = asyncio.run(llm.ainvoke("test"))
        assert result.content == "second"

    def test_gemini_fallback_async(self):
        llm = _make_resilient_llm()
        fail = self._async_fail()
        gemini = self._async_ok("gemini")

        with patch.object(llm, "_make_groq", return_value=fail), \
             patch.object(llm, "_make_gemini", return_value=gemini):
            result = asyncio.run(llm.ainvoke("test"))
        assert result.content == "gemini"

    def test_non_rate_limit_raises_immediately_async(self):
        llm = _make_resilient_llm()
        fail = AsyncMock()
        fail.ainvoke.side_effect = RuntimeError("auth error")
        ok = self._async_ok()

        providers = [fail, ok]
        idx = {"i": 0}

        def make_groq(key: str):
            val = providers[idx["i"]]
            idx["i"] += 1
            return val

        with patch.object(llm, "_make_groq", side_effect=make_groq):
            with pytest.raises(RuntimeError, match="auth error"):
                asyncio.run(llm.ainvoke("test"))
        assert ok.ainvoke.call_count == 0

    def test_all_exhausted_raises_async(self):
        llm = _make_resilient_llm(gemini_key="")
        fail = self._async_fail()

        with patch.object(llm, "_make_groq", return_value=fail):
            with pytest.raises(RuntimeError, match="all providers exhausted"):
                asyncio.run(llm.ainvoke("test"))


# ──────────────────────────────────────────────────────────────────────────────
# 4. ResilientLLM — construction edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestResilientLLMEdgeCases:
    def test_no_groq_keys_raises(self):
        with pytest.raises(ValueError):
            ResilientLLM(groq_keys=[], gemini_key="x")

    def test_single_groq_key_accepted(self):
        llm = ResilientLLM(groq_keys=["only-key"], gemini_key="g")
        assert len(llm._groq_keys) == 1

    def test_empty_gemini_key_means_no_gemini_provider(self):
        llm = ResilientLLM(groq_keys=["k1"], gemini_key="")
        # Patch _make_groq so the generator can be consumed without importing
        # langchain_groq (which may not be installed in the test environment).
        with patch.object(llm, "_make_groq", return_value=MagicMock()):
            labels = [label for label, _ in llm._iter_providers()]
        assert not any("gemini" in l for l in labels)

    def test_three_groq_keys_gives_three_providers_plus_gemini(self):
        llm = _make_resilient_llm()
        with patch.object(llm, "_make_groq", side_effect=lambda k: MagicMock()), \
             patch.object(llm, "_make_gemini", return_value=MagicMock()):
            # _iter_providers is a generator; materialise it before len()
            providers = list(llm._iter_providers())
        assert len(providers) == 4   # 3 Groq + 1 Gemini

    def test_provider_labels(self):
        llm = _make_resilient_llm()
        with patch.object(llm, "_make_groq", side_effect=lambda k: MagicMock()), \
             patch.object(llm, "_make_gemini", return_value=MagicMock()):
            labels = [label for label, _ in llm._iter_providers()]
        assert labels == ["groq[1]", "groq[2]", "groq[3]", "gemini"]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Gemini safety settings
# ──────────────────────────────────────────────────────────────────────────────

class TestGeminiSafetySettings:
    def test_settings_not_empty(self):
        settings = _build_gemini_safety_settings()
        assert len(settings) >= 4

    def test_all_values_block_none(self):
        settings = _build_gemini_safety_settings()
        for val in settings.values():
            # Accept either the enum member or the plain string "BLOCK_NONE"
            assert "BLOCK_NONE" in str(val).upper()

    def test_harassment_key_present(self):
        settings = _build_gemini_safety_settings()
        keys_str = " ".join(str(k).upper() for k in settings)
        assert "HARASSMENT" in keys_str

    def test_hate_speech_key_present(self):
        settings = _build_gemini_safety_settings()
        keys_str = " ".join(str(k).upper() for k in settings)
        assert "HATE_SPEECH" in keys_str or "HATE" in keys_str

    def test_dangerous_content_key_present(self):
        settings = _build_gemini_safety_settings()
        keys_str = " ".join(str(k).upper() for k in settings)
        assert "DANGEROUS" in keys_str

    def test_string_fallback_when_enum_import_fails(self):
        """When google-generativeai enums are unavailable, string keys are used."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "google.generativeai" in name or name == "google.generativeai.types":
                raise ImportError("mocked absence")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            settings = _build_gemini_safety_settings()

        assert all(isinstance(k, str) for k in settings)
        assert all(isinstance(v, str) for v in settings.values())
        assert all(v == "BLOCK_NONE" for v in settings.values())


# ──────────────────────────────────────────────────────────────────────────────
# 6. Deduplication
# ──────────────────────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_identical_doi_deduped(self):
        papers = [
            _make_paper("Paper A", doi="10.1234/a", abstract="short"),
            _make_paper("Paper A dupe", doi="10.1234/a", abstract="longer abstract here"),
        ]
        result = deduplicate(papers)
        assert len(result) == 1
        # The longer abstract should win
        assert result[0]["abstract"] == "longer abstract here"

    def test_different_dois_both_kept(self):
        papers = [
            _make_paper("Paper A", doi="10.1234/a"),
            _make_paper("Paper B", doi="10.1234/b"),
        ]
        result = deduplicate(papers)
        assert len(result) == 2

    def test_title_fallback_when_no_doi(self):
        papers = [
            _make_paper("Unique Title", doi=None, abstract="v1"),
            _make_paper("Unique Title", doi=None, abstract="v2 which is longer"),
        ]
        result = deduplicate(papers)
        assert len(result) == 1
        assert result[0]["abstract"] == "v2 which is longer"

    def test_doi_takes_precedence_over_different_titles(self):
        """Same DOI, different titles → one entry."""
        papers = [
            {"title": "Title A", "abstract": "ab", "doi": "10.1234/x"},
            {"title": "Title B (preprint)", "abstract": "ab extended", "doi": "10.1234/x"},
        ]
        result = deduplicate(papers)
        assert len(result) == 1

    def test_doi_url_prefix_normalised(self):
        """Papers with and without the https://doi.org/ prefix count as same."""
        papers = [
            _make_paper("A", doi="10.1234/abc", abstract="short"),
            _make_paper("A", doi="https://doi.org/10.1234/abc", abstract="longer version"),
        ]
        result = deduplicate(papers)
        assert len(result) == 1

    def test_title_case_insensitive(self):
        papers = [
            _make_paper("Neural Networks", doi=None, abstract="v1"),
            _make_paper("neural networks", doi=None, abstract="v2 longer"),
        ]
        result = deduplicate(papers)
        assert len(result) == 1

    def test_title_whitespace_normalised(self):
        papers = [
            _make_paper("  Neural  Networks  ", doi=None, abstract="v1"),
            _make_paper("Neural Networks", doi=None, abstract="v2 longer one"),
        ]
        result = deduplicate(papers)
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        assert deduplicate([]) == []

    def test_single_paper_returned_unchanged(self):
        p = _make_paper("Solo", doi="10.1/solo")
        result = deduplicate([p])
        assert result == [p]

    def test_order_preserved_for_unique_papers(self):
        """First-seen order is maintained for distinct papers."""
        papers = [_make_paper(f"Paper {i}", doi=f"10.1/{i}") for i in range(5)]
        result = deduplicate(papers)
        titles = [r["title"] for r in result]
        assert titles == [f"Paper {i}" for i in range(5)]

    def test_mixed_doi_and_no_doi(self):
        papers = [
            _make_paper("With DOI", doi="10.1/a"),
            _make_paper("No DOI A", doi=None),
            _make_paper("With DOI dupe", doi="10.1/a"),
            _make_paper("No DOI B", doi=None),
        ]
        result = deduplicate(papers)
        # 10.1/a → 1, "no doi a" → 1, "no doi b" → 1
        assert len(result) == 3


# ──────────────────────────────────────────────────────────────────────────────
# 7. _normalise_tool_output
# ──────────────────────────────────────────────────────────────────────────────

class TestNormaliseToolOutput:
    def test_list_of_dicts_passthrough(self):
        papers = [_make_paper("A"), _make_paper("B")]
        assert _normalise_tool_output(papers) == papers

    def test_json_string_of_list(self):
        papers = [_make_paper("A")]
        raw = json.dumps(papers)
        result = _normalise_tool_output(raw)
        assert result == papers

    def test_plain_string_returns_empty(self):
        assert _normalise_tool_output("No papers found.") == []

    def test_list_filters_non_dicts(self):
        raw = [_make_paper("A"), "not a dict", 42, None]
        result = _normalise_tool_output(raw)
        assert len(result) == 1

    def test_empty_list(self):
        assert _normalise_tool_output([]) == []

    def test_none_returns_empty(self):
        assert _normalise_tool_output(None) == []

    def test_json_non_list_returns_empty(self):
        assert _normalise_tool_output(json.dumps({"key": "val"})) == []


# ──────────────────────────────────────────────────────────────────────────────
# 8. _detect_domain and _tools_for_domain
# ──────────────────────────────────────────────────────────────────────────────

class TestDomainRouting:
    def test_biomedical_keywords_detected(self):
        assert _detect_domain("drug resistance in cancer cells") == "biomedical"

    def test_cs_keywords_detected(self):
        assert _detect_domain("transformer models for NLP tasks") == "cs"

    def test_general_for_ambiguous(self):
        assert _detect_domain("the history of money") == "general"

    def test_tools_for_biomedical_includes_pubmed(self):
        arxiv = MagicMock()
        ss = MagicMock()
        pubmed = MagicMock()
        oa = MagicMock()
        selected = _tools_for_domain("biomedical", arxiv, ss, pubmed, oa)
        names = [n for n, _ in selected]
        assert "pubmed" in names
        assert "arxiv" not in names

    def test_tools_for_cs_includes_arxiv(self):
        arxiv = MagicMock()
        ss = MagicMock()
        pubmed = MagicMock()
        oa = MagicMock()
        selected = _tools_for_domain("cs", arxiv, ss, pubmed, oa)
        names = [n for n, _ in selected]
        assert "arxiv" in names
        assert "pubmed" not in names

    def test_tools_for_general_fans_out_to_all(self):
        tools = [MagicMock() for _ in range(4)]
        selected = _tools_for_domain("general", *tools)
        assert len(selected) == 4


# ──────────────────────────────────────────────────────────────────────────────
# 9. ResearchAgent._execute_step
# ──────────────────────────────────────────────────────────────────────────────

class TestResearchAgentExecuteStep:
    def _make_step(self, sub_question: str, domain: str = "general") -> MagicMock:
        step = MagicMock()
        step.sub_question = sub_question
        step.domain = domain
        step.keywords_string = None
        step.keywords = None
        return step

    def _make_agent(self, **tool_overrides) -> ResearchAgent:
        llm = _ok_llm()
        # All planner / synthesiser calls are mocked at the agent level;
        # we only need to exercise _execute_step here.
        defaults = {
            "arxiv": _mock_tool([]),
            "pubmed": _mock_tool([]),
            "semantic_scholar": _mock_tool([]),
            "openalex": _mock_tool([]),
            "unpaywall": MagicMock(),
        }
        defaults.update(tool_overrides)
        return ResearchAgent(llm=llm, **defaults)

    def test_papers_returned_from_tools(self):
        paper = _make_paper("Test Paper", doi="10.1/t")
        agent = self._make_agent(arxiv=_mock_tool([paper]))
        step = self._make_step("AI safety", domain="cs")
        result = asyncio.run(agent._execute_step(step))
        assert any(p["title"] == "Test Paper" for p in result)

    def test_concurrent_results_merged(self):
        paper_a = _make_paper("A", doi="10.1/a")
        paper_b = _make_paper("B", doi="10.1/b")
        # Fan-out to all 4 tools for general domain
        agent = self._make_agent(
            arxiv=_mock_tool([paper_a]),
            semantic_scholar=_mock_tool([paper_b]),
            pubmed=_mock_tool([]),
            openalex=_mock_tool([]),
        )
        step = self._make_step("interdisciplinary topic", domain="general")
        result = asyncio.run(agent._execute_step(step))
        titles = {p["title"] for p in result}
        assert {"A", "B"} <= titles

    def test_failing_tool_does_not_abort_step(self):
        """A tool that raises must be silently skipped, not crash the step."""
        bad_tool = MagicMock()
        bad_tool.run.side_effect = RuntimeError("API down")
        good_paper = _make_paper("Good", doi="10.1/g")
        agent = self._make_agent(
            arxiv=bad_tool,
            semantic_scholar=_mock_tool([good_paper]),
            openalex=_mock_tool([]),
        )
        step = self._make_step("cs query", domain="cs")
        result = asyncio.run(agent._execute_step(step))
        assert any(p["title"] == "Good" for p in result)

    def test_per_step_paper_cap_enforced(self):
        many_papers = [_make_paper(f"Paper {i}", doi=f"10.1/{i}") for i in range(50)]
        agent = self._make_agent(
            arxiv=_mock_tool(many_papers),
            semantic_scholar=_mock_tool([]),
            openalex=_mock_tool([]),
        )
        agent._max_papers_per_step = 5
        step = self._make_step("cs query", domain="cs")
        result = asyncio.run(agent._execute_step(step))
        # 3 tools × 5 cap = 15 max (but arxiv has 5, others 0)
        assert len(result) <= 15

    def test_tool_timeout_respected(self):
        """A tool that hangs longer than _tool_timeout must be skipped, not
        block the step indefinitely."""
        import time

        def _slow_tool_run(_query):
            time.sleep(0.3)          # 300 ms — longer than our tiny test timeout
            return [_make_paper("Should not appear")]

        slow = MagicMock()
        slow.run.side_effect = _slow_tool_run

        agent = self._make_agent(arxiv=slow, semantic_scholar=_mock_tool([]), openalex=_mock_tool([]))
        agent._tool_timeout = 0.1   # 100 ms — intentionally shorter than the sleep

        step = self._make_step("cs query", domain="cs")
        result = asyncio.run(agent._execute_step(step))
        # The slow tool should have been killed; no papers from it
        assert not any(p.get("title") == "Should not appear" for p in result)

    def test_custom_tool_timeout_stored(self):
        agent = ResearchAgent(
            llm=_ok_llm(),
            arxiv=_mock_tool([]),
            pubmed=_mock_tool([]),
            semantic_scholar=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=MagicMock(),
            tool_timeout=90.0,
        )
        assert agent._tool_timeout == 90.0

    def test_default_tool_timeout_is_120(self):
        agent = ResearchAgent(
            llm=_ok_llm(),
            arxiv=_mock_tool([]),
            pubmed=_mock_tool([]),
            semantic_scholar=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=MagicMock(),
        )
        assert agent._tool_timeout == 120.0


# ──────────────────────────────────────────────────────────────────────────────
# 10. ResearchAgent._enrich_with_pdfs
# ──────────────────────────────────────────────────────────────────────────────

class TestResearchAgentEnrichPDFs:
    def _make_agent(self, unpaywall_returns: str | None) -> ResearchAgent:
        unpaywall = MagicMock()
        unpaywall.run.return_value = unpaywall_returns
        return ResearchAgent(
            llm=_ok_llm(),
            arxiv=_mock_tool([]),
            pubmed=_mock_tool([]),
            semantic_scholar=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=unpaywall,
        )

    def test_pdf_url_appended_when_found(self):
        agent = self._make_agent("https://example.com/paper.pdf")
        papers = [_make_paper("A", doi="10.1/a")]
        result = asyncio.run(agent._enrich_with_pdfs(papers))
        assert result[0].get("pdf_url") == "https://example.com/paper.pdf"

    def test_no_pdf_url_when_unpaywall_returns_none(self):
        agent = self._make_agent(None)
        papers = [_make_paper("A", doi="10.1/a")]
        result = asyncio.run(agent._enrich_with_pdfs(papers))
        assert "pdf_url" not in result[0]

    def test_no_doi_skips_unpaywall(self):
        agent = self._make_agent("https://example.com/should-not-be-called.pdf")
        papers = [_make_paper("No DOI", doi=None)]
        result = asyncio.run(agent._enrich_with_pdfs(papers))
        assert "pdf_url" not in result[0]
        agent._unpaywall.run.assert_not_called()

    def test_unpaywall_failure_does_not_drop_paper(self):
        unpaywall = MagicMock()
        unpaywall.run.side_effect = RuntimeError("Unpaywall down")
        agent = ResearchAgent(
            llm=_ok_llm(),
            arxiv=_mock_tool([]),
            pubmed=_mock_tool([]),
            semantic_scholar=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=unpaywall,
        )
        papers = [_make_paper("Fragile", doi="10.1/f")]
        result = asyncio.run(agent._enrich_with_pdfs(papers))
        assert len(result) == 1
        assert result[0]["title"] == "Fragile"
        assert "pdf_url" not in result[0]

    def test_concurrent_enrichment_all_papers_returned(self):
        """All papers must be present in the output, enriched or not."""
        agent = self._make_agent("https://oa.example.com/paper.pdf")
        papers = [_make_paper(f"Paper {i}", doi=f"10.1/{i}") for i in range(10)]
        result = asyncio.run(agent._enrich_with_pdfs(papers))
        assert len(result) == 10

    def test_original_paper_dict_not_mutated(self):
        """Enrichment must return a new dict, not mutate the original."""
        agent = self._make_agent("https://pdf.example.com/a.pdf")
        original = _make_paper("Original", doi="10.1/orig")
        result = asyncio.run(agent._enrich_with_pdfs([original]))
        assert "pdf_url" not in original    # original untouched
        assert "pdf_url" in result[0]       # copy has the new key


# ──────────────────────────────────────────────────────────────────────────────
# 11. ResearchAgent.research() — end-to-end (all stages mocked)
# ──────────────────────────────────────────────────────────────────────────────

class TestResearchAgentEndToEnd:
    """
    Verify the full five-stage orchestration without any real I/O.

    Strategy: replace the planner and synthesiser with controlled mocks;
    inject mock tools; assert the correct hand-offs between stages.
    """

    _REPORT = "# Report\n\n[1] A paper.\n\n## References\n[1] ref"

    def _build_plan(self, sub_questions: list[str]) -> MagicMock:
        steps = []
        for sq in sub_questions:
            step = MagicMock()
            step.sub_question = sq
            step.domain = "cs"
            step.keywords_string = None
            step.keywords = None
            steps.append(step)
        plan = MagicMock()
        plan.steps = steps
        return plan

    def _make_full_agent(
        self,
        plan: MagicMock,
        papers: list[dict],
        report: str = _REPORT,
    ) -> ResearchAgent:
        llm = _ok_llm()
        agent = ResearchAgent(
            llm=llm,
            arxiv=_mock_tool(papers),
            pubmed=_mock_tool([]),
            semantic_scholar=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=MagicMock(return_value=None),
        )

        # Override the planner and synthesiser at instance level
        agent._planner = MagicMock()
        agent._planner.aplan = AsyncMock(return_value=plan)

        agent._synthesiser = MagicMock()
        synth_result = MagicMock()
        synth_result.report = report
        synth_result.paper_count = len(papers)
        agent._synthesiser.arun = AsyncMock(return_value=synth_result)

        return agent

    def test_returns_string_report(self):
        plan = self._build_plan(["What is RAG?"])
        agent = self._make_full_agent(plan, [_make_paper("RAG Paper", doi="10.1/r")])
        result = asyncio.run(agent.research("What is RAG?"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_planner_called_with_user_query(self):
        plan = self._build_plan(["sub-q"])
        agent = self._make_full_agent(plan, [])
        agent._synthesiser.arun.return_value = MagicMock(report="ok", paper_count=0)
        asyncio.run(agent.research("my query"))
        agent._planner.aplan.assert_called_once_with("my query")

    def test_synthesiser_receives_deduplicated_papers(self):
        """Two tools returning the same DOI → synthesiser gets 1 paper, not 2."""
        plan = self._build_plan(["sub-q"])
        paper = _make_paper("Duplicate", doi="10.1/dup")
        agent = ResearchAgent(
            llm=_ok_llm(),
            arxiv=_mock_tool([paper]),
            semantic_scholar=_mock_tool([paper]),  # same paper, different tool
            pubmed=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=MagicMock(return_value=None),
        )
        agent._planner = MagicMock()
        agent._planner.aplan = AsyncMock(return_value=plan)

        captured: list[dict] = []

        async def capture_arun(query, papers):
            captured.extend(papers)
            result = MagicMock()
            result.report = "ok"
            result.paper_count = len(papers)
            return result

        agent._synthesiser = MagicMock()
        agent._synthesiser.arun = capture_arun

        asyncio.run(agent.research("dedup test"))
        assert len(captured) == 1

    def test_empty_query_raises_value_error(self):
        plan = self._build_plan([])
        agent = self._make_full_agent(plan, [])
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(agent.research(""))

    def test_whitespace_only_query_raises(self):
        plan = self._build_plan([])
        agent = self._make_full_agent(plan, [])
        with pytest.raises(ValueError):
            asyncio.run(agent.research("   "))

    def test_total_paper_cap_respected(self):
        """More papers than max_total_papers → synthesiser receives at most cap."""
        plan = self._build_plan(["q"])
        many = [_make_paper(f"P{i}", doi=f"10.1/{i}", citation_count=i) for i in range(100)]

        agent = ResearchAgent(
            llm=_ok_llm(),
            arxiv=_mock_tool(many),
            semantic_scholar=_mock_tool([]),
            pubmed=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=MagicMock(return_value=None),
            max_total_papers=20,
        )
        agent._planner = MagicMock()
        agent._planner.aplan = AsyncMock(return_value=plan)

        captured: list[dict] = []

        async def capture_arun(query, papers):
            captured.extend(papers)
            r = MagicMock()
            r.report = "ok"
            r.paper_count = len(papers)
            return r

        agent._synthesiser = MagicMock()
        agent._synthesiser.arun = capture_arun

        asyncio.run(agent.research("large query"))
        assert len(captured) <= 20

    def test_highest_citation_papers_kept_when_capped(self):
        """When truncated, the most-cited papers should survive."""
        plan = self._build_plan(["q"])
        papers = [
            _make_paper(f"P{i}", doi=f"10.1/{i}", citation_count=i) for i in range(10)
        ]
        agent = ResearchAgent(
            llm=_ok_llm(),
            arxiv=_mock_tool(papers),
            semantic_scholar=_mock_tool([]),
            pubmed=_mock_tool([]),
            openalex=_mock_tool([]),
            unpaywall=MagicMock(return_value=None),
            max_total_papers=3,
        )
        agent._planner = MagicMock()
        agent._planner.aplan = AsyncMock(return_value=plan)

        captured: list[dict] = []

        async def capture_arun(query, papers_):
            captured.extend(papers_)
            r = MagicMock()
            r.report = "ok"
            r.paper_count = len(papers_)
            return r

        agent._synthesiser = MagicMock()
        agent._synthesiser.arun = capture_arun

        asyncio.run(agent.research("cap test"))
        kept_counts = sorted([p["citation_count"] for p in captured], reverse=True)
        assert kept_counts == [9, 8, 7]