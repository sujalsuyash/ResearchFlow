"""
tests/test_query_planner.py

Unit test suite for chains/query_planner.py.
Zero real LLM calls — the LLM is replaced with a RunnableLambda that
returns a predetermined ResearchPlan so the real LCEL chain executes
but the network never moves.

Mock strategy
─────────────
We use RunnableLambda (not MagicMock) as the structured-output stand-in
because the LCEL pipe operator (|) requires both operands to be proper
Runnables. A MagicMock is not a Runnable and would cause a runtime error
inside the chain.

    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: plan)
    chain = PLANNER_PROMPT | mock_llm.with_structured_output(ResearchPlan)
    # chain.invoke(...) now returns `plan` without any network call.
"""

import os
import sys
from unittest.mock import MagicMock, call

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

# Project root on sys.path so "from chains.query_planner import ..." works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chains.query_planner import (
    PLANNER_PROMPT,
    ResearchPlan,
    SearchStep,
    SearchToolName,
    generate_plan,
)


# ---------------------------------------------------------------------------
# Shared fixtures & factories
# ---------------------------------------------------------------------------

def _make_step(
    tool_name: str = "semantic_scholar_search",
    search_query: str = "transformer attention mechanism NLP",
    rationale: str = "Broad search across all domains for ML foundations.",
) -> SearchStep:
    return SearchStep(
        tool_name=tool_name,
        search_query=search_query,
        rationale=rationale,
    )


def _make_plan(*steps: SearchStep) -> ResearchPlan:
    return ResearchPlan(steps=list(steps))


def _mock_llm(plan: ResearchPlan) -> MagicMock:
    """
    Return a MagicMock LLM whose .with_structured_output() gives a
    RunnableLambda that always returns `plan`.
    The lambda ignores its input (the formatted prompt messages) — we
    test the prompt separately.
    """
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: plan)
    return mock_llm


# ============================================================
# 1. PYDANTIC MODELS — schema validation
# ============================================================

class TestSearchStepModel:

    def test_valid_step_constructs_correctly(self):
        step = _make_step()
        assert step.tool_name    == "semantic_scholar_search"
        assert step.search_query == "transformer attention mechanism NLP"
        assert isinstance(step.rationale, str)

    @pytest.mark.parametrize("tool_name", [
        "semantic_scholar_search",
        "arxiv_search",
        "pubmed_search",
        "openalex_search",
    ])
    def test_all_valid_tool_names_accepted(self, tool_name):
        step = _make_step(tool_name=tool_name)
        assert step.tool_name == tool_name

    def test_invalid_tool_name_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchStep(
                tool_name="unpaywall_lookup",   # valid tool but not a planner tool
                search_query="test",
                rationale="test",
            )
        errors = exc_info.value.errors()
        assert any(e["type"] == "literal_error" for e in errors)

    def test_unknown_tool_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SearchStep(
                tool_name="google_scholar",
                search_query="test",
                rationale="test",
            )

    def test_missing_search_query_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SearchStep(tool_name="arxiv_search", rationale="test")  # type: ignore[call-arg]

    def test_missing_rationale_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SearchStep(tool_name="arxiv_search", search_query="test")  # type: ignore[call-arg]


class TestResearchPlanModel:

    def test_valid_plan_with_single_step(self):
        plan = _make_plan(_make_step())
        assert len(plan.steps) == 1
        assert isinstance(plan.steps[0], SearchStep)

    def test_valid_plan_with_multiple_steps(self):
        plan = _make_plan(
            _make_step("arxiv_search",   "transformer NLP"),
            _make_step("pubmed_search",  "transformer protein folding"),
            _make_step("openalex_search","cross-domain transformer applications"),
        )
        assert len(plan.steps) == 3
        assert all(isinstance(s, SearchStep) for s in plan.steps)

    def test_plan_preserves_step_order(self):
        s1 = _make_step("arxiv_search",  "query one")
        s2 = _make_step("pubmed_search", "query two")
        plan = _make_plan(s1, s2)
        assert plan.steps[0].search_query == "query one"
        assert plan.steps[1].search_query == "query two"

    def test_missing_steps_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ResearchPlan()   # type: ignore[call-arg]

    def test_plan_with_empty_steps_list(self):
        # Pydantic allows an empty list — domain logic can enforce minimums
        plan = ResearchPlan(steps=[])
        assert plan.steps == []


# ============================================================
# 2. generate_plan — LCEL chain execution
# ============================================================

class TestGeneratePlan:

    def test_returns_research_plan_instance(self):
        expected = _make_plan(_make_step())
        result   = generate_plan("How do transformers work?", _mock_llm(expected))
        assert isinstance(result, ResearchPlan)

    def test_returned_plan_equals_mock_output(self):
        expected = _make_plan(
            _make_step("arxiv_search", "large language model fine-tuning RLHF"),
            _make_step("semantic_scholar_search", "RLHF alignment safety survey"),
        )
        result = generate_plan("Explain RLHF for LLM alignment", _mock_llm(expected))
        assert result == expected

    def test_steps_are_searchstep_instances(self):
        expected = _make_plan(
            _make_step("pubmed_search", "glioblastoma immunotherapy checkpoint"),
            _make_step("openalex_search", "blood brain barrier drug delivery"),
        )
        result = generate_plan("Latest glioblastoma treatments?", _mock_llm(expected))
        assert all(isinstance(s, SearchStep) for s in result.steps)

    def test_with_structured_output_called_with_research_plan(self):
        expected = _make_plan(_make_step())
        mock_llm = _mock_llm(expected)
        generate_plan("test query", mock_llm)
        mock_llm.with_structured_output.assert_called_once_with(ResearchPlan)

    def test_with_structured_output_called_exactly_once(self):
        expected = _make_plan(_make_step())
        mock_llm = _mock_llm(expected)
        generate_plan("test query", mock_llm)
        assert mock_llm.with_structured_output.call_count == 1

    def test_multi_step_plan_length_preserved(self):
        steps = [
            _make_step("arxiv_search",            "diffusion model image synthesis"),
            _make_step("semantic_scholar_search",  "score matching denoising survey"),
            _make_step("openalex_search",          "generative model evaluation FID"),
        ]
        expected = ResearchPlan(steps=steps)
        result   = generate_plan("How do diffusion models work?", _mock_llm(expected))
        assert len(result.steps) == 3

    def test_tool_names_in_result_are_valid(self):
        expected = _make_plan(
            _make_step("arxiv_search",   "RAG retrieval augmented generation"),
            _make_step("pubmed_search",  "retrieval augmented clinical NLP"),
        )
        result = generate_plan("RAG in medicine", _mock_llm(expected))
        valid  = {"semantic_scholar_search", "arxiv_search", "pubmed_search", "openalex_search"}
        for step in result.steps:
            assert step.tool_name in valid

    def test_search_queries_are_non_empty_strings(self):
        expected = _make_plan(
            _make_step("semantic_scholar_search", "quantum computing error correction"),
            _make_step("arxiv_search",            "surface code topological qubits"),
        )
        result = generate_plan("Quantum error correction", _mock_llm(expected))
        for step in result.steps:
            assert isinstance(step.search_query, str)
            assert len(step.search_query.strip()) > 0

    def test_rationale_is_non_empty_string(self):
        expected = _make_plan(_make_step())
        result   = generate_plan("test", _mock_llm(expected))
        for step in result.steps:
            assert isinstance(step.rationale, str)
            assert len(step.rationale.strip()) > 0

    def test_different_queries_use_same_mock_correctly(self):
        """generate_plan should work identically regardless of the query string."""
        plan = _make_plan(_make_step("openalex_search", "climate change sea level rise"))
        for query in ["sea level", "climate tipping points", "arctic ice melt"]:
            result = generate_plan(query, _mock_llm(plan))
            assert isinstance(result, ResearchPlan)
            assert len(result.steps) == 1


# ============================================================
# 3. PLANNER_PROMPT — template content & structure
# ============================================================

class TestPlannerPrompt:

    def test_prompt_has_two_messages(self):
        # system + human
        assert len(PLANNER_PROMPT.messages) == 2

    def test_prompt_accepts_user_query_variable(self):
        """Formatting must not raise — proves {user_query} placeholder exists."""
        messages = PLANNER_PROMPT.format_messages(user_query="test research question")
        assert len(messages) == 2

    def test_user_query_is_injected_into_human_message(self):
        query    = "What causes Alzheimer's disease?"
        messages = PLANNER_PROMPT.format_messages(user_query=query)
        # Human message is the second message
        human_content = messages[1].content
        assert query in human_content

    def test_system_prompt_mentions_all_four_tools(self):
        messages = PLANNER_PROMPT.format_messages(user_query="test")
        system_content = messages[0].content
        for tool in [
            "semantic_scholar_search",
            "arxiv_search",
            "pubmed_search",
            "openalex_search",
        ]:
            assert tool in system_content, f"Tool '{tool}' missing from system prompt"

    def test_system_prompt_contains_domain_guidance(self):
        messages       = PLANNER_PROMPT.format_messages(user_query="test")
        system_content = messages[0].content
        # Each major domain should be mentioned so the LLM routes correctly
        for domain_hint in ["biomedical", "preprint", "cross"]:
            assert any(
                domain_hint.lower() in system_content.lower()
                for domain_hint in ["biomedical", "preprint", "cross-domain", "medical"]
            ), "System prompt missing domain routing guidance"

    def test_system_prompt_contains_step_count_guidance(self):
        messages       = PLANNER_PROMPT.format_messages(user_query="test")
        system_content = messages[0].content
        # The prompt should tell the LLM how many steps to produce
        assert "2" in system_content or "two" in system_content.lower()
        assert "5" in system_content or "five" in system_content.lower()

    def test_prompt_input_variables(self):
        assert "user_query" in PLANNER_PROMPT.input_variables

    def test_formatting_with_complex_query(self):
        """Multi-line, unicode, special characters must not break formatting."""
        complex_query = (
            "What are the effects of CRISPR-Cas9 on off-target mutations "
            "in haematopoietic stem cells, and how does this compare to "
            "base-editing approaches? (focus: 2020–2024)"
        )
        messages = PLANNER_PROMPT.format_messages(user_query=complex_query)
        assert complex_query in messages[1].content


# ============================================================
# 4. INTEGRATION — chain wiring
# ============================================================

class TestChainWiring:

    def test_chain_is_invokable_with_user_query_key(self):
        """The LCEL chain must accept {'user_query': ...} as input."""
        plan     = _make_plan(_make_step())
        mock_llm = _mock_llm(plan)
        # Build the chain exactly as generate_plan does
        chain    = PLANNER_PROMPT | mock_llm.with_structured_output(ResearchPlan)
        result   = chain.invoke({"user_query": "How does BERT work?"})
        assert isinstance(result, ResearchPlan)

    def test_chain_output_is_not_string(self):
        """Structured output must return a Pydantic model, never raw text."""
        plan   = _make_plan(_make_step())
        result = generate_plan("test", _mock_llm(plan))
        assert not isinstance(result, str)

    def test_chain_output_is_not_dict(self):
        """Must be a ResearchPlan, not a plain dict even if values are correct."""
        plan   = _make_plan(_make_step())
        result = generate_plan("test", _mock_llm(plan))
        assert not isinstance(result, dict)
        assert isinstance(result, ResearchPlan)

    def test_two_sequential_calls_are_independent(self):
        """Each call to generate_plan builds a fresh chain invocation."""
        plan_a = _make_plan(_make_step("arxiv_search",  "attention NLP"))
        plan_b = _make_plan(_make_step("pubmed_search", "attention ADHD"))

        result_a = generate_plan("transformers in NLP",     _mock_llm(plan_a))
        result_b = generate_plan("attention deficit disorder", _mock_llm(plan_b))

        assert result_a.steps[0].tool_name == "arxiv_search"
        assert result_b.steps[0].tool_name == "pubmed_search"
        assert result_a != result_b