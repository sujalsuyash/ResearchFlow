"""
chains/query_planner.py

Stage 2 of the ResearchFlow pipeline: Query Planner Chain.

Takes a raw user research question and uses an LLM to decompose it into an
ordered list of concrete search steps, each mapped to the most appropriate
academic API tool.

Architecture — Sequential Constrained Generation
-------------------------------------------------
Previous design (broken):
    One LLM call generates all steps simultaneously. The LLM self-enforces
    diversity through prompt instructions, which it cannot do reliably — by
    the time it writes Step 3's search_query, it has no reliable memory of
    the exact token composition of Step 1's search_query. Result: angle
    labels differ but query strings converge on the same vocabulary.

Current design (fixed):
    One LLM call per step. After each call, a deterministic Jaccard
    similarity check compares the new query against every already-accepted
    query. If the overlap exceeds SIMILARITY_THRESHOLD, the step is
    rejected and the LLM is asked to try again with the rejected query
    shown explicitly as forbidden. Diversity is now a code guarantee,
    not an LLM promise.

    Loop per slot (up to MAX_STEPS):
        for attempt in range(MAX_RETRIES_PER_SLOT):
            step = LLM(user_query, accepted_steps_so_far)
            if jaccard(step.query, every accepted query) < threshold:
                accept → break
            else:
                retry with rejection feedback
        if slot has MIN_STEPS accepted → can stop early

Key constants
-------------
SIMILARITY_THRESHOLD   : float = 0.35   Jaccard ceiling (35 % shared vocab = reject)
MAX_STEPS              : int   = 4      Hard cap on plan length
MIN_STEPS              : int   = 2      Minimum before early-stop is allowed
MAX_RETRIES_PER_SLOT   : int   = 3      Retries before abandoning a slot

Usage
-----
    from langchain_openai import ChatOpenAI
    from chains.query_planner import generate_plan

    llm  = ChatOpenAI(model="gpt-4o", temperature=0)
    plan = generate_plan("What are the latest treatments for glioblastoma?", llm)

    for step in plan.steps:
        print(f"[{step.angle}] {step.tool_name} → {step.search_query}")
"""

import asyncio
import logging
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diversity-enforcement constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD  = 0.35   # Jaccard similarity ceiling — above this = too similar
MAX_STEPS             = 4      # hard cap on steps generated
MIN_STEPS             = 2      # minimum accepted steps before early-stop is allowed
MAX_RETRIES_PER_SLOT  = 3      # LLM retry attempts per step slot before giving up

# ---------------------------------------------------------------------------
# Valid tool names
# ---------------------------------------------------------------------------

SearchToolName = Literal[
    "semantic_scholar_search",
    "arxiv_search",
    "pubmed_search",
    "openalex_search",
]

# ---------------------------------------------------------------------------
# Output schema  (unchanged from previous version)
# ---------------------------------------------------------------------------

class SearchStep(BaseModel):
    """A single concrete search action within a research plan."""

    angle: str = Field(
        description=(
            "A short label (3-6 words) naming the distinct research dimension this step "
            "explores. Every step in the plan MUST have a completely different angle — "
            "no two steps may investigate the same facet of the topic. "
            "Good examples of orthogonal angles for a drug-discovery question: "
            "  'Clinical Trial Outcomes', 'Molecular Mechanism of Action', "
            "  'Blood-Brain Barrier Drug Delivery', 'Adverse Effects & Safety Profile', "
            "  'Comparative Efficacy vs Existing Drugs'. "
            "Good examples for an ML question: "
            "  'Architecture & Training Efficiency', 'Benchmark & Evaluation Methods', "
            "  'Real-World Deployment Challenges', 'Theoretical Foundations', "
            "  'Dataset Bias & Fairness'. "
            "STRICTLY FORBIDDEN: angles that are minor keyword rewrites of each other. "
            "'Recent Treatments', 'New Therapies', 'Latest Drugs' are all the SAME angle "
            "and must never appear together in one plan."
        )
    )
    tool_name: SearchToolName = Field(
        description=(
            "The exact tool to invoke. Must be one of: "
            "semantic_scholar_search, arxiv_search, pubmed_search, openalex_search."
        )
    )
    search_query: str = Field(
        description=(
            "The precise query string to pass to the tool, written in the vocabulary "
            "researchers in this sub-field use in paper titles and abstracts. "
            "The query must be tightly scoped to the `angle` above — "
            "avoid generic terms that would match papers from other steps' angles."
        )
    )
    rationale: str = Field(
        description=(
            "One or two sentences explaining (a) why this tool was chosen for this angle "
            "and (b) what specific gap in the literature this step fills that no other "
            "step in the plan already covers."
        )
    )


class ResearchPlan(BaseModel):
    """
    A structured plan decomposing a user's research question into ordered
    search steps across academic APIs.
    """

    steps: list[SearchStep] = Field(
        description=(
            "Ordered list of search steps. Produce between 2 and 5 steps. "
            "CRITICAL CONSTRAINT: every step must explore a completely orthogonal "
            "angle of the topic. If you look at the `angle` fields of all steps and "
            "any two could be described as 'the same thing with different wording', "
            "you have failed. Rewrite until every angle is genuinely distinct."
        )
    )


# ---------------------------------------------------------------------------
# Jaccard similarity — deterministic query diversity guard
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "its", "it", "this", "that", "not",
    "use", "using", "used", "based", "novel", "new", "study", "paper",
    "approach", "method", "methods", "analysis", "review", "research",
    "towards", "via", "among", "across", "about", "two", "three",
})


def _tokenise(text: str) -> frozenset[str]:
    """Lowercase, strip punctuation, remove stop-words. Min token length: 3."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def _jaccard(a: str, b: str) -> float:
    """
    Jaccard similarity between two query strings.
    Returns 0.0 for empty inputs; 1.0 for identical token sets.
    """
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_too_similar(candidate: SearchStep, accepted: list[SearchStep]) -> tuple[bool, float, str]:
    """
    Compare candidate.search_query against every accepted step's search_query.

    Returns
    -------
    (too_similar: bool, max_similarity: float, conflicting_query: str)
        too_similar       True if any pair exceeds SIMILARITY_THRESHOLD
        max_similarity    highest Jaccard score found
        conflicting_query the accepted query that triggered the rejection (or "")
    """
    max_sim = 0.0
    conflict = ""
    for step in accepted:
        sim = _jaccard(candidate.search_query, step.search_query)
        if sim > max_sim:
            max_sim = sim
            conflict = step.search_query
    return (max_sim >= SIMILARITY_THRESHOLD, max_sim, conflict)


# ---------------------------------------------------------------------------
# Tool catalogue (shared between both prompt templates)
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS = """\
You have access to four academic search tools. Choose the most appropriate
tool for this step based on the domain and recency of the research angle.

┌─────────────────────────────┬──────────────────────────────────────────────────────────────────┐
│ Tool name                   │ When to use it                                                   │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ semantic_scholar_search     │ Default choice for most queries. Indexes 200M+ papers across     │
│                             │ all academic disciplines. Use for ML, CS, social science,        │
│                             │ economics, multi-domain, or any topic without a clear home.      │
│                             │ Supports semantic (meaning-based) search, not just keywords.     │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ arxiv_search                │ Cutting-edge preprints in CS, ML, AI, NLP, physics, mathematics, │
│                             │ quantitative biology, and economics. Use when recency matters     │
│                             │ (papers from the last 1-2 years) or the topic is fast-moving.   │
│                             │ Supports arXiv category codes (e.g. cs.LG, cs.CL, stat.ML).     │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ pubmed_search               │ Biomedical and life sciences literature. 35M+ articles indexed   │
│                             │ by the National Library of Medicine. Use for clinical trials,    │
│                             │ drug research, genomics, epidemiology, public health, and any    │
│                             │ query with a medical or biological angle.                        │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ openalex_search             │ Broad, fully open index (250M+ works). Best for cross-domain     │
│                             │ literature reviews, citation network analysis, concept-level      │
│                             │ filtering, and topics that span multiple disciplines.             │
│                             │ Use as a complement when the other tools may miss niche fields.  │
└─────────────────────────────┴──────────────────────────────────────────────────────────────────┘

DOMAIN ROUTING RULES
────────────────────
• Medical/biological mechanism or trials  → pubmed_search
• Latest ML/AI/CS preprints               → arxiv_search
• Cross-disciplinary or general           → semantic_scholar_search or openalex_search
• Niche or hard-to-find fields            → openalex_search as a catch-all

IMPORTANT: pubmed_search uses PubMed/NCBI — it indexes biomedical literature only.
Do NOT route CS, ML, or engineering queries there — they will return 0 results."""

# ---------------------------------------------------------------------------
# Prompt template A — PLANNER_PROMPT
# Used for the very first step (no prior context).
# Kept for backward compatibility; STEP_PROMPT is used for steps 2+.
# ---------------------------------------------------------------------------

_FIRST_STEP_SYSTEM = f"""\
You are the Query Planner for ResearchFlow, an autonomous academic research agent.

Your job is to generate the FIRST search step for a research question. This step
will be followed by additional steps covering different research dimensions, so
choose the single most important angle to investigate first.

{_TOOL_DESCRIPTIONS}

PLANNING RULES
──────────────
1. Decompose by research dimension, not by keyword synonym.
   Valid dimensions include (but are not limited to):
   • Mechanism / Architecture   — how does it work internally?
   • Clinical / Empirical Evidence — what do trials, benchmarks, or RCTs show?
   • Comparative Effectiveness  — how does it compare to alternatives?
   • Safety, Risks & Limitations — what are the failure modes or contraindications?
   • Deployment / Implementation — how is it used in practice or at scale?
   • Epidemiology / Prevalence  — who is affected and at what scale?
   • Dataset Bias & Fairness    — what are the equity or evaluation concerns?
   • Computational / ML angle   — what algorithmic or modelling work exists?

2. Write precise, field-specific search_query strings.
   Use vocabulary researchers in that sub-field use in paper titles/abstracts.
   Prefer: "late fusion decision-level combination multimodal classification"
   Over:   "late fusion multimodal"

Respond ONLY with the structured step — no preamble, no explanation outside the schema."""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _FIRST_STEP_SYSTEM),
    ("human", "Research question: {user_query}"),
])

# ---------------------------------------------------------------------------
# Prompt template B — STEP_PROMPT
# Used for steps 2, 3, 4 ... with full context of already-accepted steps.
# The {already_covered} variable is a formatted block injected by _format_context().
# ---------------------------------------------------------------------------

_NEXT_STEP_SYSTEM = f"""\
You are the Query Planner for ResearchFlow, an autonomous academic research agent.

Your job is to generate ONE new search step for a research question.
The steps already planned are shown in the human message — your new step
MUST cover a completely different research dimension.

{_TOOL_DESCRIPTIONS}

DIVERSITY RULE  —  this is the only rule that matters for this call
───────────────────────────────────────────────────────────────────
Look at the search_query strings of the already-planned steps.
Your new search_query must share fewer than 35% of their vocabulary.

Concretely: if a word appears in any already-planned query, avoid it
in your new query UNLESS it is the irreducible technical anchor term
(e.g. "late fusion", "Parkinson's disease", "transformer") that cannot
be dropped without losing the topic entirely.

Dimensions already planned are off-limits. Choose from:
• Mechanism / Architecture        • Clinical / Empirical Evidence
• Comparative Effectiveness       • Safety, Risks & Limitations
• Deployment / Implementation     • Epidemiology / Prevalence
• Dataset & Benchmark Analysis    • Economic / Policy Impact
• Computational / ML angle        • Diagnostic Biomarkers

DOMAIN ROUTING REMINDER
───────────────────────
pubmed_search = biomedical literature ONLY. Never route CS/ML queries there.
arxiv_search  = CS/ML/physics preprints. Best for fast-moving technical angles.

Respond ONLY with the structured step — no preamble, no explanation outside the schema."""

STEP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _NEXT_STEP_SYSTEM),
    ("human", (
        "Research question: {user_query}\n\n"
        "Steps already planned (DO NOT repeat these angles or reuse their vocabulary):\n"
        "{already_covered}\n\n"
        "Generate the next step, covering a completely different dimension:"
    )),
])

# ---------------------------------------------------------------------------
# Rejection-feedback prompt — used when a step fails the Jaccard guard
# ---------------------------------------------------------------------------

_REJECTION_SYSTEM = f"""\
You are the Query Planner for ResearchFlow, an autonomous academic research agent.

Your previous search step was rejected because its search_query was too similar
to an already-accepted query. You must produce a replacement step that covers
a DIFFERENT research dimension with DIFFERENT vocabulary.

{_TOOL_DESCRIPTIONS}

Respond ONLY with the structured step — no preamble."""

REJECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _REJECTION_SYSTEM),
    ("human", (
        "Research question: {user_query}\n\n"
        "Steps already accepted:\n{already_covered}\n\n"
        "Your REJECTED step (too similar to an accepted query):\n"
        "  angle:        {rejected_angle}\n"
        "  search_query: {rejected_query}\n"
        "  similarity:   {similarity:.0%} overlap with: \"{conflicting_query}\"\n\n"
        "Generate a replacement step with a genuinely different angle and vocabulary:"
    )),
])

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_context(accepted: list[SearchStep]) -> str:
    """
    Render accepted steps as a readable block for injection into STEP_PROMPT
    and REJECTION_PROMPT. Shows angle, tool, and full search_query so the
    LLM can see exactly what vocabulary is already committed.
    """
    if not accepted:
        return "None yet — this is the first step."
    lines = []
    for i, s in enumerate(accepted, start=1):
        lines.append(
            f"Step {i} [angle: \"{s.angle}\"]\n"
            f"  tool:  {s.tool_name}\n"
            f"  query: \"{s.search_query}\""
        )
    return "\n\n".join(lines)


def _generate_one_step(
    user_query: str,
    accepted: list[SearchStep],
    llm: BaseChatModel,
    *,
    rejected: SearchStep | None = None,
    similarity: float = 0.0,
    conflicting_query: str = "",
) -> SearchStep:
    """
    Ask the LLM to produce a single SearchStep.

    If `rejected` is provided, uses REJECTION_PROMPT with explicit feedback
    about why the previous attempt failed. Otherwise uses PLANNER_PROMPT
    (first step) or STEP_PROMPT (subsequent steps).
    """
    context = _format_context(accepted)
    chain   = llm.with_structured_output(SearchStep)

    if rejected is not None:
        prompt  = REJECTION_PROMPT
        payload = {
            "user_query":        user_query,
            "already_covered":   context,
            "rejected_angle":    rejected.angle,
            "rejected_query":    rejected.search_query,
            "similarity":        similarity,
            "conflicting_query": conflicting_query,
        }
    elif not accepted:
        prompt  = PLANNER_PROMPT
        payload = {"user_query": user_query}
    else:
        prompt  = STEP_PROMPT
        payload = {"user_query": user_query, "already_covered": context}

    return (prompt | chain).invoke(payload)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plan(
    user_query:  str,
    llm:         BaseChatModel,
    *,
    max_steps:   int   = MAX_STEPS,
    min_steps:   int   = MIN_STEPS,
    max_retries: int   = MAX_RETRIES_PER_SLOT,
    threshold:   float = SIMILARITY_THRESHOLD,
) -> ResearchPlan:
    """
    Decompose `user_query` into a diverse ResearchPlan using sequential
    constrained generation.

    Each step is generated individually. After generation, a Jaccard
    similarity check compares the new query against all already-accepted
    queries. If the overlap exceeds `threshold`, the step is rejected and
    the LLM is asked to try again with explicit rejection feedback.
    This continues for up to `max_retries` attempts per slot.

    Parameters
    ----------
    user_query  : str            Raw research question from the user.
    llm         : BaseChatModel  Any LangChain chat model with structured output.
    max_steps   : int            Hard cap on plan length (default 4).
    min_steps   : int            Minimum steps before early-stop (default 2).
    max_retries : int            Retries per slot before giving up (default 3).
    threshold   : float          Jaccard ceiling; above = too similar (default 0.35).

    Returns
    -------
    ResearchPlan
        Validated Pydantic object whose .steps are guaranteed to be
        mutually distinct by the Jaccard guard.
    """
    logger.debug("Generating research plan (sequential) for query: %r", user_query)

    accepted: list[SearchStep] = []

    for slot in range(max_steps):
        rejected_step:      SearchStep | None = None
        rejected_sim:       float             = 0.0
        rejected_conflict:  str               = ""

        for attempt in range(max_retries):
            step = _generate_one_step(
                user_query, accepted, llm,
                rejected          = rejected_step,
                similarity        = rejected_sim,
                conflicting_query = rejected_conflict,
            )

            too_similar, sim, conflict = _is_too_similar(step, accepted)

            if not too_similar:
                # ── Accepted ────────────────────────────────────────────────
                accepted.append(step)
                logger.debug(
                    "  Slot %d accepted (attempt %d): [%s] tool=%s "
                    "query=%r  jaccard=%.2f",
                    slot + 1, attempt + 1, step.angle,
                    step.tool_name, step.search_query, sim,
                )
                break
            else:
                # ── Rejected — prepare feedback for next attempt ─────────
                logger.debug(
                    "  Slot %d rejected (attempt %d): jaccard=%.2f >= %.2f "
                    "with %r — retrying",
                    slot + 1, attempt + 1, sim, threshold, conflict,
                )
                rejected_step     = step
                rejected_sim      = sim
                rejected_conflict = conflict
        else:
            # All retries exhausted for this slot — log and move on
            logger.warning(
                "Slot %d: could not generate a distinct step after %d attempts "
                "— skipping this slot.",
                slot + 1, max_retries,
            )

        # Early-stop: if we have enough steps and the topic seems exhausted
        if len(accepted) >= min_steps and len(accepted) >= (slot + 1):
            # Check whether the last slot was actually accepted
            # If we failed to accept anything for this slot, the topic may be
            # narrow enough that we're done
            if len(accepted) < slot + 1:
                logger.debug(
                    "Early stop: %d step(s) accepted, slot %d produced nothing.",
                    len(accepted), slot + 1,
                )
                break

    if not accepted:
        raise RuntimeError(
            f"Query planner could not generate any distinct search steps "
            f"for query: {user_query!r}"
        )

    plan = ResearchPlan(steps=accepted)

    logger.info(
        "Research plan generated: %d step(s) for query=%r",
        len(plan.steps), user_query,
    )
    for i, step in enumerate(plan.steps, start=1):
        logger.info(
            "  Step %d [%s]: %s → %r",
            i, step.angle, step.tool_name, step.search_query,
        )

    return plan


# ---------------------------------------------------------------------------
# QueryPlanner class wrapper (unchanged — used by ResearchAgent)
# ---------------------------------------------------------------------------

class QueryPlanner:
    """Class wrapper that the ResearchAgent uses to orchestrate planning."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def plan(self, query: str) -> ResearchPlan:
        return generate_plan(query, self.llm)

    async def aplan(self, query: str) -> ResearchPlan:
        return await asyncio.to_thread(self.plan, query)


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    from langchain_groq import ChatGroq

    logging.basicConfig(level=logging.DEBUG)
    print("Booting up Query Planner (Sequential Constrained Generation)...")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    test_questions = [
        "Late Fusion on Parkinson's Disease on multimodal dataset",
        "What are the most effective treatments for Alzheimer's disease "
        "discovered in the last two years?",
    ]

    for question in test_questions:
        print(f"\n{'='*65}")
        print(f"Question: {question}")
        print("=" * 65)

        plan = generate_plan(question, llm)

        for i, step in enumerate(plan.steps, 1):
            print(f"\nStep {i}: [{step.angle}]")
            print(f"  Tool:      {step.tool_name}")
            print(f"  Query:     {step.search_query}")
            print(f"  Rationale: {step.rationale}")