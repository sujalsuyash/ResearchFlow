"""
core/pipeline.py
────────────────
ResearchFlow pipeline orchestrator.

This module is the single public surface that application code (main.py,
notebooks, API servers) should import.  It wraps ``ResearchAgent`` with:

  • A ``PipelineResult`` dataclass carrying the report, the paper list, per-stage
    wall-clock timings, and run metadata — so callers never need to scrape the
    Markdown string for this information.

  • Optional ``ProgressCallback`` hooks that fire before and after each of the
    five stages.  Pass a custom callback to feed a CLI spinner, a WebSocket
    stream, or a test harness.

  • ``ResearchPipeline.run()`` (sync) and ``ResearchPipeline.arun()`` (async)
    so the same object works in both CLI scripts and async web frameworks.

  • ``build_pipeline_from_env()`` — a one-call factory that reads all env vars
    and wires every component together.

Typical usage
─────────────
    from core.pipeline import build_pipeline_from_env

    pipeline = build_pipeline_from_env()
    result   = pipeline.run("What are the latest advances in RAG for LLMs?")

    print(result.report)
    print(f"Papers retrieved: {result.paper_count}")
    print(f"Total wall time:  {result.total_seconds:.1f}s")
    result.save("report.md")

Environment variables (same as ResearchAgent / build_from_env)
──────────────────────────────────────────────────────────────
  GROQ_KEY_1, GROQ_KEY_2, GROQ_KEY_3   — Groq API keys (at least one required)
  GOOGLE_API_KEY                         — Gemini fallback key (optional)
  SEMANTIC_SCHOLAR_API_KEY               — Raises SS rate-limit from 1→10 req/s
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.filter import RelevanceFilter  # ← single import, top-level only

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Stage definitions
# ──────────────────────────────────────────────────────────────────────────────

class Stage:
    """String-enum of pipeline stage identifiers."""
    PLAN       = "plan"        # QueryPlanner  — decompose query into steps
    EXECUTE    = "execute"     # Research tools — fan-out API calls
    DEDUP      = "dedup"       # Global deduplication + citation-count sort
    ENRICH     = "enrich"      # Unpaywall — PDF URL injection
    FILTER     = "filter"      # RelevanceFilter — drop irrelevant papers
    SYNTHESISE = "synthesise"  # SynthesisChain — produce final Markdown report


_STAGE_LABELS: dict[str, str] = {
    Stage.PLAN:       "Planning research strategy",
    Stage.EXECUTE:    "Querying academic databases",
    Stage.DEDUP:      "Deduplicating & ranking papers",
    Stage.ENRICH:     "Enriching with open-access PDFs",
    Stage.FILTER:     "Filtering irrelevant papers",
    Stage.SYNTHESISE: "Synthesising report",
}

_STAGE_ORDER: list[str] = [
    Stage.PLAN,
    Stage.EXECUTE,
    Stage.DEDUP,
    Stage.ENRICH,
    Stage.FILTER,
    Stage.SYNTHESISE,
]


# ──────────────────────────────────────────────────────────────────────────────
# Progress callback protocol
# ──────────────────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str, str, Optional[dict[str, Any]]], None]
"""
Signature: ``callback(stage_id, message, metadata)``

Called twice per stage:

  • **Before** the stage starts:
    ``stage_id`` is one of the ``Stage.*`` constants;
    ``message``  is a human-readable description;
    ``metadata`` is ``None``.

  • **After** the stage finishes:
    ``stage_id`` is the same constant;
    ``message``  is ``"done"``;
    ``metadata`` is a dict with at minimum ``{"elapsed_s": float}``.
"""


def _noop_progress(stage_id: str, message: str, metadata: Optional[dict] = None) -> None:
    """Default no-op callback — replace with a spinner, logger, etc."""


def _logging_progress(stage_id: str, message: str, metadata: Optional[dict] = None) -> None:
    """
    A ready-made ``ProgressCallback`` that logs every event at INFO level.
    Pass this to ``ResearchPipeline`` when you want automatic stage logging.
    """
    if message == "done" and metadata:
        logger.info(
            "  ✓ [%s] done in %.1fs%s",
            stage_id,
            metadata.get("elapsed_s", 0),
            f" — {metadata.get('detail', '')}" if metadata.get("detail") else "",
        )
    else:
        logger.info("  ▶ [%s] %s …", stage_id, message)


# ──────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    All outputs and telemetry produced by a single pipeline run.

    Attributes
    ──────────
    report : str
        The final Markdown research report with inline citations and bibliography.
    papers : list[dict]
        The filtered, enriched list of papers passed to the Synthesiser.
    query : str
        The original user query.
    stage_timings : dict[str, float]
        Wall-clock seconds per stage, keyed by ``Stage.*`` constants.
    total_seconds : float
        Sum of all stage timings.
    paper_count : int
        ``len(self.papers)`` — convenience property.
    plan_steps : int
        Number of sub-questions the QueryPlanner generated.
    raw_paper_count : int
        Papers collected before deduplication.
    """

    report:          str
    papers:          list[dict[str, Any]]
    query:           str
    stage_timings:   dict[str, float] = field(default_factory=dict)
    total_seconds:   float            = 0.0
    plan_steps:      int              = 0
    raw_paper_count: int              = 0

    @property
    def paper_count(self) -> int:
        return len(self.papers)

    def timing_table(self) -> str:
        """Return a human-readable per-stage timing summary."""
        lines = ["Stage timings:"]
        for stage in _STAGE_ORDER:
            secs  = self.stage_timings.get(stage, 0.0)
            label = _STAGE_LABELS.get(stage, stage)
            lines.append(f"  {label:<40} {secs:>6.1f}s")
        lines.append(f"  {'Total':<40} {self.total_seconds:>6.1f}s")
        return "\n".join(lines)

    def save(self, path: str | Path, *, include_meta: bool = True) -> Path:
        """
        Write the report to *path* (creates parent directories as needed).
        Prepends YAML front-matter when include_meta is True.
        Returns the resolved Path.
        """
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        content = self.report
        if include_meta:
            meta_block = (
                "---\n"
                f'query: "{self.query}"\n'
                f"papers: {self.paper_count}\n"
                f"plan_steps: {self.plan_steps}\n"
                f"total_seconds: {self.total_seconds:.1f}\n"
                "---\n\n"
            )
            content = meta_block + content

        out.write_text(content, encoding="utf-8")
        logger.info("Report saved → %s", out)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# ResearchPipeline
# ──────────────────────────────────────────────────────────────────────────────

class ResearchPipeline:
    """
    High-level orchestrator for the ResearchFlow six-stage pipeline.

    Stages: Plan → Execute → Dedup → Enrich → Filter → Synthesise

    Parameters
    ──────────
    agent : ResearchAgent
        A fully-configured ``ResearchAgent`` instance.
    progress : ProgressCallback, optional
        Called before and after each stage. Defaults to a no-op.
    """

    def __init__(
        self,
        agent: Any,
        *,
        progress: ProgressCallback = _noop_progress,
    ) -> None:
        self._agent    = agent
        self._progress = progress

    # ── Public sync interface ─────────────────────────────────────────────────

    def run(self, query: str) -> PipelineResult:
        """Synchronous entry point — blocks until the pipeline completes."""
        return asyncio.run(self.arun(query))

    # ── Public async interface ────────────────────────────────────────────────

    async def arun(self, query: str) -> PipelineResult:
        """Async entry point — prefer this inside FastAPI / Jupyter / etc."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        logger.info("ResearchPipeline: starting — %r", query)

        stage_timings: dict[str, float] = {}
        run_meta: dict[str, Any] = {
            "plan_steps":      0,
            "raw_paper_count": 0,
            "papers":          [],
        }

        pipeline_start = time.perf_counter()
        report = await self._run_pipeline(query, stage_timings, run_meta)
        total  = time.perf_counter() - pipeline_start

        logger.info(
            "ResearchPipeline: complete — %d papers, %.1fs total",
            len(run_meta["papers"]), total,
        )

        return PipelineResult(
            report          = report,
            papers          = run_meta["papers"],
            query           = query,
            stage_timings   = stage_timings,
            total_seconds   = total,
            plan_steps      = run_meta["plan_steps"],
            raw_paper_count = run_meta["raw_paper_count"],
        )

    # ── Internal instrumented pipeline ────────────────────────────────────────

    async def _run_pipeline(
        self,
        query:   str,
        timings: dict[str, float],
        meta:    dict[str, Any],
    ) -> str:
        """Execute all six stages with per-stage timing and progress callbacks."""

        agent = self._agent
        cb    = self._progress

        # ── Stage 1: Plan ──────────────────────────────────────────────────
        cb(Stage.PLAN, _STAGE_LABELS[Stage.PLAN])
        t0   = time.perf_counter()
        plan = await agent._aplan(query)
        timings[Stage.PLAN] = time.perf_counter() - t0
        meta["plan_steps"]  = len(plan.steps)
        cb(Stage.PLAN, "done", {
            "elapsed_s": timings[Stage.PLAN],
            "detail":    f"{len(plan.steps)} search step(s)",
        })

        # ── Stage 2: Execute ───────────────────────────────────────────────
        cb(Stage.EXECUTE, _STAGE_LABELS[Stage.EXECUTE])
        t0         = time.perf_counter()
        all_papers: list[dict[str, Any]] = []

        for idx, step in enumerate(plan.steps, start=1):
            label = (
                getattr(step, "search_query", None)
                or getattr(step, "sub_question", None)
                or f"step {idx}"
            )
            logger.info(
                "ResearchPipeline: executing step %d/%d — %r",
                idx, len(plan.steps), label,
            )
            step_papers = await agent._execute_step(step)
            all_papers.extend(step_papers)

        timings[Stage.EXECUTE]   = time.perf_counter() - t0
        meta["raw_paper_count"]  = len(all_papers)
        cb(Stage.EXECUTE, "done", {
            "elapsed_s": timings[Stage.EXECUTE],
            "detail":    f"{len(all_papers)} paper(s) collected (pre-dedup)",
        })

        # ── Stage 3: Dedup ─────────────────────────────────────────────────
        cb(Stage.DEDUP, _STAGE_LABELS[Stage.DEDUP])
        t0 = time.perf_counter()

        from agents.research_agent import deduplicate  # type: ignore[import]

        unique = deduplicate(all_papers)
        unique = sorted(
            unique, key=lambda p: p.get("citation_count") or 0, reverse=True
        )[: agent._max_total_papers]

        timings[Stage.DEDUP] = time.perf_counter() - t0
        cb(Stage.DEDUP, "done", {
            "elapsed_s": timings[Stage.DEDUP],
            "detail":    f"{len(unique)} unique paper(s) (from {len(all_papers)} raw)",
        })

        if not unique:
            logger.warning("ResearchPipeline: no papers found — returning early.")
            return (
                "### Research Summary\n\n"
                "No academic papers were found for your query.\n"
                "Try broadening your search terms or checking API credentials."
            )

        # ── Stage 4: Enrich ────────────────────────────────────────────────
        cb(Stage.ENRICH, _STAGE_LABELS[Stage.ENRICH])
        t0       = time.perf_counter()
        enriched = await agent._enrich_with_pdfs(unique)
        pdf_count = sum(1 for p in enriched if p.get("pdf_url"))
        timings[Stage.ENRICH] = time.perf_counter() - t0
        cb(Stage.ENRICH, "done", {
            "elapsed_s": timings[Stage.ENRICH],
            "detail":    f"{pdf_count} open-access PDF(s) found",
        })

        # ── Stage 5: Filter ────────────────────────────────────────────────
        cb(Stage.FILTER, _STAGE_LABELS[Stage.FILTER])
        t0 = time.perf_counter()

        relevance_filter = RelevanceFilter(
            max_papers          = agent._max_total_papers,
            semantic_threshold  = 0.30,
            lexical_min_overlap = 0.3,
        )
        sub_queries = [step.search_query for step in plan.steps]
        filter_result = relevance_filter.run(query=query, papers=enriched, sub_queries=sub_queries)
        filtered       = filter_result.papers
        meta["papers"] = filtered

        timings[Stage.FILTER] = time.perf_counter() - t0
        cb(Stage.FILTER, "done", {
            "elapsed_s": timings[Stage.FILTER],
            "detail": (
                f"{filter_result.kept} kept, "
                f"{filter_result.dropped} dropped "
                f"[q={filter_result.layer_stats.get('quality', 0)} "
                f"lex={filter_result.layer_stats.get('lexical', 0)} "
                f"sem={filter_result.layer_stats.get('semantic', 0)}]"
            ),
        })

        if not filtered:
            logger.warning("RelevanceFilter: all papers dropped — returning early.")
            return (
                "### Research Summary\n\n"
                "No papers passed the relevance filter for your query.\n"
                "Try using more specific terminology."
            )

        # ── Stage 6: Synthesise ────────────────────────────────────────────
        cb(Stage.SYNTHESISE, _STAGE_LABELS[Stage.SYNTHESISE])
        t0     = time.perf_counter()
        result = await agent._synthesiser.arun(query, filtered)
        timings[Stage.SYNTHESISE] = time.perf_counter() - t0
        cb(Stage.SYNTHESISE, "done", {
            "elapsed_s": timings[Stage.SYNTHESISE],
            "detail":    f"{result.paper_count} paper(s) cited in report",
        })

        return result.report


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline_from_env(
    *,
    max_papers_per_step: int   = 10,
    max_total_papers:    int   = 40,
    tool_timeout:        float = 120.0,
    progress:            ProgressCallback = _logging_progress,
) -> ResearchPipeline:
    """
    Build a production ``ResearchPipeline`` from environment variables.

    Required env vars: at least one of GROQ_KEY_1 / GROQ_KEY_2 / GROQ_KEY_3
    Optional env vars: GOOGLE_API_KEY, SEMANTIC_SCHOLAR_API_KEY

    Raises EnvironmentError if no Groq keys are configured.
    """
    from agents.research_agent import build_from_env as _build_agent  # type: ignore[import]

    agent = _build_agent(
        max_papers_per_step = max_papers_per_step,
        max_total_papers    = max_total_papers,
        tool_timeout        = tool_timeout,
    )
    return ResearchPipeline(agent=agent, progress=progress)