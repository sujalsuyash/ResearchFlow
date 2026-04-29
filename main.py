"""
main.py
───────
ResearchFlow — CLI entry point.

Usage
─────
  # Basic run (prints Markdown to stdout)
  python main.py "What are the latest advances in RAG for LLMs?"

  # Save report to a file
  python main.py "CRISPR off-target effects in clinical trials" --output report.md

  # Tune retrieval
  python main.py "Alzheimer's treatments 2024" --max-papers 60 --papers-per-step 15

  # Verbose mode (DEBUG logging)
  python main.py "quantum error correction" --verbose

  # Silent mode (no stage progress, only final report)
  python main.py "solar cell efficiency records" --quiet

  # Non-interactive: skip the confirmation prompt
  python main.py "mRNA vaccine delivery mechanisms" --yes

Environment variables
─────────────────────
  GROQ_KEY_1, GROQ_KEY_2, GROQ_KEY_3   At least one required.
  GOOGLE_API_KEY                         Optional Gemini fallback.
  SEMANTIC_SCHOLAR_API_KEY               Optional; raises SS rate-limit to 10 req/s.

Exit codes
──────────
  0  Success.
  1  Bad arguments / missing env vars / pipeline error.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

from core.cache import get_redis_client
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup (before any project imports so early errors are visible)
# ──────────────────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(levelname)-8s %(name)s — %(message)s"


def _configure_logging(verbose: bool, quiet: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=_LOG_FORMAT)
    elif quiet:
        logging.basicConfig(level=logging.ERROR, format=_LOG_FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
        # Suppress noisy third-party loggers unless verbose
        for noisy in (
            "httpx", "httpcore", "urllib3", "openai", "anthropic",
            "groq", "langchain", "langchain_core", "langchain_community",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("researchflow.main")


# ──────────────────────────────────────────────────────────────────────────────
# ANSI helpers (gracefully disabled on non-TTY / Windows without colorama)
# ──────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and os.name != "nt"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str)  -> str: return _c("1",    text)
def _dim(text: str)   -> str: return _c("2",    text)
def _green(text: str) -> str: return _c("32",   text)
def _cyan(text: str)  -> str: return _c("36",   text)
def _yellow(text: str)-> str: return _c("33",   text)
def _red(text: str)   -> str: return _c("31",   text)


# ──────────────────────────────────────────────────────────────────────────────
# Progress callback
# ──────────────────────────────────────────────────────────────────────────────

_STAGE_ICONS: dict[str, str] = {
    "plan":       "🗺 ",
    "execute":    "🔍",
    "dedup":      "🔗",
    "enrich":     "📄",
    "synthesise": "✍️ ",
}


def _make_progress_callback(quiet: bool):
    """
    Return a ``ProgressCallback`` that prints stage progress to stderr.

    Uses stderr so that piping stdout to a file still captures only the final
    Markdown report, not the spinner/status lines.
    """
    if quiet:
        from core.pipeline import _noop_progress  # type: ignore[import]
        return _noop_progress

    _stage_start: dict[str, float] = {}

    def _callback(stage_id: str, message: str, metadata: Optional[dict] = None) -> None:
        icon = _STAGE_ICONS.get(stage_id, "•")
        if message == "done":
            elapsed = (metadata or {}).get("elapsed_s", 0.0)
            detail  = (metadata or {}).get("detail", "")
            suffix  = f"  {_dim(detail)}" if detail else ""
            print(
                f"  {icon} {_green('done')}  {_dim(f'{elapsed:.1f}s')}{suffix}",
                file=sys.stderr,
            )
        else:
            _stage_start[stage_id] = time.perf_counter()
            label = message.rstrip("…").rstrip()
            print(f"\n  {icon} {_cyan(label)} …", file=sys.stderr)

    return _callback


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ResearchFlow — autonomous academic research agent.

            Decomposes your question into targeted sub-queries, fans out across
            Semantic Scholar, arXiv, PubMed, and OpenAlex, deduplicates the
            results, enriches them with open-access PDFs via Unpaywall, and
            synthesises a cited Markdown report.
        """),
        epilog=textwrap.dedent("""\
            Environment variables:
              GROQ_KEY_1 / _2 / _3     Groq API keys (at least one required)
              GOOGLE_API_KEY            Gemini fallback (optional)
              SEMANTIC_SCHOLAR_API_KEY  Raises Semantic Scholar rate limit (optional)

            Examples:
              python main.py "What are the latest advances in RAG for LLMs?"
              python main.py "CRISPR off-target effects" --output report.md
              python main.py "Alzheimer treatment 2024" --max-papers 60 --verbose
        """),
    )

    parser.add_argument(
        "query",
        nargs="?",
        metavar="QUERY",
        help="Research question or topic to investigate.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help=(
            "Save the Markdown report to FILE instead of (or in addition to) "
            "printing it.  Creates parent directories automatically."
        ),
    )
    output_group.add_argument(
        "--no-print",
        action="store_true",
        default=False,
        help="Suppress printing the report to stdout (useful when --output is set).",
    )
    output_group.add_argument(
        "--no-meta",
        action="store_true",
        default=False,
        help="Omit YAML front-matter from the saved file.",
    )

    # ── Retrieval tuning ──────────────────────────────────────────────────────
    retrieval_group = parser.add_argument_group("retrieval")
    retrieval_group.add_argument(
        "--max-papers", "-n",
        type=int,
        default=40,
        metavar="N",
        help="Maximum papers to pass to the synthesiser (default: 40).",
    )
    retrieval_group.add_argument(
        "--papers-per-step",
        type=int,
        default=10,
        metavar="N",
        help="Maximum papers returned per individual tool call (default: 10).",
    )
    retrieval_group.add_argument(
        "--tool-timeout",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help=(
            "Per-tool timeout in seconds (default: 120). "
            "Must exceed the tool's full retry/backoff sequence."
        ),
    )

    # ── Verbosity ─────────────────────────────────────────────────────────────
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    verbosity_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress all progress output (only the final report is printed).",
    )

    # ── Interaction ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip the confirmation prompt before running.",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Environment validation
# ──────────────────────────────────────────────────────────────────────────────

def _check_env() -> list[str]:
    """
    Return a list of human-readable warnings about the current environment.
    Fatal missing keys cause an ``EnvironmentError``; optional keys emit warnings.
    """
    warnings: list[str] = []

    groq_keys = [
        os.getenv(f"GROQ_KEY_{i}") for i in range(1, 4) if os.getenv(f"GROQ_KEY_{i}")
    ]
    if not groq_keys:
        raise EnvironmentError(
            "No Groq API keys found.\n"
            "Set at least one of: GROQ_KEY_1 / GROQ_KEY_2 / GROQ_KEY_3\n"
            "Get a free key at https://console.groq.com"
        )

    if not os.getenv("GOOGLE_API_KEY"):
        warnings.append(
            "GOOGLE_API_KEY not set — Gemini fallback disabled. "
            "If all Groq keys hit rate limits, the run will fail."
        )
    if not os.getenv("S2_API_KEY"):
        warnings.append(
            "SEMANTIC_SCHOLAR_API_KEY not set — using unauthenticated access "
            "(1 req/s). Set the key for 10 req/s throughput."
        )

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Confirmation prompt
# ──────────────────────────────────────────────────────────────────────────────

def _confirm(query: str, args: argparse.Namespace) -> bool:
    """Print a run summary and ask the user to confirm (unless ``--yes``)."""
    if args.yes or not sys.stdin.isatty():
        return True

    print(file=sys.stderr)
    print(_bold("  ResearchFlow"), file=sys.stderr)
    print(f"  Query      : {_cyan(query)}", file=sys.stderr)
    print(f"  Max papers : {args.max_papers}", file=sys.stderr)
    if args.output:
        print(f"  Output     : {args.output}", file=sys.stderr)
    print(file=sys.stderr)

    try:
        answer = input("  Proceed? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return False

    return answer in ("", "y", "yes")


# ──────────────────────────────────────────────────────────────────────────────
# Core async runner
# ──────────────────────────────────────────────────────────────────────────────

async def _arun(query: str, args: argparse.Namespace) -> int:
    """
    Build the pipeline and run the query.  Returns an exit code (0 = success).
    """
    from core.pipeline import build_pipeline_from_env  # type: ignore[import]

    progress_cb = _make_progress_callback(quiet=args.quiet)

    if not args.quiet:
        print(
            f"\n{_bold('ResearchFlow')} — searching for: {_cyan(repr(query))}",
            file=sys.stderr,
        )

    try:
        pipeline = build_pipeline_from_env(
            max_papers_per_step=args.papers_per_step,
            max_total_papers=args.max_papers,
            tool_timeout=args.tool_timeout,
            progress=progress_cb,
        )
    except EnvironmentError as exc:
        print(_red(f"\nConfiguration error: {exc}"), file=sys.stderr)
        return 1

    try:
        result = await pipeline.arun(query)
    except KeyboardInterrupt:
        print(_yellow("\nInterrupted by user."), file=sys.stderr)
        return 1
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        print(_red(f"\nPipeline error: {exc}"), file=sys.stderr)
        return 1

    # ── Print timing summary ──────────────────────────────────────────────────
    if not args.quiet:
        print(file=sys.stderr)
        print(_dim(result.timing_table()), file=sys.stderr)
        print(
            _green(
                f"\n  ✓ Report ready — "
                f"{result.paper_count} papers, "
                f"{result.total_seconds:.1f}s total"
            ),
            file=sys.stderr,
        )
        print(file=sys.stderr)

    # ── Write to file if requested ────────────────────────────────────────────
    if args.output:
        saved = result.save(args.output, include_meta=not args.no_meta)
        if not args.quiet:
            print(f"  Report saved → {_cyan(str(saved))}", file=sys.stderr)

    # ── Print to stdout ───────────────────────────────────────────────────────
    if not args.no_print:
        print(result.report)

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    # ── Resolve the query ─────────────────────────────────────────────────────
    query: str = ""

    if args.query:
        query = args.query.strip()
    elif sys.stdin.isatty():
        # Interactive fallback: prompt the user
        print(_bold("ResearchFlow — enter your research question:"), file=sys.stderr)
        try:
            query = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            sys.exit(1)
    else:
        # Non-interactive stdin: read the query from a pipe
        query = sys.stdin.read().strip()

    if not query:
        parser.print_help()
        sys.exit(1)

# ── Env check ─────────────────────────────────────────────────────────────
    try:
        env_warnings = _check_env()
    except EnvironmentError as exc:
        print(_red(f"\nError: {exc}"), file=sys.stderr)
        sys.exit(1)

    for w in env_warnings:                          # ← keep only this one
        print(_yellow(f"  Warning: {w}"), file=sys.stderr)

    # ── Redis cache setup ─────────────────────────────────────────────────────
    try:
        set_llm_cache(RedisCache(redis_=get_redis_client()))
        logger.info("Redis LLM cache enabled")
    except Exception as exc:
        print(_yellow(f"  Warning: Redis cache unavailable — {exc}"), file=sys.stderr)

    # ── Confirmation ──────────────────────────────────────────────────────────
    if not _confirm(query, args):
        print("Aborted.", file=sys.stderr)
        sys.exit(0)

    # ── Run ───────────────────────────────────────────────────────────────────
    exit_code = asyncio.run(_arun(query, args))
    sys.exit(exit_code)                             # ← this stays LAST

    

if __name__ == "__main__":
    main()