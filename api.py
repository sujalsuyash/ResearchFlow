from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import uuid
from enum import Enum
from typing import Any, Optional

import certifi
import redis
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
# Embedding model singleton — imported here so the startup event can
# preload it before the first request arrives.
from core.filter import preload_embedding_model as _preload_embedding_model  # noqa: E402

# Thread pool for running the CPU-bound / blocking pipeline
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("researchflow.api")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResearchFlow API",
    description="Autonomous academic research agent — submit a query, get a cited report.",
    version="1.0.0",
)

# Allow all origins for now — tighten this once you have a real frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: warm the embedding model ─────────────────────────────────────────

@app.on_event("startup")
async def _warm_embedding_model() -> None:
    """
    Load the SentenceTransformer into RAM before the first request arrives.

    Running this in a thread executor keeps the FastAPI event loop free
    during the ~6 s CPU/IO burst of the initial model load.  All subsequent
    calls to get_embedding_model() return the cached instance in microseconds.

    If the model is already on disk (Docker image baked with
    scripts/preload_model.py) this completes in ~0.3 s.
    If it needs to be downloaded it takes ~8 s on first boot only.
    """
    import asyncio  # noqa: PLC0415 — already imported at module level, this is fine
    await asyncio.get_running_loop().run_in_executor(None, _preload_embedding_model)
    logger.info("Startup: embedding model warm and ready.")


# ── Redis job store ────────────────────────────────────────────────────────────

def _get_redis() -> redis.Redis:
    """Reuse the same Redis connection config as core/cache.py."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise EnvironmentError("REDIS_URL is not set.")
    return redis.from_url(
        redis_url,
        ssl_ca_certs=certifi.where(),
        ssl_cert_reqs="required",
        socket_keepalive=True,
        socket_timeout=10,
        socket_connect_timeout=10,
        retry_on_timeout=True,
        health_check_interval=30,
        decode_responses=True,       # return str, not bytes
    )


JOB_TTL_SECONDS = 3600  # jobs expire from Redis after 1 hour
JOB_KEY_PREFIX  = "researchflow:job:"


def _job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def _save_job(r: redis.Redis, job_id: str, data: dict) -> None:
    r.setex(_job_key(job_id), JOB_TTL_SECONDS, json.dumps(data))


def _load_job(r: redis.Redis, job_id: str) -> Optional[dict]:
    raw = r.get(_job_key(job_id))
    return json.loads(raw) if raw else None


# ── Schemas ───────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"


class ResearchRequest(BaseModel):
    query:              str = Field(..., min_length=3, max_length=500,
                                   description="Research question or topic.")
    max_papers:         int = Field(default=40,  ge=5,  le=100)
    papers_per_step:    int = Field(default=10,  ge=3,  le=20)
    tool_timeout:     float = Field(default=120.0, ge=10.0, le=300.0)


class JobAccepted(BaseModel):
    job_id:  str
    status:  JobStatus = JobStatus.PENDING
    message: str       = "Job queued. Poll GET /research/{job_id} for updates."


class StageTimings(BaseModel):
    plan:       Optional[float] = None
    execute:    Optional[float] = None
    dedup:      Optional[float] = None
    enrich:     Optional[float] = None
    filter:     Optional[float] = None
    synthesise: Optional[float] = None


class JobResult(BaseModel):
    job_id:          str
    status:          JobStatus
    query:           Optional[str]   = None
    report:          Optional[str]   = None
    paper_count:     Optional[int]   = None
    raw_paper_count: Optional[int]   = None
    plan_steps:      Optional[int]   = None
    total_seconds:   Optional[float] = None
    stage_timings:   Optional[StageTimings] = None
    error:           Optional[str]   = None


# ── Background worker ─────────────────────────────────────────────────────────

def _run_pipeline_in_thread(job_id: str, request: ResearchRequest) -> None:
    """
    Runs the full pipeline in a dedicated thread with its own event loop.

    Why a thread?
    ─────────────
    The RelevanceFilter loads sentence-transformers (all-MiniLM-L6-v2), which
    is CPU-bound and blocking. Running it directly as a FastAPI BackgroundTask
    would freeze the shared event loop, making the server unresponsive.
    A separate thread with its own event loop keeps FastAPI's loop free.
    """
    # Each thread needs its own event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_pipeline_async(job_id, request))
    finally:
        loop.close()


async def _pipeline_async(job_id: str, request: ResearchRequest) -> None:
    """Async pipeline logic — runs inside the thread's own event loop."""
    r = _get_redis()

    # Mark as running
    _save_job(r, job_id, {"status": JobStatus.RUNNING, "query": request.query})
    logger.info("Job %s — started for query: %r", job_id, request.query)

    try:
        from core.pipeline import build_pipeline_from_env  # type: ignore[import]

        pipeline = build_pipeline_from_env(
            max_papers_per_step = request.papers_per_step,
            max_total_papers    = request.max_papers,
            tool_timeout        = request.tool_timeout,
        )

        result = await pipeline.arun(request.query)

        _save_job(r, job_id, {
            "status":          JobStatus.DONE,
            "query":           result.query,
            "report":          result.report,
            "paper_count":     result.paper_count,
            "raw_paper_count": result.raw_paper_count,
            "plan_steps":      result.plan_steps,
            "total_seconds":   result.total_seconds,
            "stage_timings":   result.stage_timings,
        })
        logger.info(
            "Job %s — done: %d papers, %.1fs",
            job_id, result.paper_count, result.total_seconds,
        )

    except Exception as exc:
        logger.exception("Job %s — failed: %s", job_id, exc)
        _save_job(r, job_id, {
            "status": JobStatus.FAILED,
            "query":  request.query,
            "error":  str(exc),
        })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/research",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit a research query",
)
async def submit_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> JobAccepted:
    """
    Submit a research query. Returns a job_id immediately.
    The pipeline runs in a background thread — poll GET /research/{job_id} for results.
    """
    # Validate at least one Groq key exists before queuing
    groq_keys = [os.getenv(f"GROQ_KEY_{i}") for i in range(1, 4)]
    if not any(groq_keys):
        raise HTTPException(
            status_code=503,
            detail="No LLM API keys configured. Contact the administrator.",
        )

    job_id = str(uuid.uuid4())
    r      = _get_redis()

    # Create initial job record
    _save_job(r, job_id, {"status": JobStatus.PENDING, "query": request.query})

    # Run in a thread pool — keeps FastAPI's event loop free
    #loop = asyncio.get_event_loop()

    running_jobs = _executor._work_queue.qsize() + len([
        t for t in _executor._threads if t.is_alive()
    ])
    if running_jobs >= _executor._max_workers:
        raise HTTPException(
            status_code=503,
            detail="Server is at capacity. Please retry in a few minutes.",
        )


    #loop.run_in_executor(_executor, _run_pipeline_in_thread, job_id, request)
    asyncio.get_running_loop().run_in_executor(_executor, _run_pipeline_in_thread, job_id, request)

    return JobAccepted(job_id=job_id)


@app.get(
    "/research/{job_id}",
    response_model=JobResult,
    summary="Poll job status / retrieve result",
)
async def get_research(job_id: str) -> JobResult:
    """
    Poll the status of a submitted research job.

    - **pending**  → queued, not started yet
    - **running**  → pipeline is executing (~50–60s)
    - **done**     → report is ready in the response
    - **failed**   → error message included
    """
    r    = _get_redis()
    data = _load_job(r, job_id)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired (TTL: 1 hour).",
        )

    timings = data.get("stage_timings")
    return JobResult(
        job_id          = job_id,
        status          = data["status"],
        query           = data.get("query"),
        report          = data.get("report"),
        paper_count     = data.get("paper_count"),
        raw_paper_count = data.get("raw_paper_count"),
        plan_steps      = data.get("plan_steps"),
        total_seconds   = data.get("total_seconds"),
        stage_timings   = StageTimings(**timings) if timings else None,
        error           = data.get("error"),
    )


@app.get(
    "/health",
    summary="Liveness check",
)
async def health() -> dict:
    """Returns 200 if the API is up. Also checks Redis connectivity."""
    try:
        r = _get_redis()
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok",
        "redis":  "connected" if redis_ok else "unreachable",
    }