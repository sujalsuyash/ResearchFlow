<<<<<<< HEAD
# ResearchFlow

**Autonomous academic research agent.** Submit a natural-language question, get back a cited Markdown report grounded in real papers from Semantic Scholar, PubMed, OpenAlex, and arXiv — in under 60 seconds on fresh API keys.

```
POST /research   →   202 Accepted + job_id
GET  /research/{job_id}   →   { status, report, papers, timings }
```

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the Server](#running-the-server)
- [CLI Usage](#cli-usage)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Design Decisions](#design-decisions)

---

## Problem Statement

Academic research is slow by default. Finding relevant papers across multiple databases, deduplicating results, filtering noise, and synthesising findings into a coherent summary takes hours of manual work. Existing tools either search a single database, return raw results with no filtering, or produce summaries with no citations.

ResearchFlow solves this by chaining five stages into a single autonomous pipeline:

1. **Decompose** the query into targeted sub-questions via an LLM
2. **Fan out** across four databases in parallel
3. **Filter** results through a four-layer relevance pipeline (including local semantic embeddings — no API calls)
4. **Enrich** survivors with open-access PDF links
5. **Synthesise** a cited Markdown report via an LLM

The API is designed around a job-queue pattern because the pipeline takes 36–60 seconds. Keeping an HTTP connection open that long breaks on proxies and mobile clients. Instead, `POST /research` returns a `job_id` immediately and the client polls `GET /research/{job_id}` until done.

---

## How It Works

```
POST /research
     │
     ▼
Job queued in Redis  (status: pending)
     │
     ▼
ThreadPoolExecutor   (status: running)
     │
     ▼
Stage 1 — PLAN (~5s)
  QueryPlanner asks the LLM to decompose the query into 4 diverse
  sub-questions, each targeting a different angle and database.
     │
     ▼
Stage 2 — EXECUTE (~20s)
  ResearchAgent fans out in parallel to:
    • Semantic Scholar  (200M+ papers, all domains)
    • PubMed / MEDLINE  (35M+ biomedical papers)
    • OpenAlex          (250M+ works, fully open)
    • arXiv             (preprints, CS / physics / math)
  Up to 10 papers per tool per step → ~40–50 raw papers collected.
     │
     ▼
Stage 3 — DEDUP (~0ms)
  Exact-match deduplication by DOI + title normalisation.
  Survivors sorted by citation count descending.
     │
     ▼
Stage 4 — ENRICH (~5s)
  Unpaywall API injects open-access PDF URLs where available.
     │
     ▼
Stage 5 — FILTER (~0.2s)
  Four-layer relevance filter (see below).
     │
     ▼
Stage 6 — SYNTHESISE (~4s)
  SynthesisChain sends the filtered papers to the LLM.
  Output: cited Markdown report with bibliography.
     │
     ▼
Result saved to Redis  (status: done, TTL: 1 hour)
     │
     ▼
GET /research/{job_id}  →  full report + stage timings
```

### The Four-Layer Relevance Filter

This is the core of what makes ResearchFlow's output quality high. It runs entirely locally — no API calls.

**Layer 1 — Quality pre-checks** (pure Python, zero cost)
Drops papers with no title+abstract, stub titles, raw blobs from failed API parsing, or publication years below a field-aware floor. The year floor is auto-detected: CS/ML papers default to 2018 (the field moves fast), biomedical papers are allowed back to 1980 (clinical literature stays relevant for decades).

**Layer 2 — Lexical overlap** (zero cost)
Tokenises the user query and each paper's title+abstract. Stopwords are excluded. The required overlap is calculated as `ceil(len(query_tokens) × 0.65)` — so a 10-token query requires 7 shared content words. A second-chance pass against title+abstract combined allows papers to survive with one extra word beyond the primary threshold.

**Layer 3 — Semantic similarity** (local model, no API calls)
Embeds the query and each paper's `title + abstract` using `sentence-transformers/all-MiniLM-L6-v2` (80 MB, CPU-friendly). Cosine similarity is computed via a vectorised matrix multiply. Papers below the 0.30 threshold are dropped. For broad multi-topic queries (detected by signal phrases like "role of", "impact of", "comparison of"), the threshold is automatically raised by +0.10 to compensate for the diffuse embedding space wide queries produce.

**Layer 4 — Citation-weighted reranking** (pure maths, zero cost)
Survivors are scored by:
```
score = semantic_similarity × log(citation_count + 2) × recency_weight
```
`recency_weight` decays exponentially with paper age. The half-life is field-aware: 4 years for CS, 10 years for biomedical, 7 years for general topics. The top `max_papers` are returned in descending score order.

### LLM Resilience

`ResilientLLM` is a LangChain-compatible `Runnable` that implements ordered multi-provider fallback:

```
Groq key 1  →  Gemini  →  Groq key 2  →  Groq key 3
```

Gemini is placed second (not last) because all three Groq keys share the same organisation's daily token quota. When key 1 hits the 100k token/day limit, keys 2 and 3 fail instantly for the same reason. Gemini absorbs the overflow, with the remaining Groq keys as last-resort fallback for Gemini outages.

A 429 or quota error on any provider causes an immediate retry on the next. All other errors are re-raised so bugs surface clearly. The error detection covers Groq SDK native errors, httpx HTTP 429 status codes, and a string heuristic that catches LangChain wrapping and Gemini gRPC `RESOURCE_EXHAUSTED` messages.

---

## Architecture

```
researchflow/
├── api.py                     ← FastAPI app + job queue (entry point for the server)
├── main.py                    ← CLI entry point
│
├── core/
│   ├── pipeline.py            ← Orchestrator: 6 stages, per-stage timings, progress hooks
│   ├── filter.py              ← 4-layer relevance filter + embedding model singleton
│   └── cache.py               ← Redis client factory (shared across tools)
│
├── agents/
│   └── research_agent.py      ← ResilientLLM + ResearchAgent + tool loader
│
├── chains/
│   ├── query_planner.py       ← LLM-powered query decomposition → ResearchPlan
│   └── synthesizer.py         ← LLM-powered cited report generation
│
├── tools/
│   ├── semantic_scholar.py    ← Semantic Scholar Graph API wrapper
│   ├── pubmed_search.py       ← NCBI PubMed E-utilities wrapper (esearch + efetch)
│   ├── openalex_search.py     ← OpenAlex Works API wrapper
│   ├── arxiv_search.py        ← arXiv API wrapper
│   └── unpaywall_fetcher.py   ← Unpaywall open-access PDF enrichment
│
├── scripts/
│   └── preload_model.py       ← Bakes embedding model into Docker image layer
│
└── tests/                     ← pytest suite covering all tools and chains
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| API framework | FastAPI | Native async, automatic OpenAPI docs, Pydantic validation |
| Job queue | Redis (Upstash) | Persistent job state, 1-hour TTL, survives server restarts |
| LLM providers | Groq (llama-3.3-70b) + Gemini (gemini-2.0-flash) | Groq for speed, Gemini as fallback; both free-tier viable |
| LLM framework | LangChain | Structured output, prompt templates, tool abstractions |
| Embedding model | all-MiniLM-L6-v2 | 80 MB, CPU-only, 0.2s inference after startup preload |
| Academic databases | Semantic Scholar, PubMed, OpenAlex, arXiv | Complementary coverage: CS, biomedical, cross-domain, preprints |
| PDF enrichment | Unpaywall | DOI-based open-access PDF discovery, no auth required |
| Caching | Redis (7-day TTL per query) | Identical sub-queries skip the API entirely |
| HTTP client | httpx | Async-native, used by all tool wrappers |
| Concurrency | ThreadPoolExecutor (6 workers) | Keeps FastAPI's event loop free from blocking pipeline work |

---

## Project Structure

### `api.py` — HTTP entry point

FastAPI application with two endpoints and startup preloading logic.

The `_warm_embedding_model` startup event runs `preload_embedding_model()` inside `asyncio.get_running_loop().run_in_executor()`. This is critical: `SentenceTransformer()` does blocking file I/O and CPU work. Running it directly in the async startup handler would freeze the event loop for 6–8 seconds. The executor keeps the loop free while the model loads.

Job state is stored in Redis as JSON with a 1-hour TTL. The capacity check (`running_jobs >= max_workers`) returns a 503 with a clear error message rather than silently queuing requests that will never be processed within a reasonable time.

### `core/pipeline.py` — Stage orchestrator

`ResearchPipeline` wraps `ResearchAgent` with per-stage wall-clock timing, optional `ProgressCallback` hooks, and a `PipelineResult` dataclass that carries the report, paper list, and all run metadata. The same object supports both `run()` (synchronous, for CLI use) and `arun()` (async, for FastAPI).

`PipelineResult.save()` writes the report to disk with optional YAML front-matter containing query, paper count, and total runtime.

`build_pipeline_from_env()` is the one-call factory that reads environment variables and wires every component together.

### `core/filter.py` — Relevance filter

The embedding model is a module-level singleton loaded lazily on first call and cached for the lifetime of the process. `preload_embedding_model()` exposes this for startup preloading. If `sentence-transformers` is not installed, Layer 3 is skipped with a warning and only Layers 1, 2, and 4 run — the pipeline degrades gracefully rather than crashing.

### `agents/research_agent.py` — Research orchestrator

`ResilientLLM` uses a generator (`_iter_providers`) to yield providers one at a time. This is deliberate: `_make_gemini()` is never called when a Groq key succeeds, avoiding unnecessary object construction. The tool loader (`_lazy_import_tool`) uses a four-pass resolution strategy to find runnable tools regardless of how they are exported from their modules.

### `main.py` — CLI interface

Full-featured CLI with argument parsing, environment validation, Redis LLM cache setup, interactive confirmation prompt, and ANSI colour output on TTY. Progress is written to stderr so piping stdout to a file captures only the final Markdown report.

```bash
python main.py "CRISPR off-target effects in clinical trials" --output report.md
python main.py "Alzheimer's treatments" --max-papers 60 --papers-per-step 15
python main.py "quantum error correction" --verbose
python main.py "solar cell efficiency" --quiet --yes
```

### `tools/` — Database wrappers

Each tool follows the same pattern: token-bucket rate limiter, Redis cache (7-day TTL, MD5-keyed on query parameters), Redis-unavailable fallback to an in-process dict, exponential backoff on 429s, and a LangChain `StructuredTool` wrapper for agent compatibility. Redis credentials are logged at hostname only — never the full connection string.

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- A Redis instance (Upstash free tier works)
- At least one Groq API key (free at [console.groq.com](https://console.groq.com))

### Install dependencies

```bash
pip install fastapi uvicorn[standard] python-dotenv redis certifi httpx \
            pydantic langchain langchain-core langchain-groq \
            langchain-google-genai langchain-community \
            sentence-transformers
```

### Clone and configure

```bash
git clone https://github.com/your-username/researchflow.git
cd researchflow
cp .env.example .env   # then fill in your keys
```

---

## Environment Variables

Create a `.env` file in the project root:

```dotenv
# LLM providers — at least one Groq key is required
GROQ_KEY_1=gsk_...
GROQ_KEY_2=gsk_...          # optional — same org = same daily quota
GROQ_KEY_3=gsk_...          # optional
GOOGLE_API_KEY=AIza...      # optional — Gemini fallback

# Academic databases
SEMANTIC_SCHOLAR_API_KEY=s2k-...   # optional — raises rate limit 1→100 req/s
S2_API_KEY=s2k-...                 # same key, read by the tool module directly

# Job store & tool cache
REDIS_URL=rediss://default:PASSWORD@hostname:6379

# Embedding model (offline mode — skip HuggingFace version checks)
HF_TOKEN=hf_...
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

> **Security**: Never commit `.env`. Confirm `.gitignore` contains `.env`, `*.env`, and `__pycache__/`.

---

## Running the Server

```bash
# Development (with auto-reload)
python -m uvicorn api:app --reload --port 8000

# Production
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use `--workers 1` in production. The embedding model singleton is process-local — multiple worker processes each load their own copy, which is fine, but `workers > 1` is not necessary given the `ThreadPoolExecutor` handles concurrency internally.

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## CLI Usage

```bash
# Basic — prints report to stdout
python main.py "attention mechanisms in transformer neural networks"

# Save to file with YAML front-matter
python main.py "CRISPR off-target effects" --output report.md

# Tune retrieval
python main.py "Alzheimer's treatments 2024" --max-papers 60 --papers-per-step 15

# Verbose (DEBUG logging)
python main.py "quantum error correction" --verbose

# Silent (only final report, no progress output)
python main.py "solar cell efficiency records" --quiet --yes
```

---

## API Reference

### `POST /research`

Submit a research query. Returns immediately with a `job_id`.

**Request body:**
```json
{
  "query": "neuro-symbolic AI in medical diagnosis",
  "max_papers": 40,
  "papers_per_step": 10,
  "tool_timeout": 120.0
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job queued. Poll GET /research/{job_id} for updates."
}
```

---

### `GET /research/{job_id}`

Poll for results. Status transitions: `pending → running → done | failed`.

**Response when done:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "done",
  "query": "neuro-symbolic AI in medical diagnosis",
  "report": "# Research Summary: ...",
  "paper_count": 21,
  "raw_paper_count": 47,
  "plan_steps": 4,
  "total_seconds": 36.4,
  "stage_timings": {
    "plan": 5.3,
    "execute": 21.3,
    "dedup": 0.0,
    "enrich": 5.1,
    "filter": 0.2,
    "synthesise": 4.5
  },
  "error": null
}
```

Jobs expire from Redis after 1 hour.

---

### `GET /health`

Liveness check. Returns Redis connectivity status.

```json
{ "status": "ok", "redis": "connected" }
```

---

## Performance

Measured on fresh API keys with no rate limiting:

| Stage | Time | Notes |
|---|---|---|
| plan | ~5s | 4–6 LLM calls for query decomposition |
| execute | ~20s | Parallel fan-out across 4 databases |
| dedup | ~0ms | Pure Python dict operations |
| enrich | ~5s | Parallel Unpaywall lookups |
| filter | ~0.2s | Embedding model preloaded at startup |
| synthesise | ~4s | Single LLM call |
| **total** | **~36s** | |

The embedding model cold start (6–8s) is paid once at server startup, not per request. Subsequent filter passes take 0.2–0.4s regardless of paper count.

On exhausted free-tier keys, rate-limit backoff dominates. The same pipeline runs in 150–220s when both Groq and Gemini are throttling simultaneously. Rotating to fresh keys resolves this immediately.

---

## Design Decisions

**Why a job queue instead of streaming?**
The pipeline takes 36–60 seconds. Keeping an HTTP connection open that long breaks on load balancers, mobile clients, and proxies. The `POST → poll` pattern is reliable across all network conditions. Redis provides durable job state that survives server restarts.

**Why run the pipeline in a ThreadPoolExecutor?**
`SentenceTransformer` inference and the synchronous parts of the tool chain are blocking. Running them directly in FastAPI's async handlers would stall the event loop, making the server unresponsive to all other requests during a pipeline run. Each thread gets its own event loop for the async portions.

**Why is Gemini second in the fallback chain, not last?**
Multiple Groq keys from the same account share one daily token quota. When key 1 hits the limit, key 2 fails instantly for the same reason. Gemini is placed second to absorb the overflow, with remaining Groq keys as last-resort fallback for Gemini outages.

**Why preload the embedding model at startup?**
Without preloading, the first pipeline run pays 6–8 seconds mid-pipeline while the model loads. The `startup` event runs `preload_embedding_model()` in an executor, paying this cost once during boot. All subsequent runs use the cached singleton at near-zero cost.

**Why 6 ThreadPoolExecutor workers?**
4 active pipelines + 2 headroom for burst traffic. Right-sized for ~10 concurrent users given that pipelines take 36–60 seconds and not all users submit simultaneously. The 7th simultaneous request gets a 503 with a clear retry message rather than silently queuing.
=======
# ResearchFlow

**Autonomous academic research agent.** Submit a natural-language question, get back a cited Markdown report grounded in real papers from Semantic Scholar, PubMed, OpenAlex, and arXiv — in under 60 seconds on fresh API keys.

```
POST /research   →   202 Accepted + job_id
GET  /research/{job_id}   →   { status, report, papers, timings }
```

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the Server](#running-the-server)
- [CLI Usage](#cli-usage)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Design Decisions](#design-decisions)

---

## Problem Statement

Academic research is slow by default. Finding relevant papers across multiple databases, deduplicating results, filtering noise, and synthesising findings into a coherent summary takes hours of manual work. Existing tools either search a single database, return raw results with no filtering, or produce summaries with no citations.

ResearchFlow solves this by chaining five stages into a single autonomous pipeline:

1. **Decompose** the query into targeted sub-questions via an LLM
2. **Fan out** across four databases in parallel
3. **Filter** results through a four-layer relevance pipeline (including local semantic embeddings — no API calls)
4. **Enrich** survivors with open-access PDF links
5. **Synthesise** a cited Markdown report via an LLM

The API is designed around a job-queue pattern because the pipeline takes 36–60 seconds. Keeping an HTTP connection open that long breaks on proxies and mobile clients. Instead, `POST /research` returns a `job_id` immediately and the client polls `GET /research/{job_id}` until done.

---

## How It Works

```
POST /research
     │
     ▼
Job queued in Redis  (status: pending)
     │
     ▼
ThreadPoolExecutor   (status: running)
     │
     ▼
Stage 1 — PLAN (~5s)
  QueryPlanner asks the LLM to decompose the query into 4 diverse
  sub-questions, each targeting a different angle and database.
     │
     ▼
Stage 2 — EXECUTE (~20s)
  ResearchAgent fans out in parallel to:
    • Semantic Scholar  (200M+ papers, all domains)
    • PubMed / MEDLINE  (35M+ biomedical papers)
    • OpenAlex          (250M+ works, fully open)
    • arXiv             (preprints, CS / physics / math)
  Up to 10 papers per tool per step → ~40–50 raw papers collected.
     │
     ▼
Stage 3 — DEDUP (~0ms)
  Exact-match deduplication by DOI + title normalisation.
  Survivors sorted by citation count descending.
     │
     ▼
Stage 4 — ENRICH (~5s)
  Unpaywall API injects open-access PDF URLs where available.
     │
     ▼
Stage 5 — FILTER (~0.2s)
  Four-layer relevance filter (see below).
     │
     ▼
Stage 6 — SYNTHESISE (~4s)
  SynthesisChain sends the filtered papers to the LLM.
  Output: cited Markdown report with bibliography.
     │
     ▼
Result saved to Redis  (status: done, TTL: 1 hour)
     │
     ▼
GET /research/{job_id}  →  full report + stage timings
```

### The Four-Layer Relevance Filter

This is the core of what makes ResearchFlow's output quality high. It runs entirely locally — no API calls.

**Layer 1 — Quality pre-checks** (pure Python, zero cost)
Drops papers with no title+abstract, stub titles, raw blobs from failed API parsing, or publication years below a field-aware floor. The year floor is auto-detected: CS/ML papers default to 2018 (the field moves fast), biomedical papers are allowed back to 1980 (clinical literature stays relevant for decades).

**Layer 2 — Lexical overlap** (zero cost)
Tokenises the user query and each paper's title+abstract. Stopwords are excluded. The required overlap is calculated as `ceil(len(query_tokens) × 0.65)` — so a 10-token query requires 7 shared content words. A second-chance pass against title+abstract combined allows papers to survive with one extra word beyond the primary threshold.

**Layer 3 — Semantic similarity** (local model, no API calls)
Embeds the query and each paper's `title + abstract` using `sentence-transformers/all-MiniLM-L6-v2` (80 MB, CPU-friendly). Cosine similarity is computed via a vectorised matrix multiply. Papers below the 0.30 threshold are dropped. For broad multi-topic queries (detected by signal phrases like "role of", "impact of", "comparison of"), the threshold is automatically raised by +0.10 to compensate for the diffuse embedding space wide queries produce.

**Layer 4 — Citation-weighted reranking** (pure maths, zero cost)
Survivors are scored by:
```
score = semantic_similarity × log(citation_count + 2) × recency_weight
```
`recency_weight` decays exponentially with paper age. The half-life is field-aware: 4 years for CS, 10 years for biomedical, 7 years for general topics. The top `max_papers` are returned in descending score order.

### LLM Resilience

`ResilientLLM` is a LangChain-compatible `Runnable` that implements ordered multi-provider fallback:

```
Groq key 1  →  Gemini  →  Groq key 2  →  Groq key 3
```

Gemini is placed second (not last) because all three Groq keys share the same organisation's daily token quota. When key 1 hits the 100k token/day limit, keys 2 and 3 fail instantly for the same reason. Gemini absorbs the overflow, with the remaining Groq keys as last-resort fallback for Gemini outages.

A 429 or quota error on any provider causes an immediate retry on the next. All other errors are re-raised so bugs surface clearly. The error detection covers Groq SDK native errors, httpx HTTP 429 status codes, and a string heuristic that catches LangChain wrapping and Gemini gRPC `RESOURCE_EXHAUSTED` messages.

---

## Architecture

```
researchflow/
├── api.py                     ← FastAPI app + job queue (entry point for the server)
├── main.py                    ← CLI entry point
│
├── core/
│   ├── pipeline.py            ← Orchestrator: 6 stages, per-stage timings, progress hooks
│   ├── filter.py              ← 4-layer relevance filter + embedding model singleton
│   └── cache.py               ← Redis client factory (shared across tools)
│
├── agents/
│   └── research_agent.py      ← ResilientLLM + ResearchAgent + tool loader
│
├── chains/
│   ├── query_planner.py       ← LLM-powered query decomposition → ResearchPlan
│   └── synthesizer.py         ← LLM-powered cited report generation
│
├── tools/
│   ├── semantic_scholar.py    ← Semantic Scholar Graph API wrapper
│   ├── pubmed_search.py       ← NCBI PubMed E-utilities wrapper (esearch + efetch)
│   ├── openalex_search.py     ← OpenAlex Works API wrapper
│   ├── arxiv_search.py        ← arXiv API wrapper
│   └── unpaywall_fetcher.py   ← Unpaywall open-access PDF enrichment
│
├── scripts/
│   └── preload_model.py       ← Bakes embedding model into Docker image layer
│
└── tests/                     ← pytest suite covering all tools and chains
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| API framework | FastAPI | Native async, automatic OpenAPI docs, Pydantic validation |
| Job queue | Redis (Upstash) | Persistent job state, 1-hour TTL, survives server restarts |
| LLM providers | Groq (llama-3.3-70b) + Gemini (gemini-2.0-flash) | Groq for speed, Gemini as fallback; both free-tier viable |
| LLM framework | LangChain | Structured output, prompt templates, tool abstractions |
| Embedding model | all-MiniLM-L6-v2 | 80 MB, CPU-only, 0.2s inference after startup preload |
| Academic databases | Semantic Scholar, PubMed, OpenAlex, arXiv | Complementary coverage: CS, biomedical, cross-domain, preprints |
| PDF enrichment | Unpaywall | DOI-based open-access PDF discovery, no auth required |
| Caching | Redis (7-day TTL per query) | Identical sub-queries skip the API entirely |
| HTTP client | httpx | Async-native, used by all tool wrappers |
| Concurrency | ThreadPoolExecutor (6 workers) | Keeps FastAPI's event loop free from blocking pipeline work |

---

## Project Structure

### `api.py` — HTTP entry point

FastAPI application with two endpoints and startup preloading logic.

The `_warm_embedding_model` startup event runs `preload_embedding_model()` inside `asyncio.get_running_loop().run_in_executor()`. This is critical: `SentenceTransformer()` does blocking file I/O and CPU work. Running it directly in the async startup handler would freeze the event loop for 6–8 seconds. The executor keeps the loop free while the model loads.

Job state is stored in Redis as JSON with a 1-hour TTL. The capacity check (`running_jobs >= max_workers`) returns a 503 with a clear error message rather than silently queuing requests that will never be processed within a reasonable time.

### `core/pipeline.py` — Stage orchestrator

`ResearchPipeline` wraps `ResearchAgent` with per-stage wall-clock timing, optional `ProgressCallback` hooks, and a `PipelineResult` dataclass that carries the report, paper list, and all run metadata. The same object supports both `run()` (synchronous, for CLI use) and `arun()` (async, for FastAPI).

`PipelineResult.save()` writes the report to disk with optional YAML front-matter containing query, paper count, and total runtime.

`build_pipeline_from_env()` is the one-call factory that reads environment variables and wires every component together.

### `core/filter.py` — Relevance filter

The embedding model is a module-level singleton loaded lazily on first call and cached for the lifetime of the process. `preload_embedding_model()` exposes this for startup preloading. If `sentence-transformers` is not installed, Layer 3 is skipped with a warning and only Layers 1, 2, and 4 run — the pipeline degrades gracefully rather than crashing.

### `agents/research_agent.py` — Research orchestrator

`ResilientLLM` uses a generator (`_iter_providers`) to yield providers one at a time. This is deliberate: `_make_gemini()` is never called when a Groq key succeeds, avoiding unnecessary object construction. The tool loader (`_lazy_import_tool`) uses a four-pass resolution strategy to find runnable tools regardless of how they are exported from their modules.

### `main.py` — CLI interface

Full-featured CLI with argument parsing, environment validation, Redis LLM cache setup, interactive confirmation prompt, and ANSI colour output on TTY. Progress is written to stderr so piping stdout to a file captures only the final Markdown report.

```bash
python main.py "CRISPR off-target effects in clinical trials" --output report.md
python main.py "Alzheimer's treatments" --max-papers 60 --papers-per-step 15
python main.py "quantum error correction" --verbose
python main.py "solar cell efficiency" --quiet --yes
```

### `tools/` — Database wrappers

Each tool follows the same pattern: token-bucket rate limiter, Redis cache (7-day TTL, MD5-keyed on query parameters), Redis-unavailable fallback to an in-process dict, exponential backoff on 429s, and a LangChain `StructuredTool` wrapper for agent compatibility. Redis credentials are logged at hostname only — never the full connection string.

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- A Redis instance (Upstash free tier works)
- At least one Groq API key (free at [console.groq.com](https://console.groq.com))

### Install dependencies

```bash
pip install fastapi uvicorn[standard] python-dotenv redis certifi httpx \
            pydantic langchain langchain-core langchain-groq \
            langchain-google-genai langchain-community \
            sentence-transformers
```

### Clone and configure

```bash
git clone https://github.com/your-username/researchflow.git
cd researchflow
cp .env.example .env   # then fill in your keys
```

---

## Environment Variables

Create a `.env` file in the project root:

```dotenv
# LLM providers — at least one Groq key is required
GROQ_KEY_1=gsk_...
GROQ_KEY_2=gsk_...          # optional — same org = same daily quota
GROQ_KEY_3=gsk_...          # optional
GOOGLE_API_KEY=AIza...      # optional — Gemini fallback

# Academic databases
SEMANTIC_SCHOLAR_API_KEY=s2k-...   # optional — raises rate limit 1→100 req/s
S2_API_KEY=s2k-...                 # same key, read by the tool module directly

# Job store & tool cache
REDIS_URL=rediss://default:PASSWORD@hostname:6379

# Embedding model (offline mode — skip HuggingFace version checks)
HF_TOKEN=hf_...
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

> **Security**: Never commit `.env`. Confirm `.gitignore` contains `.env`, `*.env`, and `__pycache__/`.

---

## Running the Server

```bash
# Development (with auto-reload)
python -m uvicorn api:app --reload --port 8000

# Production
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use `--workers 1` in production. The embedding model singleton is process-local — multiple worker processes each load their own copy, which is fine, but `workers > 1` is not necessary given the `ThreadPoolExecutor` handles concurrency internally.

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## CLI Usage

```bash
# Basic — prints report to stdout
python main.py "attention mechanisms in transformer neural networks"

# Save to file with YAML front-matter
python main.py "CRISPR off-target effects" --output report.md

# Tune retrieval
python main.py "Alzheimer's treatments 2024" --max-papers 60 --papers-per-step 15

# Verbose (DEBUG logging)
python main.py "quantum error correction" --verbose

# Silent (only final report, no progress output)
python main.py "solar cell efficiency records" --quiet --yes
```

---

## API Reference

### `POST /research`

Submit a research query. Returns immediately with a `job_id`.

**Request body:**
```json
{
  "query": "neuro-symbolic AI in medical diagnosis",
  "max_papers": 40,
  "papers_per_step": 10,
  "tool_timeout": 120.0
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job queued. Poll GET /research/{job_id} for updates."
}
```

---

### `GET /research/{job_id}`

Poll for results. Status transitions: `pending → running → done | failed`.

**Response when done:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "done",
  "query": "neuro-symbolic AI in medical diagnosis",
  "report": "# Research Summary: ...",
  "paper_count": 21,
  "raw_paper_count": 47,
  "plan_steps": 4,
  "total_seconds": 36.4,
  "stage_timings": {
    "plan": 5.3,
    "execute": 21.3,
    "dedup": 0.0,
    "enrich": 5.1,
    "filter": 0.2,
    "synthesise": 4.5
  },
  "error": null
}
```

Jobs expire from Redis after 1 hour.

---

### `GET /health`

Liveness check. Returns Redis connectivity status.

```json
{ "status": "ok", "redis": "connected" }
```

---

## Performance

Measured on fresh API keys with no rate limiting:

| Stage | Time | Notes |
|---|---|---|
| plan | ~5s | 4–6 LLM calls for query decomposition |
| execute | ~20s | Parallel fan-out across 4 databases |
| dedup | ~0ms | Pure Python dict operations |
| enrich | ~5s | Parallel Unpaywall lookups |
| filter | ~0.2s | Embedding model preloaded at startup |
| synthesise | ~4s | Single LLM call |
| **total** | **~36s** | |

The embedding model cold start (6–8s) is paid once at server startup, not per request. Subsequent filter passes take 0.2–0.4s regardless of paper count.

On exhausted free-tier keys, rate-limit backoff dominates. The same pipeline runs in 150–220s when both Groq and Gemini are throttling simultaneously. Rotating to fresh keys resolves this immediately.

---

## Design Decisions

**Why a job queue instead of streaming?**
The pipeline takes 36–60 seconds. Keeping an HTTP connection open that long breaks on load balancers, mobile clients, and proxies. The `POST → poll` pattern is reliable across all network conditions. Redis provides durable job state that survives server restarts.

**Why run the pipeline in a ThreadPoolExecutor?**
`SentenceTransformer` inference and the synchronous parts of the tool chain are blocking. Running them directly in FastAPI's async handlers would stall the event loop, making the server unresponsive to all other requests during a pipeline run. Each thread gets its own event loop for the async portions.

**Why is Gemini second in the fallback chain, not last?**
Multiple Groq keys from the same account share one daily token quota. When key 1 hits the limit, key 2 fails instantly for the same reason. Gemini is placed second to absorb the overflow, with remaining Groq keys as last-resort fallback for Gemini outages.

**Why preload the embedding model at startup?**
Without preloading, the first pipeline run pays 6–8 seconds mid-pipeline while the model loads. The `startup` event runs `preload_embedding_model()` in an executor, paying this cost once during boot. All subsequent runs use the cached singleton at near-zero cost.

**Why 6 ThreadPoolExecutor workers?**
4 active pipelines + 2 headroom for burst traffic. Right-sized for ~10 concurrent users given that pipelines take 36–60 seconds and not all users submit simultaneously. The 7th simultaneous request gets a 503 with a clear retry message rather than silently queuing.
>>>>>>> 1dd8484 (fix: Redis SSL cert, Groq key rotation order, resilient LLM fallback fix)
