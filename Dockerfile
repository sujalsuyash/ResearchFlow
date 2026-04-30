FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the install step.
# Only re-runs when requirements.txt changes, not on every code change.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model ─────────────────────────────────────────
# Downloads all-MiniLM-L6-v2 into the image at build time so:
#   • No HuggingFace download on first request (saves 6-7s cold start)
#   • No HF_TOKEN needed at runtime
#   • Works even if HuggingFace is unreachable at runtime
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', \
cache_folder='/app/models')"

# ── Copy application code ─────────────────────────────────────────────────────
# Done AFTER pip install so code changes don't invalidate the dependency cache
COPY . .

# ── Environment defaults ──────────────────────────────────────────────────────
# Tell sentence-transformers to use the pre-downloaded model
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
# Disable HuggingFace Hub telemetry
ENV HF_HUB_DISABLE_TELEMETRY=1
# Ensure Python output is not buffered (logs appear immediately)
ENV PYTHONUNBUFFERED=1
# Disable HuggingFace progress bars in logs
ENV TRANSFORMERS_VERBOSITY=error

# ── Port ──────────────────────────────────────────────────────────────────────
# Railway injects $PORT at runtime; default to 8000 for local development.
ENV PORT=8000
EXPOSE ${PORT}

# ── Health check ─────────────────────────────────────────────────────────────
# Docker/Railway will restart the container if /health stops returning 200.
# Uses $PORT so it matches whatever Railway assigns at runtime.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# Shell form (not JSON array) so $PORT is expanded at runtime by the shell.
# Workers=1 keeps memory usage low on free-tier deployments.
# Increase to 2-4 workers on paid plans for better concurrency.
CMD python -m uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1