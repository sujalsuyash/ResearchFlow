"""
scripts/preload_model.py
────────────────────────
Downloads and caches the embedding model used by RelevanceFilter.

Run this ONCE during Docker image build (or CI) so the model weights are
baked into the image layer.  At runtime the server loads from disk — zero
network calls, zero cold-start delay.

Usage
─────
  # In your Dockerfile (after pip install):
  RUN python scripts/preload_model.py

  # Locally (first-time setup or after changing MODEL_CACHE_DIR):
  MODEL_CACHE_DIR=./models python scripts/preload_model.py

  # Verify the cached files exist:
  python scripts/preload_model.py --check

Environment variables
─────────────────────
  EMBEDDING_MODEL    Model name / HF repo id
                     (default: sentence-transformers/all-MiniLM-L6-v2)
  MODEL_CACHE_DIR    Where to write the weights
                     (default: ~/.cache/torch/sentence_transformers)
  HF_TOKEN          HuggingFace token — raises the anonymous rate limit and
                     enables access to gated models.  Optional for this model.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_DIR  = os.getenv("MODEL_CACHE_DIR")


def _cache_dir_for_model(model_name: str, cache_root: str | None) -> str:
    """
    Return the directory sentence-transformers would use for this model.

    sentence-transformers replaces '/' with '_' in the directory name when
    using a HuggingFace repo id.
    """
    if cache_root is None:
        import torch  # noqa: PLC0415  (optional dep for path resolution)
        cache_root = os.path.join(torch.hub.get_dir(), "..", "sentence_transformers")

    safe_name = model_name.replace("/", "_").replace("\\", "_")
    return os.path.join(cache_root, safe_name)


def _model_appears_cached(model_name: str, cache_root: str | None) -> bool:
    """
    Heuristic check: the config.json and tokenizer_config.json exist in
    the expected cache directory.
    """
    model_dir = _cache_dir_for_model(model_name, cache_root)
    required  = ["config.json", "tokenizer_config.json"]
    return all(
        os.path.isfile(os.path.join(model_dir, f)) for f in required
    )


def download(model_name: str = MODEL_NAME, cache_dir: str | None = CACHE_DIR) -> None:
    """Download *model_name* and verify it encodes a test sentence."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
        logger.info("Cache dir: %s", cache_dir)
    else:
        logger.info("Cache dir: (default ~/.cache/torch/sentence_transformers)")

    logger.info("Downloading model: %s", model_name)
    t0 = time.perf_counter()

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)

    elapsed = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs — running smoke test …", elapsed)

    # Smoke test: encode a single sentence and check the output shape
    embedding = model.encode(["ResearchFlow embedding smoke test"])
    dim = len(embedding[0])
    logger.info("Smoke test passed — embedding dim=%d", dim)
    logger.info("✓ Model ready. Weights are cached and will be used at runtime.")


def check(model_name: str = MODEL_NAME, cache_dir: str | None = CACHE_DIR) -> None:
    """Check whether the model is already cached without downloading."""
    if _model_appears_cached(model_name, cache_dir):
        logger.info("✓ Model cache looks good: %s", model_name)
        sys.exit(0)
    else:
        logger.warning("✗ Model not cached — run without --check to download.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download the ResearchFlow embedding model.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name / HF repo id (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--cache-dir",
        default=CACHE_DIR,
        help="Cache directory (default: $MODEL_CACHE_DIR or HF default)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify the cache exists; don't download. Exit 0 if ok.",
    )
    args = parser.parse_args()

    if args.check:
        check(args.model, args.cache_dir)
    else:
        download(args.model, args.cache_dir)


if __name__ == "__main__":
    main()