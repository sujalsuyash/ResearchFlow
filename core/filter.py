"""
core/filter.py
──────────────
Four-layer relevance filter for ResearchFlow.

Sits between the Research Agent's raw retrieval output and the Synthesis
Chain. Every paper that enters here is scored; papers that fail the
minimum relevance bar are dropped before the synthesiser ever sees them.

Architecture
────────────
Layer 1 — Quality pre-checks  (pure Python, zero cost)
    Drops papers with no title+abstract, stub titles, raw blobs, or years
    before effective_min_year. Catches completely unparsable tool output
    immediately. The effective_min_year is auto-raised per field when the
    caller has not overridden min_year:
        CS / ML    → 2018  (field moves fast; pre-2018 work is rarely relevant)
        General    → 2015  (safe default for tech-adjacent topics)
        Biomedical → 1980  (clinical literature stays relevant for decades)

Layer 2 — Lexical overlap  (zero cost, instant)
    Tokenises the user query and each paper's title+abstract. Papers with
    insufficient shared content words between query tokens and paper tokens
    are dropped.

    lexical_min_overlap is a RATIO (0.0–1.0). It is converted to a required
    integer count ONCE before the loop:
        required_overlap = ceil(len(query_tokens) × lexical_min_overlap)
    At the default of 0.65, a 10-token query requires 7 shared words.
    A second-chance check against title+abstract combined requires one
    additional word beyond the primary threshold.

Layer 3 — Semantic similarity  (local model, no API calls)
    Embeds the query and each paper's title+abstract using
    sentence-transformers/all-MiniLM-L6-v2 (80 MB, CPU-friendly).
    Papers whose cosine similarity falls below effective_semantic_threshold
    are dropped.

    For broad multi-topic queries (detected via _is_broad_query()), the
    effective threshold is automatically raised by +0.10 (capped at 0.50)
    to compensate for the diffuse embedding space produced by wide queries.

Layer 4 — Citation-weighted reranking  (pure maths, zero cost)
    Survivors are re-scored by:
        final_score = semantic_sim × log(citation_count + 2) × recency_weight
    Recency weight decays exponentially with paper age; half-life is
    field-aware (CS: 4 yrs, biomedical: 10 yrs, general: 7 yrs).
    The top `max_papers` are returned in descending score order.

Graceful degradation
────────────────────
If sentence-transformers is not installed, Layer 3 is skipped with a
warning and only Layers 1, 2, and 4 run. Install with:
    pip install sentence-transformers

Usage
─────
    from core.filter import RelevanceFilter

    f = RelevanceFilter(max_papers=25, semantic_threshold=0.30)
    result = f.run(query="sepsis prediction ICU machine learning", papers=raw)
    clean_papers = result.papers
    print(result.summary())
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── Stop-words excluded from lexical overlap scoring ─────────────────────────
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "its", "it", "this",
    "that", "these", "those", "not", "no", "nor", "so", "yet", "both",
    "either", "neither", "such", "as", "than", "then", "when", "where",
    "which", "who", "whom", "how", "what", "why", "whether", "if", "use",
    "using", "used", "based", "novel", "new", "study", "paper", "approach",
    "method", "methods", "analysis", "review", "research", "towards",
    "toward", "via", "among", "across", "through", "during", "between",
    "into", "about", "after", "before", "within", "without", "over",
    "under", "above", "below", "two", "three", "four", "five",
})

# ── Prefixes emitted by the normaliser for completely unparsable blobs ────────
_RAW_BLOB_PREFIXES: tuple[str, ...] = ("Raw: ", "raw: ")

# ── Current year for recency decay ───────────────────────────────────────────
_CURRENT_YEAR: int = datetime.now().year

# ── Per-field recency half-lives (years) ─────────────────────────────────────
_FIELD_HALF_LIVES: dict[str, int] = {
    "biomedical": 10,
    "cs":          4,
    "general":     7,
}

# ── Auto min_year floors (Issue 2) ───────────────────────────────────────────
# Applied only when the caller has not overridden min_year (i.e. it equals
# DEFAULT_MIN_YEAR). Set an explicit min_year to bypass auto-detection.
DEFAULT_MIN_YEAR: int = 1980
_AUTO_MIN_YEAR: dict[str, int] = {
    "cs":          2018,   # pre-2018 ML/CS is rarely useful for current queries
    "general":     2015,   # safe tech-adjacent floor
    "biomedical":  1980,   # clinical lit stays relevant for decades — no floor raised
}

# ── Broad-query signals for auto threshold elevation (Issue 3) ────────────────
_BROAD_QUERY_SIGNALS: tuple[str, ...] = (
    "and the role", "and how", "impact of", "influence of",
    "role of", "relationship between", "comparison of",
    "effect of", "applications of", "use of",
)


# ──────────────────────────────────────────────────────────────────────────────
# FilterResult
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """
    Output of ``RelevanceFilter.run()``.

    Attributes
    ──────────
    papers : list[dict]
        Filtered and reranked papers, ready for the Synthesiser.
    dropped : int
        Total papers removed across all layers.
    kept : int
        len(papers)
    layer_stats : dict[str, int]
        Per-layer drop counts. Keys: "quality", "lexical", "semantic".
    elapsed_s : float
        Wall-clock seconds for the entire filter pass.
    semantic_available : bool
        Whether the sentence-transformers model was available (Layer 3).
    """

    papers:             list[dict[str, Any]]
    dropped:            int
    kept:               int
    layer_stats:        dict[str, int]  = field(default_factory=dict)
    elapsed_s:          float           = 0.0
    semantic_available: bool            = False

    def summary(self) -> str:
        sem = "on" if self.semantic_available else "OFF (pip install sentence-transformers)"
        return (
            f"Filter: {self.kept} kept, {self.dropped} dropped "
            f"[quality={self.layer_stats.get('quality', 0)} "
            f"lexical={self.layer_stats.get('lexical', 0)} "
            f"semantic={self.layer_stats.get('semantic', 0)}] "
            f"semantic_layer={sem} "
            f"({self.elapsed_s:.2f}s)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loader
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE: Any = None
_MODEL_UNAVAILABLE: bool = False


def _get_embedding_model() -> Any:
    """
    Return a cached SentenceTransformer, loading it on first call.
    Returns None if sentence-transformers is not installed.
    """
    global _MODEL_CACHE, _MODEL_UNAVAILABLE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if _MODEL_UNAVAILABLE:
        return None

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info(
            "RelevanceFilter: loading all-MiniLM-L6-v2 (one-time, ~80 MB)..."
        )
        t0 = time.perf_counter()
        _MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("RelevanceFilter: model ready in %.1fs", time.perf_counter() - t0)
        return _MODEL_CACHE

    except ImportError:
        logger.warning(
            "RelevanceFilter: sentence-transformers not installed — "
            "Layer 3 (semantic similarity) SKIPPED. "
            "Run: pip install sentence-transformers"
        )
        _MODEL_UNAVAILABLE = True
        return None

    except Exception as exc:
        logger.warning(
            "RelevanceFilter: model load failed (%s) — Layer 3 SKIPPED.", exc
        )
        _MODEL_UNAVAILABLE = True
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> frozenset[str]:
    """Lowercase, strip punctuation, remove stop-words. Min token length: 3."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def _paper_embed_text(paper: dict[str, Any]) -> str:
    """Build the string embedded for semantic scoring. Title is doubled to up-weight it."""
    title    = (paper.get("title")    or "").strip()
    abstract = (paper.get("abstract") or "").strip()[:600]
    return f"{title}. {title}. {abstract}"


def _recency_weight(year: int | None, half_life: int, min_weight: float) -> float:
    """
    Exponential decay from 1.0 (current year) toward min_weight.
    Formula: min_weight + (1 - min_weight) × 0.5^(age / half_life)
    """
    if year is None:
        return 0.75
    age = max(0, _CURRENT_YEAR - year)
    return min_weight + (1.0 - min_weight) * (0.5 ** (age / half_life))


def _rank_score(
    sim: float,
    citation_count: int | None,
    year: int | None,
    half_life: int,
    min_recency: float,
) -> float:
    """Combined relevance × authority × recency score used for Layer 4."""
    cit = math.log(max(0, citation_count or 0) + 2)
    rec = _recency_weight(year, half_life, min_recency)
    return sim * cit * rec


def _detect_field(query: str) -> str:
    """
    Lightweight domain detector.
    Returns 'biomedical', 'cs', or 'general'.
    Used to select recency half-life and effective min_year automatically.
    """
    q = query.lower()
    bio_hits = sum(1 for kw in (
        "drug", "clinical", "patient", "disease", "therapy", "cancer",
        "gene", "protein", "cell", "medical", "health", "virus",
        "vaccine", "genomic", "dna", "rna", "pharmacol", "pathogen",
        "mutation", "tumor", "tumour", "biomarker", "trial", "surgery",
        "diagnosis", "prognosis", "epidemic", "treatment", "hospital",
    ) if kw in q)

    cs_hits = sum(1 for kw in (
        "machine learning", "deep learning", "neural network",
        "transformer", "language model", "llm", "reinforcement learning",
        "computer vision", "nlp", "natural language", "artificial intelligence",
        "diffusion model", "retrieval", "graph network", "autoencoder",
        "attention mechanism", "fine-tuning", "embedding", "multimodal",
        "fusion", "classification", "dataset",
    ) if kw in q)

    if bio_hits > cs_hits:
        return "biomedical"
    if cs_hits > bio_hits:
        return "cs"
    return "general"


def _is_broad_query(query: str) -> bool:
    """
    Return True when the query spans multiple topics or domains.

    Broad queries produce diffuse embedding spaces — a topic like
    "AI in strategic decisions AND augmented reality in e-commerce"
    makes the query centroid land between both topics, so papers from
    either topic score lower than expected. Raising the semantic threshold
    compensates by requiring stronger per-paper relevance signal.
    """
    q = query.lower()
    return any(signal in q for signal in _BROAD_QUERY_SIGNALS)


# ──────────────────────────────────────────────────────────────────────────────
# RelevanceFilter
# ──────────────────────────────────────────────────────────────────────────────

class RelevanceFilter:
    """
    Four-layer relevance filter: quality → lexical → semantic → rerank.

    Parameters
    ──────────
    max_papers : int
        Maximum papers returned after Layer 4 reranking (default 25).
    semantic_threshold : float
        Cosine similarity floor for Layer 3 (default 0.30).
        For broad multi-topic queries this is automatically raised by +0.10
        (capped at 0.50) to compensate for diffuse query embeddings.
    lexical_min_overlap : float
        Fraction (0.0–1.0) of query content-tokens that must appear in the
        paper's title or title+abstract to pass Layer 2 (default 0.65).
        Converted to a required integer count at run time:
            required = ceil(len(query_tokens) × lexical_min_overlap)
        Set to 0.0 to disable lexical filtering entirely.
    min_year : int
        Papers published before this year are dropped (default 1980).
        When left at the default, the filter auto-raises the floor by field:
            CS / ML → 2018,  General → 2015,  Biomedical → 1980.
        Pass an explicit year to override auto-detection entirely.
    min_recency_weight : float
        Floor for the recency decay multiplier (default 0.40).
    auto_field : bool
        When True (default), detects domain from query and adjusts the
        recency half-life and min_year automatically.
    recency_half_life : int | None
        Manually override the half-life in years. Overrides auto_field.
    """

    def __init__(
        self,
        *,
        max_papers:           int        = 25,
        semantic_threshold:   float      = 0.30,
        lexical_min_overlap:  float      = 0.65,
        min_year:             int        = DEFAULT_MIN_YEAR,
        min_recency_weight:   float      = 0.40,
        auto_field:           bool       = True,
        recency_half_life:    int | None = None,
    ) -> None:
        self.max_papers           = max_papers
        self.semantic_threshold   = semantic_threshold
        self.lexical_min_overlap  = lexical_min_overlap
        self.min_year             = min_year
        self.min_recency_weight   = min_recency_weight
        self.auto_field           = auto_field
        self._override_half_life  = recency_half_life

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        papers: list[dict[str, Any]],
        sub_queries: list[str] | None = None,
    ) -> FilterResult:
        """
        Run the full filter pipeline synchronously.

        Parameters
        ──────────
        query : str
            The original user research question.
        papers : list[dict]
            Raw paper dicts from the Research Agent (post-dedup, pre-filter).

        Returns
        ───────
        FilterResult
            .papers     — filtered and reranked list (at most max_papers)
            .summary()  — one-line human-readable stats string
        """
        t0 = time.perf_counter()
        layer_stats: dict[str, int] = {"quality": 0, "lexical": 0, "semantic": 0}

        # ── Resolve field, half-life, effective_min_year ──────────────────────
        field_name = _detect_field(query) if self.auto_field else "general"
        half_life  = self._override_half_life or _FIELD_HALF_LIVES.get(field_name, 7)

        # Issue 2: auto-raise min_year when caller left it at the default value
        if self.min_year == DEFAULT_MIN_YEAR and self.auto_field:
            effective_min_year = _AUTO_MIN_YEAR.get(field_name, DEFAULT_MIN_YEAR)
        else:
            effective_min_year = self.min_year   # caller explicitly overrode it

        # Issue 3: raise semantic threshold for broad multi-topic queries
        if _is_broad_query(query):
            effective_threshold = min(self.semantic_threshold + 0.10, 0.50)
            logger.debug(
                "RelevanceFilter: broad query detected — semantic threshold "
                "raised %.2f → %.2f",
                self.semantic_threshold, effective_threshold,
            )
        else:
            effective_threshold = self.semantic_threshold

        logger.debug(
            "RelevanceFilter: field=%s half_life=%d effective_min_year=%d "
            "effective_threshold=%.2f",
            field_name, half_life, effective_min_year, effective_threshold,
        )

        # ── Layer 1: Quality pre-checks ───────────────────────────────────────
        after_quality: list[dict[str, Any]] = []
        for p in papers:
            title    = (p.get("title")    or "").strip()
            abstract = (p.get("abstract") or "").strip()

            # Drop raw blobs from the normaliser
            if any(title.startswith(pfx) for pfx in _RAW_BLOB_PREFIXES):
                layer_stats["quality"] += 1
                continue

            # Must have a title of substance OR a non-trivial abstract
            if len(title) < 10 and len(abstract) < 30:
                layer_stats["quality"] += 1
                continue

            # Year floor — uses effective_min_year, not raw self.min_year
            year = p.get("year")
            if year is not None:
                try:
                    if int(year) < effective_min_year:
                        layer_stats["quality"] += 1
                        logger.debug(
                            "Filter L1 drop (year %s < %d): %r",
                            year, effective_min_year, title[:60],
                        )
                        continue
                except (TypeError, ValueError):
                    pass

            after_quality.append(p)

        logger.debug(
            "Filter L1 (quality): %d -> %d (%d dropped)",
            len(papers), len(after_quality), layer_stats["quality"],
        )

# ── Layer 2: Lexical overlap ──────────────────────────────────────────
        # Expand vocabulary using the planner's corrected sub-queries so papers
        # are matched against field-standard terms, not just the user's raw input.
        # IMPORTANT: required_overlap is calculated from base_token_count only —
        # adding sub_queries must not inflate the required overlap threshold.
        all_query_text  = " ".join(filter(None, [query] + (sub_queries or [])))
        query_tokens    = _tokenise(all_query_text)
        base_token_count = len(_tokenise(query))

        after_lexical: list[dict[str, Any]] = []

        if self.lexical_min_overlap <= 0 or not query_tokens:
            after_lexical = after_quality
        else:
            required_overlap = max(1, math.ceil(
                base_token_count * self.lexical_min_overlap
            ))
            logger.debug(
                "Filter L2: query_tokens=%d base_tokens=%d required_overlap=%d (ratio=%.2f)",
                len(query_tokens), base_token_count, required_overlap, self.lexical_min_overlap,
            )

            for p in after_quality:
                title_tokens    = _tokenise(p.get("title")    or "")
                abstract_tokens = _tokenise(p.get("abstract") or "")

                # Primary check: title tokens alone
                if len(query_tokens & title_tokens) >= required_overlap:
                    after_lexical.append(p)
                    continue

                # Second chance: title + abstract combined (one extra word required)
                combined = title_tokens | abstract_tokens
                if len(query_tokens & combined) >= required_overlap + 1:
                    after_lexical.append(p)
                else:
                    layer_stats["lexical"] += 1
                    logger.debug(
                        "Filter L2 drop (lexical): %r",
                        (p.get("title") or "")[:60],
                    )

        logger.debug(
            "Filter L2 (lexical): %d -> %d (%d dropped)",
            len(after_quality), len(after_lexical), layer_stats["lexical"],
        )

        # ── Layer 3: Semantic similarity ──────────────────────────────────────
        model = _get_embedding_model()
        semantic_available = model is not None
        after_semantic: list[tuple[float, dict[str, Any]]] = []

        if model is None or not after_lexical:
            after_semantic = [(0.5, p) for p in after_lexical]
        else:
            try:
                corpus = [_paper_embed_text(p) for p in after_lexical]
                q_emb  = model.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True
                )
                p_embs = model.encode(
                    corpus, convert_to_numpy=True, normalize_embeddings=True
                )
                sims: list[float] = (p_embs @ q_emb.T).flatten().tolist()

                for sim, paper in zip(sims, after_lexical):
                    # Uses effective_threshold (auto-raised for broad queries)
                    if sim >= effective_threshold:
                        after_semantic.append((sim, paper))
                    else:
                        layer_stats["semantic"] += 1
                        logger.debug(
                            "Filter L3 drop (semantic=%.3f < %.3f): %r",
                            sim, effective_threshold,
                            (paper.get("title") or "")[:60],
                        )

            except Exception as exc:
                logger.warning(
                    "RelevanceFilter: semantic scoring failed (%s) — "
                    "keeping all lexical survivors.", exc
                )
                after_semantic = [(0.5, p) for p in after_lexical]

        logger.debug(
            "Filter L3 (semantic): %d -> %d (%d dropped)",
            len(after_lexical), len(after_semantic), layer_stats["semantic"],
        )

        # ── Layer 4: Citation-weighted reranking ──────────────────────────────
        ranked = sorted(
            after_semantic,
            key=lambda t: _rank_score(
                sim            = t[0],
                citation_count = t[1].get("citation_count"),
                year           = t[1].get("year"),
                half_life      = half_life,
                min_recency    = self.min_recency_weight,
            ),
            reverse=True,
        )

        final_papers  = [p for _, p in ranked[: self.max_papers]]
        total_dropped = len(papers) - len(final_papers)
        elapsed       = time.perf_counter() - t0

        result = FilterResult(
            papers             = final_papers,
            dropped            = total_dropped,
            kept               = len(final_papers),
            layer_stats        = layer_stats,
            elapsed_s          = elapsed,
            semantic_available = semantic_available,
        )
        logger.info("RelevanceFilter: %s", result.summary())
        return result

def preload_embedding_model() -> None:
        _get_embedding_model()