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
    before effective_min_year. The effective_min_year is auto-raised per
    field when the caller has not overridden min_year:
        CS / ML    → 2018  (field moves fast; pre-2018 work is rarely relevant)
        General    → 2015  (safe default for tech-adjacent topics)
        Biomedical → 1980  (clinical literature stays relevant for decades)

Layer 2 — Semantic similarity  (local model, no API calls)
    Embeds the query and each paper's title+abstract using
    sentence-transformers/all-MiniLM-L6-v2 (80 MB, CPU-friendly).
    Papers whose cosine similarity falls below effective_semantic_threshold
    are dropped. Runs BEFORE lexical so the embedding model judges all
    quality-passed papers, not a pre-screened subset.

    Domain detection uses the same embedding model — the query is compared
    against three prototype sentences (cs / biomedical / general) by cosine
    similarity. No keyword arrays. When the top two domain scores are within
    0.05 of each other the query is treated as cross-domain and the semantic
    threshold is auto-raised by +0.10 (capped at 0.50).

Layer 3 — Lexical safety net  (zero cost, instant)
    Hard-drops only papers with zero content-word overlap with the query
    (expanded with planner sub_queries). This is a last-resort noise guard,
    not a primary relevance gate.

Layer 4 — Citation-weighted reranking  (pure maths, zero cost)
    Survivors are re-scored by:
        final_score = semantic_sim
                      × log(citation_count + 2)
                      × recency_weight
                      × (1 + lexical_boost)
    lexical_boost adds up to +0.20 for papers with strong token overlap
    without hard-dropping papers that score well semantically but use
    different vocabulary. Recency weight decays exponentially; half-life
    is field-aware (CS: 4 yrs, biomedical: 10 yrs, general: 7 yrs).
    The top `max_papers` are returned in descending score order.

Graceful degradation
────────────────────
If sentence-transformers is not installed, Layer 2 is skipped with a
warning and only Layers 1, 3, and 4 run. Domain detection falls back to
"general" when the model is unavailable.

Public API
──────────
    from core.filter import RelevanceFilter, detect_domain_semantic

    # Domain detection (shared with research_agent.py)
    domain = detect_domain_semantic("CRISPR gene editing cancer therapy")
    # → "biomedical"

    # Full filter pipeline
    f = RelevanceFilter(max_papers=25, semantic_threshold=0.30)
    result = f.run(
        query="sepsis prediction ICU machine learning",
        papers=raw,
        sub_queries=["ICU mortality prediction", "sepsis early warning"],
    )
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
# Deliberately excludes domain-significant terms like "analysis",
# "classification", "detection" — these carry real meaning in academic
# queries and must not be silently stripped.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "its", "it", "this",
    "that", "these", "those", "not", "no", "nor", "so", "yet", "both",
    "either", "neither", "such", "as", "than", "then", "when", "where",
    "which", "who", "whom", "how", "what", "why", "whether", "if",
    "novel", "new", "paper", "towards", "toward", "via", "among",
    "across", "through", "during", "between", "into", "about", "after",
    "before", "within", "without", "over", "under", "above", "below",
    "two", "three", "four", "five",
})

# ── Prefixes emitted by the normaliser for completely unparsable blobs ────────
_RAW_BLOB_PREFIXES: tuple[str, ...] = ("Raw: ", "raw: ")

# ── Current year for recency decay ───────────────────────────────────────────
_CURRENT_YEAR: int = datetime.now().year

# ── Per-field recency half-lives (years) ─────────────────────────────────────
# These are stable real-world constants — not vocabulary lists.
# CS literature goes stale in ~4 years; clinical literature stays relevant
# for decades. These values won't need updating as fields evolve.
_FIELD_HALF_LIVES: dict[str, int] = {
    "biomedical": 10,
    "cs":          4,
    "general":     7,
}

# ── Auto min_year floors ──────────────────────────────────────────────────────
# Same reasoning: these describe time constants, not vocabulary.
DEFAULT_MIN_YEAR: int = 1980
_AUTO_MIN_YEAR: dict[str, int] = {
    "cs":          2015,
    "biomedical":  1980,
    "general":     2012,
}

# ── Domain prototype sentences for semantic domain detection ──────────────────
# Three short descriptive sentences — one per domain. The embedding model
# compares the query against all three and picks the closest by cosine
# similarity. No keyword arrays needed; the model handles vocabulary
# variation, synonyms, and cross-domain queries automatically.
#
# Updating: only needed if you add a new domain (e.g. "legal", "physics").
# Rephrasing individual sentences rarely improves results — the model is
# robust to wording variation at this level of abstraction.
_DOMAIN_PROTOTYPES: dict[str, str] = {
    "cs": (
        "machine learning deep learning neural networks artificial intelligence "
        "computer vision natural language processing transformers reinforcement "
        "learning algorithms data science software systems"
    ),
    "biomedical": (
        "clinical trials patient health disease treatment drug therapy cancer "
        "genomics medical diagnosis surgery hospital epidemiology vaccine "
        "pharmacology biomarker protein cell biology"
    ),
    "general": (
        "social science economics history policy humanities education law "
        "philosophy literature psychology sociology political science "
        "environmental science physics chemistry engineering"
    ),
}

# ── Cross-domain detection threshold ─────────────────────────────────────────
# When the top two domain similarity scores are within this gap, the query
# spans multiple domains and gets full tool fan-out + raised semantic threshold.
_CROSS_DOMAIN_GAP: float = 0.08

# ── Maximum lexical boost added to Layer 4 score ─────────────────────────────
_MAX_LEXICAL_BOOST: float = 0.20


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
        Per-layer drop counts. Keys: "quality", "semantic", "lexical".
    elapsed_s : float
        Wall-clock seconds for the entire filter pass.
    semantic_available : bool
        Whether the sentence-transformers model was available (Layer 2).
    detected_domain : str
        Domain inferred from the query: "cs", "biomedical", or "general".
    """

    papers:             list[dict[str, Any]]
    dropped:            int
    kept:               int
    layer_stats:        dict[str, int]  = field(default_factory=dict)
    elapsed_s:          float           = 0.0
    semantic_available: bool            = False
    detected_domain:    str             = "general"

    def summary(self) -> str:
        sem = "on" if self.semantic_available else "OFF (pip install sentence-transformers)"
        return (
            f"Filter: {self.kept} kept, {self.dropped} dropped "
            f"[quality={self.layer_stats.get('quality', 0)} "
            f"semantic={self.layer_stats.get('semantic', 0)} "
            f"lexical={self.layer_stats.get('lexical', 0)}] "
            f"domain={self.detected_domain} "
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
            "Layer 2 (semantic similarity) and domain detection SKIPPED. "
            "Run: pip install sentence-transformers"
        )
        _MODEL_UNAVAILABLE = True
        return None

    except Exception as exc:
        logger.warning(
            "RelevanceFilter: model load failed (%s) — Layer 2 SKIPPED.", exc
        )
        _MODEL_UNAVAILABLE = True
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Semantic domain detection  (public — imported by research_agent.py)
# ──────────────────────────────────────────────────────────────────────────────

def detect_domain_semantic(query: str) -> str:
    """
    Classify *query* into "cs", "biomedical", or "general" using the
    embedding model — no keyword arrays.

    The query embedding is compared against three domain prototype sentences
    by cosine similarity. When the top two domain scores are within
    _CROSS_DOMAIN_GAP (0.05) of each other, the query spans multiple domains
    and "general" is returned — which triggers full four-tool fan-out in the
    Research Agent and a raised semantic threshold in the filter.

    Falls back to "general" if the embedding model is unavailable.

    Returns
    ───────
    "cs"         — primarily computer science / ML / AI
    "biomedical" — primarily clinical / life sciences
    "general"    — cross-domain or neither of the above
    """
    model = _get_embedding_model()
    if model is None:
        logger.debug("detect_domain_semantic: model unavailable — returning 'general'")
        return "general"

    try:
        domain_names  = list(_DOMAIN_PROTOTYPES.keys())
        prototype_texts = list(_DOMAIN_PROTOTYPES.values())

        q_emb = model.encode([query], normalize_embeddings=True)
        p_emb = model.encode(prototype_texts, normalize_embeddings=True)

        sims = (p_emb @ q_emb.T).flatten().tolist()

        # Sort domain indices by similarity descending
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        best_domain = domain_names[ranked[0]]
        best_score  = sims[ranked[0]]
        second_score = sims[ranked[1]] if len(ranked) > 1 else 0.0

        gap = best_score - second_score

        logger.debug(
            "detect_domain_semantic: query=%r best=%s(%.3f) second=%s(%.3f) gap=%.3f",
            query[:60], best_domain, best_score,
            domain_names[ranked[1]] if len(ranked) > 1 else "n/a",
            second_score, gap,
        )

        if gap < _CROSS_DOMAIN_GAP:
            logger.debug(
                "detect_domain_semantic: gap %.3f < %.3f → cross-domain → 'general'",
                gap, _CROSS_DOMAIN_GAP,
            )
            return "general"

        return best_domain

    except Exception as exc:
        logger.warning("detect_domain_semantic: failed (%s) — returning 'general'", exc)
        return "general"

def detect_domain_semantic_multi(queries: list[str]) -> str:
    """
    Classify domain from a list of planner-generated search queries rather
    than the raw user query. Aggregates cosine similarity scores across all
    queries and picks by averaged signal.

    This is more reliable than classifying the raw user query because the
    planner decomposes vague umbrella terms ("artificial intelligence") into
    specific sub-queries whose vocabulary maps more cleanly to a domain.

    Falls back to "general" if the model is unavailable or queries is empty.
    """
    if not queries:
        return "general"

    model = _get_embedding_model()
    if model is None:
        return "general"

    try:
        domain_names    = list(_DOMAIN_PROTOTYPES.keys())
        prototype_texts = list(_DOMAIN_PROTOTYPES.values())

        p_emb  = model.encode(prototype_texts, normalize_embeddings=True)
        scores = [0.0] * len(domain_names)

        for q in queries:
            q_emb = model.encode([q], normalize_embeddings=True)
            sims  = (p_emb @ q_emb.T).flatten().tolist()
            for i, s in enumerate(sims):
                scores[i] += s

        # Average across all sub-queries
        scores = [s / len(queries) for s in scores]

        ranked      = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best_domain = domain_names[ranked[0]]
        gap         = scores[ranked[0]] - scores[ranked[1]]

        logger.debug(
            "detect_domain_semantic_multi: queries=%d best=%s gap=%.3f scores=%s",
            len(queries), best_domain, gap,
            {domain_names[i]: f"{scores[i]:.3f}" for i in range(len(scores))},
        )

        if gap < _CROSS_DOMAIN_GAP:
            logger.debug(
                "detect_domain_semantic_multi: gap %.3f < %.3f → cross-domain → 'general'",
                gap, _CROSS_DOMAIN_GAP,
            )
            return "general"

        return best_domain

    except Exception as exc:
        logger.warning(
            "detect_domain_semantic_multi: failed (%s) — returning 'general'", exc
        )
        return "general"


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> frozenset[str]:
    """Lowercase, strip punctuation, remove stop-words. Min token length: 3."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def _paper_embed_text(paper: dict[str, Any]) -> str:
    """Build the string embedded for semantic scoring. Title doubled to up-weight it."""
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


def _lexical_boost(
    query_tokens: frozenset[str],
    paper: dict[str, Any],
    max_boost: float = _MAX_LEXICAL_BOOST,
) -> float:
    """
    Returns a [0, max_boost] additive score bonus based on token overlap.
    Used in Layer 4 reranking — not as a hard filter gate.
    """
    title_tokens    = _tokenise(paper.get("title")    or "")
    abstract_tokens = _tokenise(paper.get("abstract") or "")
    combined        = title_tokens | abstract_tokens

    if not query_tokens or not combined:
        return 0.0

    overlap_ratio = len(query_tokens & combined) / len(query_tokens)
    return min(overlap_ratio, 1.0) * max_boost


def _rank_score(
    sim: float,
    citation_count: int | None,
    year: int | None,
    half_life: int,
    min_recency: float,
    lex_boost: float,
) -> float:
    """Combined relevance × authority × recency × lexical score for Layer 4."""
    cit = math.log(max(0, citation_count or 0) + 2)
    rec = _recency_weight(year, half_life, min_recency)
    return sim * cit * rec * (1.0 + lex_boost)


# ──────────────────────────────────────────────────────────────────────────────
# RelevanceFilter
# ──────────────────────────────────────────────────────────────────────────────

class RelevanceFilter:
    """
    Four-layer relevance filter: quality → semantic → lexical → rerank.

    Domain classification uses the embedding model (no keyword arrays).
    When a query spans multiple domains the semantic threshold is
    automatically raised to compensate for diffuse query embeddings.

    Parameters
    ──────────
    max_papers : int
        Maximum papers returned after Layer 4 reranking (default 25).
    semantic_threshold : float
        Cosine similarity floor for Layer 2 (default 0.30).
        Auto-raised by +0.10 (capped at 0.50) for cross-domain queries.
    min_year : int
        Papers before this year are dropped (default 1980).
        Auto-raised by field: CS/ML → 2018, General → 2015, Biomedical → 1980.
        Pass an explicit year to override auto-detection.
    min_recency_weight : float
        Floor for the recency decay multiplier (default 0.40).
    auto_field : bool
        When True (default), uses semantic domain detection to adjust
        recency half-life and min_year automatically.
    recency_half_life : int | None
        Manually override the half-life in years.
    """

    def __init__(
        self,
        *,
        max_papers:           int        = 25,
        semantic_threshold:   float      = 0.30,
        min_year:             int        = DEFAULT_MIN_YEAR,
        min_recency_weight:   float      = 0.40,
        auto_field:           bool       = True,
        recency_half_life:    int | None = None,
    ) -> None:
        self.max_papers          = max_papers
        self.semantic_threshold  = semantic_threshold
        self.min_year            = min_year
        self.min_recency_weight  = min_recency_weight
        self.auto_field          = auto_field
        self._override_half_life = recency_half_life

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
        sub_queries : list[str] | None
            Planner-generated sub-questions. Used to expand the lexical
            vocabulary. Always pass these from the pipeline for best results.

        Returns
        ───────
        FilterResult
            .papers          — filtered and reranked list (at most max_papers)
            .detected_domain — domain inferred from query
            .summary()       — one-line human-readable stats string
        """
        t0 = time.perf_counter()
        layer_stats: dict[str, int] = {"quality": 0, "semantic": 0, "lexical": 0}

        # ── Domain detection via embedding model (no keyword arrays) ──────────
        # detect_domain_semantic returns "general" for cross-domain queries,
        # which triggers full fan-out AND a raised semantic threshold below.
        if self.auto_field:
            if sub_queries:
                field_name = detect_domain_semantic_multi(sub_queries)
            else:
                field_name = detect_domain_semantic(query)
        else:
            field_name = "general"

        half_life  = self._override_half_life or _FIELD_HALF_LIVES.get(field_name, 7)

        if self.min_year == DEFAULT_MIN_YEAR and self.auto_field:
            effective_min_year = _AUTO_MIN_YEAR.get(field_name, DEFAULT_MIN_YEAR)
        else:
            effective_min_year = self.min_year

        # Cross-domain queries ("general") produce diffuse embeddings — raise
        # the threshold so we require stronger per-paper relevance signal.
        if field_name == "general":
            effective_threshold = min(self.semantic_threshold + 0.10, 0.50)
            logger.debug(
                "RelevanceFilter: cross-domain query — semantic threshold "
                "raised %.2f → %.2f",
                self.semantic_threshold, effective_threshold,
            )
        else:
            effective_threshold = self.semantic_threshold

        logger.debug(
            "RelevanceFilter: domain=%s half_life=%d effective_min_year=%d "
            "effective_threshold=%.2f",
            field_name, half_life, effective_min_year, effective_threshold,
        )

        # ── Layer 1: Quality pre-checks ───────────────────────────────────────
        after_quality: list[dict[str, Any]] = []
        for p in papers:
            title    = (p.get("title")    or "").strip()
            abstract = (p.get("abstract") or "").strip()

            if any(title.startswith(pfx) for pfx in _RAW_BLOB_PREFIXES):
                layer_stats["quality"] += 1
                continue

            if len(title) < 10 and len(abstract) < 30:
                layer_stats["quality"] += 1
                continue

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

        # ── Layer 2: Semantic similarity ──────────────────────────────────────
        # Runs on ALL after_quality papers — not a lexically pre-screened
        # subset. The embedding model is already loaded for domain detection
        # so this costs no additional startup time.
        model = _get_embedding_model()
        semantic_available = model is not None

        after_semantic: list[tuple[float, dict[str, Any]]] = []

        if model is None or not after_quality:
            after_semantic = [(0.5, p) for p in after_quality]
        else:
            try:
                corpus = [_paper_embed_text(p) for p in after_quality]
                scoring_text = " ".join(sub_queries) if sub_queries else query
                q_emb  = model.encode(
                    [scoring_text], convert_to_numpy=True, normalize_embeddings=True
                )
                p_embs = model.encode(
                    corpus, convert_to_numpy=True, normalize_embeddings=True
                )
                sims: list[float] = (p_embs @ q_emb.T).flatten().tolist()

                for sim, paper in zip(sims, after_quality):
                    if sim >= effective_threshold:
                        after_semantic.append((sim, paper))
                    else:
                        layer_stats["semantic"] += 1
                        logger.debug(
                            "Filter L2 drop (semantic=%.3f < %.3f): %r",
                            sim, effective_threshold,
                            (paper.get("title") or "")[:60],
                        )

            except Exception as exc:
                logger.warning(
                    "RelevanceFilter: semantic scoring failed (%s) — "
                    "keeping all quality survivors.", exc
                )
                after_semantic = [(0.5, p) for p in after_quality]

        logger.debug(
            "Filter L2 (semantic): %d -> %d (%d dropped)",
            len(after_quality), len(after_semantic), layer_stats["semantic"],
        )

        # ── Layer 3: Lexical safety net ───────────────────────────────────────
        # Hard-drops only papers with ZERO content-word overlap with the query.
        # Vocabulary expanded with sub_queries so field-standard terms count
        # even when absent from the user's raw input. Required overlap = 1.
        all_query_text = " ".join(filter(None, [query] + (sub_queries or [])))
        query_tokens   = _tokenise(all_query_text)

        after_lexical: list[tuple[float, dict[str, Any]]] = []

        if not query_tokens:
            after_lexical = after_semantic
        else:
            for sim, paper in after_semantic:
                title_tokens    = _tokenise(paper.get("title")    or "")
                abstract_tokens = _tokenise(paper.get("abstract") or "")
                combined        = title_tokens | abstract_tokens

                if len(query_tokens & combined) >= 1:
                    after_lexical.append((sim, paper))
                else:
                    layer_stats["lexical"] += 1
                    logger.debug(
                        "Filter L3 drop (zero lexical overlap): %r",
                        (paper.get("title") or "")[:60],
                    )

        logger.debug(
            "Filter L3 (lexical): %d -> %d (%d dropped)",
            len(after_semantic), len(after_lexical), layer_stats["lexical"],
        )

        # ── Layer 4: Citation-weighted reranking with lexical boost ───────────
        ranked = sorted(
            after_lexical,
            key=lambda t: _rank_score(
                sim            = t[0],
                citation_count = t[1].get("citation_count"),
                year           = t[1].get("year"),
                half_life      = half_life,
                min_recency    = self.min_recency_weight,
                lex_boost      = _lexical_boost(query_tokens, t[1]),
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
            detected_domain    = field_name,
        )
        logger.info("RelevanceFilter: %s", result.summary())
        return result


def preload_embedding_model() -> None:
    """Call at server startup to warm the embedding model singleton."""
    _get_embedding_model()