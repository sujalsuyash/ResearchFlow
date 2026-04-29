"""
tests/test_openalex_search.py

Pytest suite for tools/openalex_search.py.

Approach
--------
All HTTP I/O is mocked so the suite runs fully offline without consuming
OpenAlex rate-limit quota.

For the majority of tests (happy path, caching, schema, tool output) we patch
tools.openalex_search._throttled_get — one clean mock per call, no HTTP
plumbing involved.

For retry-logic tests we go one level deeper and patch httpx.Client directly,
letting the real _throttled_get loop execute so we can assert it actually
retried the configured number of times.

Coverage targets
----------------
* _reconstruct_abstract  — real inverted index, multi-position words, empty,
                           None, malformed input
* _extract_doi           — https://doi.org/… prefix stripping, already bare,
                           None
* _extract_arxiv_id      — https://arxiv.org/abs/… prefix stripping, already
                           bare, None
* _build_filter          — no filters, year_start only, year_end only, both,
                           concept only, all three, whitespace trimming
* _build_params          — mailto present, select present, filter injected only
                           when non-None
* _parse_work            — full record, no abstract, no doi, no arxiv, no
                           primary_location, URL priority ladder, multi-author
                           truncation
* search_openalex        — happy path, limit trimming, zero results, fallback
                           cache hit, Redis cache hit + miss, Redis GET failure
                           falls through, tool= / email= params present
* _throttled_get retry   — 429 × 2 then success (httpx.Client mock), call
                           count == 3, Retry-After header honoured
* _tool_fn / openalex_tool — Markdown shape, header, per-paper fields, et al.,
                              zero-result string, LangChain invoke, tool name
                              and description
* OpenAlexInput schema   — defaults, limit bounds, required query
* clear_cache            — fallback dict emptied, Redis SCAN+DELETE called,
                           Redis error does not raise
* cache_stats            — fallback count + keys, Redis entries + prefix,
                           Redis error returns error dict
* _cache_key             — determinism, sensitivity to each argument, prefix,
                           no whitespace in key
"""

from __future__ import annotations

import sys
import os

# Make `import tools.openalex_search` resolve from the repo root regardless of
# how pytest is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared patch target
# ---------------------------------------------------------------------------

PATCH_THROTTLED_GET = "tools.openalex_search._throttled_get"

# ---------------------------------------------------------------------------
# Realistic API fixtures
# ---------------------------------------------------------------------------

# ---- works ----
# Work 1: fully populated — has doi, arxiv, pdf_url, 4 authors, real abstract
_WORK_1 = {
    "id": "https://openalex.org/W1111111111",
    "title": "Attention Is All You Need",
    "display_name": "Attention Is All You Need",
    "publication_year": 2017,
    "cited_by_count": 98765,
    "doi": "https://doi.org/10.48550/arXiv.1706.03762",
    "abstract_inverted_index": {
        "The": [0],
        "dominant": [1],
        "sequence": [2],
        "transduction": [3],
        "models": [4],
        "are": [5],
        "based": [6],
        "on": [7],
        "complex": [8],
        "recurrent": [9],
    },
    "authorships": [
        {"author": {"display_name": "Ashish Vaswani",  "id": "A1"}},
        {"author": {"display_name": "Noam Shazeer",    "id": "A2"}},
        {"author": {"display_name": "Niki Parmar",     "id": "A3"}},
        {"author": {"display_name": "Jakob Uszkoreit", "id": "A4"}},
    ],
    "ids": {
        "openalex": "https://openalex.org/W1111111111",
        "doi":      "https://doi.org/10.48550/arXiv.1706.03762",
        "arxiv":    "https://arxiv.org/abs/1706.03762",
    },
    "primary_location": {
        "pdf_url":          "https://arxiv.org/pdf/1706.03762",
        "landing_page_url": "https://arxiv.org/abs/1706.03762",
    },
}

# Work 2: doi present, no arxiv in ids, no pdf_url (landing page only), 2 authors
_WORK_2 = {
    "id": "https://openalex.org/W2222222222",
    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "display_name": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "publication_year": 2019,
    "cited_by_count": 75000,
    "doi": "https://doi.org/10.18653/v1/N19-1423",
    "abstract_inverted_index": {
        "We": [0],
        "introduce": [1],
        "a": [2, 6],
        "new": [3],
        "language": [4],
        "representation": [5],
        "model": [7],
        "called": [8],
        "BERT.": [9],
    },
    "authorships": [
        {"author": {"display_name": "Jacob Devlin",    "id": "A5"}},
        {"author": {"display_name": "Ming-Wei Chang",  "id": "A6"}},
    ],
    "ids": {
        "openalex": "https://openalex.org/W2222222222",
        "doi":      "https://doi.org/10.18653/v1/N19-1423",
    },
    "primary_location": {
        "pdf_url":          None,
        "landing_page_url": "https://aclanthology.org/N19-1423",
    },
}

# Work 3: no doi on record, arxiv in ids, no abstract, no primary_location
_WORK_3 = {
    "id": "https://openalex.org/W3333333333",
    "title": "Minimal Paper With No Abstract",
    "display_name": "Minimal Paper With No Abstract",
    "publication_year": 2024,
    "cited_by_count": 0,
    "doi": None,
    "abstract_inverted_index": None,
    "authorships": [
        {"author": {"display_name": "Anonymous Author", "id": "A7"}},
    ],
    "ids": {
        "openalex": "https://openalex.org/W3333333333",
        "arxiv":    "https://arxiv.org/abs/2401.00001",
    },
    "primary_location": None,
}

# Full 3-result response
OPENALEX_JSON_3 = json.dumps({
    "meta":    {"count": 3, "page": 1, "per_page": 5, "db_response_time_ms": 45},
    "results": [_WORK_1, _WORK_2, _WORK_3],
})

# Zero-result response
OPENALEX_JSON_0 = json.dumps({
    "meta":    {"count": 0, "page": 1, "per_page": 5, "db_response_time_ms": 10},
    "results": [],
})

# ---------------------------------------------------------------------------
# Mock-response helpers
# ---------------------------------------------------------------------------

def _mock_response(body: str, status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code  = status_code
    r.text         = body
    r.json.return_value = json.loads(body)
    r.raise_for_status  = MagicMock()
    return r


def _429_error(retry_after: str | None = None) -> "httpx.HTTPStatusError":
    """Return a realistic HTTPStatusError with status 429."""
    import httpx
    req  = MagicMock()
    resp = MagicMock()
    resp.status_code = 429
    resp.headers     = {"Retry-After": retry_after} if retry_after else {}
    return httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)


# ===========================================================================
# 1. _reconstruct_abstract
# ===========================================================================

class TestReconstructAbstract:
    def setup_method(self):
        from tools.openalex_search import _reconstruct_abstract
        self.fn = _reconstruct_abstract

    def test_none_returns_placeholder(self):
        assert self.fn(None) == "No abstract available."

    def test_empty_dict_returns_placeholder(self):
        assert self.fn({}) == "No abstract available."

    def test_simple_reconstruction(self):
        idx = {"Hello": [0], "world": [1]}
        assert self.fn(idx) == "Hello world"

    def test_word_order_by_position_not_insertion_order(self):
        # "world" is at position 0, "Hello" at position 1
        idx = {"Hello": [1], "world": [0]}
        assert self.fn(idx) == "world Hello"

    def test_word_appearing_at_multiple_positions(self):
        # "a" appears at positions 2 and 6
        idx = {"We": [0], "introduce": [1], "a": [2, 6], "new": [3],
               "language": [4], "model": [5], "called": [7], "BERT.": [8]}
        result = self.fn(idx)
        words = result.split()
        assert words[2] == "a"
        assert words[6] == "a"
        assert words[0] == "We"

    def test_real_inverted_index_from_fixture(self):
        idx = _WORK_1["abstract_inverted_index"]
        result = self.fn(idx)
        assert result.startswith("The dominant sequence transduction")

    def test_single_word_abstract(self):
        assert self.fn({"Preprint": [0]}) == "Preprint"

    def test_non_contiguous_positions_padded_correctly(self):
        # gaps in positions are fine — we only join the words we have
        idx = {"first": [0], "third": [2]}
        result = self.fn(idx)
        assert result == "first third"


# ===========================================================================
# 2. _extract_doi
# ===========================================================================

class TestExtractDoi:
    def setup_method(self):
        from tools.openalex_search import _extract_doi
        self.fn = _extract_doi

    def test_none_returns_none(self):
        assert self.fn(None) is None

    def test_empty_string_returns_none(self):
        assert self.fn("") is None

    def test_strips_https_doi_org_prefix(self):
        assert self.fn("https://doi.org/10.1038/s41586-023-05543-x") == "10.1038/s41586-023-05543-x"

    def test_strips_http_doi_org_prefix(self):
        assert self.fn("http://doi.org/10.1038/nature") == "10.1038/nature"

    def test_bare_doi_returned_unchanged(self):
        assert self.fn("10.48550/arXiv.1706.03762") == "10.48550/arXiv.1706.03762"

    def test_arxiv_doi_handled(self):
        result = self.fn("https://doi.org/10.48550/arXiv.1706.03762")
        assert result == "10.48550/arXiv.1706.03762"


# ===========================================================================
# 3. _extract_arxiv_id
# ===========================================================================

class TestExtractArxivId:
    def setup_method(self):
        from tools.openalex_search import _extract_arxiv_id
        self.fn = _extract_arxiv_id

    def test_none_returns_none(self):
        assert self.fn(None) is None

    def test_empty_string_returns_none(self):
        assert self.fn("") is None

    def test_strips_arxiv_abs_prefix(self):
        assert self.fn("https://arxiv.org/abs/1706.03762") == "1706.03762"

    def test_strips_versioned_arxiv_id(self):
        assert self.fn("https://arxiv.org/abs/2301.12345v2") == "2301.12345v2"

    def test_bare_id_returned_unchanged(self):
        assert self.fn("2301.12345") == "2301.12345"

    def test_new_format_id(self):
        assert self.fn("https://arxiv.org/abs/2401.00001") == "2401.00001"


# ===========================================================================
# 4. _build_filter
# ===========================================================================

class TestBuildFilter:
    def setup_method(self):
        from tools.openalex_search import _build_filter
        self.fn = _build_filter

    def test_no_filters_returns_none(self):
        assert self.fn(None, None) is None

    def test_year_start_only(self):
        result = self.fn(2020, None)
        assert result == "from_publication_date:2020-01-01"

    def test_year_end_only(self):
        result = self.fn(None, 2023)
        assert result == "to_publication_date:2023-12-31"

    def test_both_dates(self):
        result = self.fn(2020, 2023)
        assert "from_publication_date:2020-01-01" in result
        assert "to_publication_date:2023-12-31" in result
        assert result == "from_publication_date:2020-01-01,to_publication_date:2023-12-31"

# ===========================================================================
# 5. _build_params
# ===========================================================================

class TestBuildParams:
    def setup_method(self):
        from tools.openalex_search import _build_params, OpenAlexInput, _OPENALEX_EMAIL
        self.fn    = _build_params
        self.Input = OpenAlexInput
        self.email = _OPENALEX_EMAIL

    def _inp(self, **kw) -> OpenAlexInput:
        kw.setdefault("query", "test query")
        return self.Input(**kw)

    def test_mailto_always_present(self):
        params = self.fn(self._inp())
        assert "mailto" in params
        assert params["mailto"] == self.email

    def test_select_always_present(self):
        params = self.fn(self._inp())
        assert "select" in params

    def test_query_mapped_to_search(self):
        params = self.fn(self._inp(query="transformers in NLP"))
        assert params["search"] == "transformers in NLP"

    def test_concept_appended_to_search(self):
        params = self.fn(self._inp(query="cancer", concept=" genomics "))
        assert params["search"] == "cancer genomics"

    def test_limit_mapped_to_per_page(self):
        params = self.fn(self._inp(limit=7))
        assert params["per-page"] == 7

    def test_filter_absent_when_no_constraints(self):
        params = self.fn(self._inp())
        assert "filter" not in params

    def test_filter_present_with_year_start(self):
        params = self.fn(self._inp(year_start=2020))
        assert "filter" in params
        assert "2020-01-01" in params["filter"]

# ===========================================================================
# 6. _parse_work
# ===========================================================================

class TestParseWork:
    def setup_method(self):
        from tools.openalex_search import _parse_work
        self.fn = _parse_work

    # ---- universal schema ----

    def test_required_schema_keys_present(self):
        p = self.fn(_WORK_1)
        required = {"title", "abstract", "authors", "year", "citation_count",
                    "doi", "arxiv_id", "url", "source"}
        assert required.issubset(p.keys())

    def test_source_is_openalex(self):
        assert self.fn(_WORK_1)["source"] == "OpenAlex"
        assert self.fn(_WORK_2)["source"] == "OpenAlex"
        assert self.fn(_WORK_3)["source"] == "OpenAlex"

    # ---- work 1 (fully populated) ----

    def test_work1_title(self):
        assert self.fn(_WORK_1)["title"] == "Attention Is All You Need"

    def test_work1_doi_bare(self):
        assert self.fn(_WORK_1)["doi"] == "10.48550/arXiv.1706.03762"

    def test_work1_arxiv_id_bare(self):
        assert self.fn(_WORK_1)["arxiv_id"] == "1706.03762"

    def test_work1_year(self):
        assert self.fn(_WORK_1)["year"] == 2017

    def test_work1_citation_count(self):
        assert self.fn(_WORK_1)["citation_count"] == 98765

    def test_work1_authors_list(self):
        authors = self.fn(_WORK_1)["authors"]
        assert authors == ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"]

    def test_work1_url_prefers_pdf(self):
        # pdf_url is the top preference
        assert self.fn(_WORK_1)["url"] == "https://arxiv.org/pdf/1706.03762"

    def test_work1_abstract_reconstructed(self):
        abstract = self.fn(_WORK_1)["abstract"]
        assert abstract.startswith("The dominant sequence transduction")

    # ---- work 2 (no arxiv, no pdf_url) ----

    def test_work2_no_arxiv_id(self):
        assert self.fn(_WORK_2)["arxiv_id"] is None

    def test_work2_url_falls_back_to_landing_page(self):
        # No pdf_url → landing_page_url
        assert self.fn(_WORK_2)["url"] == "https://aclanthology.org/N19-1423"

    def test_work2_multi_position_word_in_abstract(self):
        # "a" appears at positions 2 and 6
        abstract = self.fn(_WORK_2)["abstract"]
        words = abstract.split()
        assert words[2] == "a"
        assert words[6] == "a"

    # ---- work 3 (no doi, no primary_location, arxiv in ids) ----

    def test_work3_no_doi(self):
        assert self.fn(_WORK_3)["doi"] is None

    def test_work3_arxiv_id_from_ids_dict(self):
        assert self.fn(_WORK_3)["arxiv_id"] == "2401.00001"

    def test_work3_no_abstract_placeholder(self):
        assert self.fn(_WORK_3)["abstract"] == "No abstract available."

    def test_work3_url_falls_back_to_arxiv_id_url(self):
        # No primary_location, no doi → constructed arxiv URL
        assert self.fn(_WORK_3)["url"] == "https://arxiv.org/abs/2401.00001"

    # ---- edge cases ----

    def test_missing_title_uses_display_name(self):
        work = dict(_WORK_1)
        work["title"] = None
        p = self.fn(work)
        assert p["title"] == "Attention Is All You Need"

    def test_both_title_and_display_name_missing(self):
        work = {**_WORK_1, "title": None, "display_name": None}
        assert self.fn(work)["title"] == "Unknown Title"

    def test_no_authorships_yields_empty_list(self):
        work = {**_WORK_1, "authorships": None}
        assert self.fn(work)["authors"] == []

    def test_authorships_with_missing_display_name_skipped(self):
        work = {**_WORK_1, "authorships": [
            {"author": {"display_name": "Alice"}},
            {"author": {}},                        # no display_name
            {"author": {"display_name": "Bob"}},
        ]}
        assert self.fn(work)["authors"] == ["Alice", "Bob"]

    def test_no_primary_location_doi_fallback(self):
        work = {**_WORK_2, "primary_location": None}
        p = self.fn(work)
        # doi is present → doi URL fallback
        assert p["url"] == "https://doi.org/10.18653/v1/N19-1423"

    def test_no_primary_location_no_doi_arxiv_fallback(self):
        work = {**_WORK_3}   # already no primary_location, no doi, has arxiv
        assert self.fn(work)["url"] == "https://arxiv.org/abs/2401.00001"

    def test_no_primary_location_no_doi_no_arxiv_uses_openalex_id(self):
        work = {**_WORK_3, "ids": {"openalex": "https://openalex.org/W3333333333"}}
        p = self.fn(work)
        assert p["url"] == "https://openalex.org/W3333333333"


# ===========================================================================
# 7. search_openalex — integration (HTTP mocked via _throttled_get)
# ===========================================================================

class TestSearchOpenAlex:
    def setup_method(self):
        import tools.openalex_search as mod
        mod._fallback_cache.clear()

    @patch(PATCH_THROTTLED_GET)
    def test_happy_path_returns_list_of_dicts(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        results = search_openalex("transformer attention", limit=3)
        assert isinstance(results, list)
        assert len(results) == 3

    @patch(PATCH_THROTTLED_GET)
    def test_universal_schema_on_every_result(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        results = search_openalex("transformer attention", limit=3)
        required = {"title", "abstract", "authors", "year", "citation_count",
                    "doi", "arxiv_id", "url", "source"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing: {required - r.keys()}"

    @patch(PATCH_THROTTLED_GET)
    def test_source_is_openalex_on_every_result(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        results = search_openalex("x", limit=3)
        assert all(r["source"] == "OpenAlex" for r in results)

    @patch(PATCH_THROTTLED_GET)
    def test_limit_trims_results(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        results = search_openalex("x", limit=2)
        assert len(results) <= 2

    @patch(PATCH_THROTTLED_GET)
    def test_zero_results_returns_empty_list(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_0)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        results = search_openalex("xyzzy_nonexistent_quux_12345", limit=5)
        assert results == []

    @patch(PATCH_THROTTLED_GET)
    def test_fallback_cache_hit_skips_api(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        r1 = search_openalex("transformer attention", limit=3)
        r2 = search_openalex("transformer attention", limit=3)
        assert mock_get.call_count == 1    # second call was a cache hit
        assert r1 == r2

    @patch(PATCH_THROTTLED_GET)
    def test_mailto_polite_pool_param_sent(self, mock_get):
        """Every request must carry the mailto param (OpenAlex Polite Pool policy)."""
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        search_openalex("cancer", limit=3)
        call_params = mock_get.call_args[0][1]   # positional: (url, params)
        assert "mailto" in call_params

    @patch(PATCH_THROTTLED_GET)
    def test_year_filter_in_params(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        search_openalex("cancer", year_start=2020, year_end=2023, limit=3)
        call_params = mock_get.call_args[0][1]
        assert "filter" in call_params
        assert "2020-01-01" in call_params["filter"]
        assert "2023-12-31" in call_params["filter"]

    @patch(PATCH_THROTTLED_GET)
    def test_concept_filter_in_params(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        search_openalex("cancer", concept="genomics", limit=3)
        call_params = mock_get.call_args[0][1]
        
        # It should now be appended to the search string, not the filter!
        assert call_params["search"] == "cancer genomics"

    @patch(PATCH_THROTTLED_GET)
    def test_no_filter_param_when_no_constraints(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import search_openalex
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        search_openalex("cancer", limit=3)
        call_params = mock_get.call_args[0][1]
        assert "filter" not in call_params

    @patch(PATCH_THROTTLED_GET)
    def test_redis_write_on_cache_miss(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod

        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        fake_redis.setex.return_value = True

        original = mod._redis
        try:
            mod._redis = fake_redis
            mod._fallback_cache.clear()
            from tools.openalex_search import search_openalex
            results = search_openalex("test write", limit=3)
            fake_redis.get.assert_called_once()
            fake_redis.setex.assert_called_once()
            stored = json.loads(fake_redis.setex.call_args[0][2])
            assert stored == results
        finally:
            mod._redis = original

    @patch(PATCH_THROTTLED_GET)
    def test_redis_read_hit_skips_api(self, mock_get):
        import tools.openalex_search as mod
        from tools.openalex_search import search_openalex, _cache_key

        cached_data = [{"title": "Cached", "source": "OpenAlex"}]
        key = _cache_key("cached query", None, None, None, 3)

        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps(cached_data)

        original = mod._redis
        try:
            mod._redis = fake_redis
            result = search_openalex("cached query", limit=3)
            assert result == cached_data
            mock_get.assert_not_called()
        finally:
            mod._redis = original

    @patch(PATCH_THROTTLED_GET)
    def test_redis_get_failure_falls_through_to_api(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod

        fake_redis = MagicMock()
        fake_redis.get.side_effect = Exception("Redis connection reset")
        fake_redis.setex.return_value = True

        original = mod._redis
        try:
            mod._redis = fake_redis
            mod._fallback_cache.clear()
            from tools.openalex_search import search_openalex
            results = search_openalex("redis fail fallthrough", limit=3)
            assert isinstance(results, list)
        finally:
            mod._redis = original


# ===========================================================================
# 8. _throttled_get — retry logic (httpx.Client mocked directly)
# ===========================================================================

class TestRetryLogic:
    def setup_method(self):
        import tools.openalex_search as mod
        mod._fallback_cache.clear()

    @patch("time.sleep", return_value=None)
    def test_retries_twice_then_succeeds(self, mock_sleep):
        """
        _throttled_get raises 429 on the first two attempts then gets a 200.
        httpx.Client.get must be called exactly three times.
        """
        import tools.openalex_search as mod

        exc_429 = _429_error()
        success = _mock_response(OPENALEX_JSON_0)
        success.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            instance = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=instance)
            MockClient.return_value.__exit__  = MagicMock(return_value=False)
            instance.get.side_effect = [exc_429, exc_429, success]

            result = mod._throttled_get(mod.BASE_URL, {})

        assert result is success
        assert instance.get.call_count == 3

    @patch("time.sleep", return_value=None)
    def test_retry_after_header_used_for_sleep(self, mock_sleep):
        """When Retry-After is present, time.sleep must receive that exact value."""
        import tools.openalex_search as mod

        exc_429 = _429_error(retry_after="30")
        success = _mock_response(OPENALEX_JSON_0)
        success.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            instance = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=instance)
            MockClient.return_value.__exit__  = MagicMock(return_value=False)
            instance.get.side_effect = [exc_429, success]

            mod._throttled_get(mod.BASE_URL, {})

        sleep_values = [call[0][0] for call in mock_sleep.call_args_list]
        assert 30 in sleep_values

    @patch("time.sleep", return_value=None)
    def test_exponential_backoff_when_no_retry_after(self, mock_sleep):
        """Without Retry-After, delay should start at RETRY_BASE_DELAY and double."""
        import tools.openalex_search as mod

        exc_429 = _429_error()   # no Retry-After header
        success = _mock_response(OPENALEX_JSON_0)
        success.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            instance = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=instance)
            MockClient.return_value.__exit__  = MagicMock(return_value=False)
            instance.get.side_effect = [exc_429, exc_429, success]

            mod._throttled_get(mod.BASE_URL, {})

        sleep_values = [call[0][0] for call in mock_sleep.call_args_list
                        if call[0][0] >= mod.RETRY_BASE_DELAY]
        # First backoff = RETRY_BASE_DELAY, second = RETRY_BASE_DELAY * 2
        assert mod.RETRY_BASE_DELAY in sleep_values
        assert mod.RETRY_BASE_DELAY * 2 in sleep_values

    @patch("time.sleep", return_value=None)
    def test_non_429_http_error_not_retried(self, mock_sleep):
        """A 500 Internal Server Error should propagate immediately, not be retried."""
        import httpx
        import tools.openalex_search as mod

        req  = MagicMock()
        resp = MagicMock()
        resp.status_code = 500
        resp.headers     = {}
        exc_500 = httpx.HTTPStatusError("500", request=req, response=resp)

        with patch("httpx.Client") as MockClient:
            instance = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=instance)
            MockClient.return_value.__exit__  = MagicMock(return_value=False)
            instance.get.side_effect = [exc_500]

            with pytest.raises(httpx.HTTPStatusError):
                mod._throttled_get(mod.BASE_URL, {})

        assert instance.get.call_count == 1   # no retries

    @patch("time.sleep", return_value=None)
    def test_exhausted_retries_raises_last_exception(self, mock_sleep):
        """After RETRY_ATTEMPTS failures, the last 429 exception should be raised."""
        import tools.openalex_search as mod

        exc_429 = _429_error()

        with patch("httpx.Client") as MockClient:
            instance = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=instance)
            MockClient.return_value.__exit__  = MagicMock(return_value=False)
            instance.get.side_effect = [exc_429] * mod.RETRY_ATTEMPTS

            with pytest.raises(Exception):
                mod._throttled_get(mod.BASE_URL, {})

        assert instance.get.call_count == mod.RETRY_ATTEMPTS


# ===========================================================================
# 9. _tool_fn and openalex_tool (LangChain interface)
# ===========================================================================

class TestToolFn:
    def setup_method(self):
        import tools.openalex_search as mod
        mod._fallback_cache.clear()

    @patch(PATCH_THROTTLED_GET)
    def test_returns_string(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        from tools.openalex_search import _tool_fn
        assert isinstance(_tool_fn("transformers", limit=3), str)

    @patch(PATCH_THROTTLED_GET)
    def test_header_contains_openalex_and_query(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("transformers", limit=3)
        assert "OpenAlex results" in result
        assert "transformers" in result

    @patch(PATCH_THROTTLED_GET)
    def test_output_contains_title(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("x", limit=3)
        assert "Attention Is All You Need" in result

    @patch(PATCH_THROTTLED_GET)
    def test_output_contains_doi(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("x", limit=3)
        assert "10.48550/arXiv.1706.03762" in result

    @patch(PATCH_THROTTLED_GET)
    def test_output_contains_arxiv_id(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("x", limit=3)
        assert "1706.03762" in result

    @patch(PATCH_THROTTLED_GET)
    def test_output_contains_citation_count(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("x", limit=3)
        assert "98765" in result

    @patch(PATCH_THROTTLED_GET)
    def test_authors_beyond_three_show_et_al(self, mock_get):
        """Work 1 has 4 authors — output should include 'et al.'"""
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("x", limit=1)
        assert "et al." in result

    @patch(PATCH_THROTTLED_GET)
    def test_url_in_output(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("x", limit=1)
        assert "https://arxiv.org/pdf/1706.03762" in result

    @patch(PATCH_THROTTLED_GET)
    def test_zero_results_returns_no_results_string(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_0)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import _tool_fn
        result = _tool_fn("xyzzy_nonexistent", limit=5)
        assert "No results found" in result
        assert "xyzzy_nonexistent" in result

    @patch(PATCH_THROTTLED_GET)
    def test_langchain_tool_invoke(self, mock_get):
        mock_get.return_value = _mock_response(OPENALEX_JSON_3)
        import tools.openalex_search as mod
        mod._fallback_cache.clear()
        from tools.openalex_search import openalex_tool
        result = openalex_tool.invoke({"query": "transformers", "limit": 3})
        assert isinstance(result, str)
        assert "OpenAlex results" in result

    def test_tool_name(self):
        from tools.openalex_search import openalex_tool
        assert openalex_tool.name == "openalex_search"

    def test_tool_description_mentions_openalex(self):
        from tools.openalex_search import openalex_tool
        assert "OpenAlex" in openalex_tool.description

    def test_tool_description_mentions_pdf(self):
        from tools.openalex_search import openalex_tool
        assert "PDF" in openalex_tool.description

    def test_tool_schema_has_all_fields(self):
        from tools.openalex_search import OpenAlexInput
        fields = OpenAlexInput.model_fields
        for expected in ("query", "year_start", "year_end", "concept", "limit"):
            assert expected in fields, f"Missing field: {expected}"


# ===========================================================================
# 10. OpenAlexInput schema validation
# ===========================================================================

class TestOpenAlexInputSchema:
    def setup_method(self):
        from tools.openalex_search import OpenAlexInput, MAX_RESULTS_CAP
        self.Input = OpenAlexInput
        self.cap   = MAX_RESULTS_CAP

    def test_query_is_required(self):
        with pytest.raises(Exception):
            self.Input()

    def test_default_limit_is_5(self):
        assert self.Input(query="x").limit == 5

    def test_limit_below_1_raises(self):
        with pytest.raises(Exception):
            self.Input(query="x", limit=0)

    def test_limit_above_cap_raises(self):
        with pytest.raises(Exception):
            self.Input(query="x", limit=self.cap + 1)

    def test_limit_at_cap_is_valid(self):
        assert self.Input(query="x", limit=self.cap).limit == self.cap

    def test_optional_fields_default_to_none(self):
        inp = self.Input(query="x")
        assert inp.year_start is None
        assert inp.year_end   is None
        assert inp.concept    is None


# ===========================================================================
# 11. clear_cache and cache_stats
# ===========================================================================

class TestCacheManagement:
    def setup_method(self):
        import tools.openalex_search as mod
        mod._fallback_cache.clear()

    def test_clear_cache_fallback_empties_dict(self):
        import tools.openalex_search as mod
        mod._fallback_cache["k"] = [{"title": "t"}]
        mod.clear_cache()
        assert len(mod._fallback_cache) == 0

    def test_clear_cache_fallback_leaves_other_keys_unrelated(self):
        """Two entries — both removed by clear_cache."""
        import tools.openalex_search as mod
        mod._fallback_cache["k1"] = []
        mod._fallback_cache["k2"] = []
        mod.clear_cache()
        assert mod._fallback_cache == {}

    def test_cache_stats_fallback_entry_count(self):
        import tools.openalex_search as mod
        mod._fallback_cache["k1"] = []
        mod._fallback_cache["k2"] = []
        stats = mod.cache_stats()
        assert stats["backend"].startswith("in-process")
        assert stats["entries"] == 2

    def test_cache_stats_fallback_lists_keys(self):
        import tools.openalex_search as mod
        mod._fallback_cache["k1"] = []
        stats = mod.cache_stats()
        assert "k1" in stats["keys"]

    def test_clear_cache_redis_calls_scan_and_delete(self):
        import tools.openalex_search as mod
        fake = MagicMock()
        fake.scan.return_value = (0, ["researchflow:openalex:abc",
                                       "researchflow:openalex:def"])
        fake.delete.return_value = 2
        original = mod._redis
        try:
            mod._redis = fake
            mod.clear_cache()
            fake.scan.assert_called()
            fake.delete.assert_called()
        finally:
            mod._redis = original

    def test_clear_cache_redis_error_does_not_raise(self):
        import tools.openalex_search as mod
        fake = MagicMock()
        fake.scan.side_effect = Exception("Redis down")
        original = mod._redis
        try:
            mod._redis = fake
            mod.clear_cache()  # must not raise
        finally:
            mod._redis = original

    def test_cache_stats_redis_mode(self):
        import tools.openalex_search as mod
        fake = MagicMock()
        fake.scan.return_value = (0, ["researchflow:openalex:abc"])
        fake.ttl.return_value  = 86400
        original = mod._redis
        try:
            mod._redis = fake
            stats = mod.cache_stats()
            assert stats["backend"]    == "redis"
            assert stats["entries"]    == 1
            assert stats["key_prefix"] == mod.CACHE_KEY_PREFIX
        finally:
            mod._redis = original

    def test_cache_stats_redis_error_returns_error_dict(self):
        import tools.openalex_search as mod
        fake = MagicMock()
        fake.scan.side_effect = Exception("Redis down")
        original = mod._redis
        try:
            mod._redis = fake
            stats = mod.cache_stats()
            assert stats["backend"] == "redis"
            assert "error" in stats
        finally:
            mod._redis = original


# ===========================================================================
# 12. _cache_key determinism
# ===========================================================================

class TestCacheKey:
    def setup_method(self):
        from tools.openalex_search import _cache_key, CACHE_KEY_PREFIX
        self.fn     = _cache_key
        self.prefix = CACHE_KEY_PREFIX

    def test_same_args_same_key(self):
        k1 = self.fn("cancer", 2020, 2023, "genomics", 5)
        k2 = self.fn("cancer", 2020, 2023, "genomics", 5)
        assert k1 == k2

    def test_different_query_different_key(self):
        assert self.fn("cancer", None, None, None, 5) != \
               self.fn("diabetes", None, None, None, 5)

    def test_different_year_start_different_key(self):
        assert self.fn("x", 2019, None, None, 5) != \
               self.fn("x", 2020, None, None, 5)

    def test_different_concept_different_key(self):
        assert self.fn("x", None, None, "genomics", 5) != \
               self.fn("x", None, None, "oncology", 5)

    def test_different_limit_different_key(self):
        assert self.fn("x", None, None, None, 3) != \
               self.fn("x", None, None, None, 5)

    def test_key_has_correct_prefix(self):
        k = self.fn("x", None, None, None, 5)
        assert k.startswith(self.prefix)

    def test_key_has_no_whitespace(self):
        k = self.fn("multi word query", 2020, 2023, "machine learning", 5)
        assert " " not in k

    def test_none_concept_vs_set_concept_differ(self):
        assert self.fn("x", None, None, None, 5) != \
               self.fn("x", None, None, "genomics", 5)