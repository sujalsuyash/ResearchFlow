"""
tests/test_arxiv_search.py

Full unit test suite for tools/arxiv_search.py.
No network access and no Redis required — everything is mocked.
"""

import json
import sys
import os
import hashlib
from datetime import datetime
from unittest.mock import patch, MagicMock, call

import pytest
import arxiv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers to build fake arxiv.Result objects
# ---------------------------------------------------------------------------

def _make_result(
    title="Attention Is All You Need",
    summary="The dominant sequence transduction models...",
    authors=("Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones"),
    year=2017,
    doi="https://doi.org/10.48550/arXiv.1706.03762",
    entry_id="https://arxiv.org/abs/1706.03762v5",
    pdf_url="https://arxiv.org/pdf/1706.03762v5",
    categories=("cs.CL", "cs.LG"),
) -> arxiv.Result:
    """Construct a minimal arxiv.Result using its public __init__."""
    return arxiv.Result(
        entry_id=entry_id,
        updated=datetime(year, 6, 12),
        published=datetime(year, 6, 12),
        title=title,
        authors=[arxiv.Result.Author(name=a) for a in authors],
        summary=summary,
        doi=doi,
        categories=list(categories),
        links=[
            arxiv.Result.Link(
                href=pdf_url,
                title="pdf",
                rel="related",
                content_type="application/pdf",
            )
        ],
    )


FAKE_RESULT = _make_result()

FAKE_RESULT_NO_DOI = _make_result(
    doi="",
    entry_id="https://arxiv.org/abs/2101.99999v1",
    pdf_url="https://arxiv.org/pdf/2101.99999v1",
)

FAKE_RESULT_OLD = _make_result(
    title="Old Paper",
    year=2018,
    entry_id="https://arxiv.org/abs/1801.00001v1",
    pdf_url="https://arxiv.org/pdf/1801.00001v1",
    doi="",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def force_no_redis(monkeypatch):
    monkeypatch.setattr("tools.arxiv_search._redis", None)


@pytest.fixture(autouse=True)
def clean_fallback_cache():
    from tools.arxiv_search import clear_cache
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# 1. PARSING
# ---------------------------------------------------------------------------

class TestParsing:

    @patch("tools.arxiv_search._arxiv_client")
    def test_all_fields_mapped_correctly(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import search_arxiv
        results = search_arxiv("attention", limit=1)

        assert len(results) == 1
        p = results[0]
        assert p["title"]          == "Attention Is All You Need"
        assert p["abstract"]       == "The dominant sequence transduction models..."
        assert p["year"]           == 2017
        assert p["doi"]            == "10.48550/arXiv.1706.03762"   # bare, not URL
        assert p["arxiv_id"]       == "1706.03762v5"
        assert p["url"]            == "https://arxiv.org/abs/1706.03762v5"
        assert p["pdf_url"]        == "https://arxiv.org/pdf/1706.03762v5"
        assert "cs.CL"             in p["categories"]
        assert p["source"]         == "arXiv"
        assert p["citation_count"] is None          # arXiv never has this
        assert "Vaswani"           in p["authors"]

    @patch("tools.arxiv_search._arxiv_client")
    def test_doi_stripped_of_resolver_prefix(self, mock_client):
        result = _make_result(doi="https://doi.org/10.1234/test.paper")
        mock_client.results.return_value = iter([result])
        from tools.arxiv_search import search_arxiv
        p = search_arxiv("test", limit=1)[0]
        assert p["doi"] == "10.1234/test.paper"
        assert "doi.org" not in p["doi"]

    @patch("tools.arxiv_search._arxiv_client")
    def test_empty_doi_becomes_none(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT_NO_DOI])
        from tools.arxiv_search import search_arxiv
        p = search_arxiv("no doi", limit=1)[0]
        assert p["doi"] is None

    @patch("tools.arxiv_search._arxiv_client")
    def test_bare_doi_passes_through_unchanged(self, mock_client):
        result = _make_result(doi="10.9999/bare.doi")
        mock_client.results.return_value = iter([result])
        from tools.arxiv_search import search_arxiv
        p = search_arxiv("bare doi", limit=1)[0]
        assert p["doi"] == "10.9999/bare.doi"

    @patch("tools.arxiv_search._arxiv_client")
    def test_empty_api_response_returns_empty_list(self, mock_client):
        mock_client.results.return_value = iter([])
        from tools.arxiv_search import search_arxiv
        assert search_arxiv("nothing", limit=5) == []

    @patch("tools.arxiv_search._arxiv_client")
    def test_multiline_abstract_is_flattened(self, mock_client):
        result = _make_result(summary="Line one.\nLine two.\nLine three.")
        mock_client.results.return_value = iter([result])
        from tools.arxiv_search import search_arxiv
        p = search_arxiv("multiline", limit=1)[0]
        assert "\n" not in p["abstract"]
        assert "Line one." in p["abstract"]


# ---------------------------------------------------------------------------
# 2. YEAR FILTER
# ---------------------------------------------------------------------------

class TestYearFilter:

    @patch("tools.arxiv_search._arxiv_client")
    def test_year_start_filters_old_papers(self, mock_client):
        old   = _make_result(title="Old", year=2019, entry_id="https://arxiv.org/abs/1901.00001v1", doi="")
        fresh = _make_result(title="Fresh", year=2023, entry_id="https://arxiv.org/abs/2301.00001v1", doi="")
        mock_client.results.return_value = iter([old, fresh])

        from tools.arxiv_search import search_arxiv
        results = search_arxiv("test", year_start=2022, limit=5)

        titles = [r["title"] for r in results]
        assert "Old" not in titles
        assert "Fresh" in titles

    @patch("tools.arxiv_search._arxiv_client")
    def test_year_end_filters_new_papers(self, mock_client):
        early = _make_result(title="Early", year=2020, entry_id="https://arxiv.org/abs/2001.00001v1", doi="")
        late  = _make_result(title="Late",  year=2024, entry_id="https://arxiv.org/abs/2401.00001v1", doi="")
        mock_client.results.return_value = iter([early, late])

        from tools.arxiv_search import search_arxiv
        results = search_arxiv("test", year_end=2021, limit=5)

        titles = [r["title"] for r in results]
        assert "Late" not in titles
        assert "Early" in titles

    @patch("tools.arxiv_search._arxiv_client")
    def test_year_range_keeps_papers_within_bounds(self, mock_client):
        p2020 = _make_result(title="2020", year=2020, entry_id="https://arxiv.org/abs/2001.11111v1", doi="")
        p2022 = _make_result(title="2022", year=2022, entry_id="https://arxiv.org/abs/2201.11111v1", doi="")
        p2024 = _make_result(title="2024", year=2024, entry_id="https://arxiv.org/abs/2401.11111v1", doi="")
        mock_client.results.return_value = iter([p2020, p2022, p2024])

        from tools.arxiv_search import search_arxiv
        results = search_arxiv("test", year_start=2021, year_end=2023, limit=5)

        titles = [r["title"] for r in results]
        assert "2020" not in titles
        assert "2022" in titles
        assert "2024" not in titles

    @patch("tools.arxiv_search._arxiv_client")
    def test_year_filter_over_fetches(self, mock_client):
        """When year filter is active, max_results passed to Search is limit * multiplier."""
        mock_client.results.return_value = iter([])
        from tools.arxiv_search import search_arxiv, YEAR_FILTER_MULTIPLIER
        search_arxiv("test", year_start=2022, limit=5)

        search_obj = mock_client.results.call_args[0][0]
        assert search_obj.max_results == 5 * YEAR_FILTER_MULTIPLIER

    @patch("tools.arxiv_search._arxiv_client")
    def test_no_year_filter_fetches_exactly_limit(self, mock_client):
        mock_client.results.return_value = iter([])
        from tools.arxiv_search import search_arxiv
        search_arxiv("test", limit=5)

        search_obj = mock_client.results.call_args[0][0]
        assert search_obj.max_results == 5

    @patch("tools.arxiv_search._arxiv_client")
    def test_paper_with_unknown_year_is_kept(self, mock_client):
        """A result with year=None should not be dropped by the year filter."""
        result = _make_result(title="Dateless", year=2020, entry_id="https://arxiv.org/abs/9999.00001v1", doi="")
        result.published = None         # simulate missing date
        # year will be None after _parse_result
        mock_client.results.return_value = iter([result])

        from tools.arxiv_search import search_arxiv
        results = search_arxiv("test", year_start=2022, limit=5)
        # paper with None year should survive the filter
        assert any(r["title"] == "Dateless" for r in results)


# ---------------------------------------------------------------------------
# 3. QUERY BUILDER
# ---------------------------------------------------------------------------

class TestQueryBuilder:

    def test_plain_query_unchanged(self):
        from tools.arxiv_search import _build_query, ArxivInput
        inp = ArxivInput(query="transformer attention")
        assert _build_query(inp) == "transformer attention"

    def test_single_category_appended(self):
        from tools.arxiv_search import _build_query, ArxivInput
        inp = ArxivInput(query="graph neural network", categories="cs.LG")
        result = _build_query(inp)
        assert "cat:cs.LG" in result
        assert "graph neural network" in result

    def test_multiple_categories_or_combined(self):
        from tools.arxiv_search import _build_query, ArxivInput
        inp = ArxivInput(query="diffusion models", categories="cs.LG,cs.CV")
        result = _build_query(inp)
        assert "cat:cs.LG" in result
        assert "cat:cs.CV" in result
        assert " OR " in result

    def test_category_whitespace_is_stripped(self):
        from tools.arxiv_search import _build_query, ArxivInput
        inp = ArxivInput(query="test", categories=" cs.LG , cs.AI ")
        result = _build_query(inp)
        assert "cat:cs.LG" in result
        assert "cat:cs.AI" in result
        # No stray spaces inside cat: clauses
        assert "cat: " not in result

    @patch("tools.arxiv_search._arxiv_client")
    def test_category_filter_reaches_arxiv_client(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import search_arxiv
        search_arxiv("attention", categories="cs.CL", limit=1)

        search_obj = mock_client.results.call_args[0][0]
        assert "cat:cs.CL" in search_obj.query

    @patch("tools.arxiv_search._arxiv_client")
    def test_sort_criterion_is_relevance(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import search_arxiv
        search_arxiv("test", limit=1)

        search_obj = mock_client.results.call_args[0][0]
        assert search_obj.sort_by == arxiv.SortCriterion.Relevance


# ---------------------------------------------------------------------------
# 4. FALLBACK CACHE
# ---------------------------------------------------------------------------

class TestFallbackCache:

    @patch("tools.arxiv_search._arxiv_client")
    def test_second_call_hits_cache_not_api(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import search_arxiv
        r1 = search_arxiv("cache test", limit=1)
        # Reset iterator so a second API call would return nothing
        mock_client.results.return_value = iter([])
        r2 = search_arxiv("cache test", limit=1)
        assert r1 == r2
        assert mock_client.results.call_count == 1

    @patch("tools.arxiv_search._arxiv_client")
    def test_different_limit_is_separate_cache_entry(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import search_arxiv, cache_stats
        search_arxiv("isolation", limit=2)
        mock_client.results.return_value = iter([FAKE_RESULT])
        search_arxiv("isolation", limit=4)
        assert cache_stats()["entries"] == 2

    @patch("tools.arxiv_search._arxiv_client")
    def test_clear_cache_empties_fallback(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import search_arxiv, clear_cache, cache_stats
        search_arxiv("to clear", limit=1)
        assert cache_stats()["entries"] == 1
        clear_cache()
        assert cache_stats()["entries"] == 0


# ---------------------------------------------------------------------------
# 5. REDIS CACHE
# ---------------------------------------------------------------------------

class TestRedisCache:

    @patch("tools.arxiv_search._arxiv_client")
    def test_redis_hit_skips_api(self, mock_client):
        cached = [{"title": "Cached", "doi": None, "source": "arXiv"}]
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached)

        with patch("tools.arxiv_search._redis", mock_redis):
            from tools.arxiv_search import search_arxiv
            results = search_arxiv("cached query", limit=1)

        assert results == cached
        mock_client.results.assert_not_called()

    @patch("tools.arxiv_search._arxiv_client")
    def test_redis_miss_writes_with_correct_ttl(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        with patch("tools.arxiv_search._redis", mock_redis):
            from tools.arxiv_search import search_arxiv, CACHE_TTL_SECONDS
            search_arxiv("redis miss", limit=1)

        mock_redis.setex.assert_called_once()
        _, ttl, payload = mock_redis.setex.call_args[0]
        assert ttl == CACHE_TTL_SECONDS
        assert json.loads(payload)[0]["title"] == "Attention Is All You Need"

    @patch("tools.arxiv_search._arxiv_client")
    def test_redis_get_failure_falls_through_to_api(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("connection reset")

        with patch("tools.arxiv_search._redis", mock_redis):
            from tools.arxiv_search import search_arxiv
            results = search_arxiv("redis get error", limit=1)

        assert len(results) == 1
        mock_client.results.assert_called_once()

    @patch("tools.arxiv_search._arxiv_client")
    def test_redis_set_failure_does_not_crash(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.setex.side_effect = Exception("write timeout")

        with patch("tools.arxiv_search._redis", mock_redis):
            from tools.arxiv_search import search_arxiv
            results = search_arxiv("redis set error", limit=1)

        assert len(results) == 1

    def test_cache_stats_reports_redis_backend(self):
        mock_redis = MagicMock()
        mock_redis.scan.return_value = (0, ["researchflow:arxiv:abc123"])
        mock_redis.ttl.return_value = 400000

        with patch("tools.arxiv_search._redis", mock_redis):
            from tools.arxiv_search import cache_stats, CACHE_TTL_SECONDS
            stats = cache_stats()

        assert stats["backend"]                  == "redis"
        assert stats["entries"]                  == 1
        assert stats["ttl_seconds"]              == CACHE_TTL_SECONDS
        assert stats["sample_key_ttl_remaining"] == 400000


# ---------------------------------------------------------------------------
# 6. CACHE KEY
# ---------------------------------------------------------------------------

class TestCacheKey:

    def test_key_is_deterministic(self):
        from tools.arxiv_search import _cache_key
        assert _cache_key("RAG", 2022, None, None, 5) == _cache_key("RAG", 2022, None, None, 5)

    def test_key_differs_by_query(self):
        from tools.arxiv_search import _cache_key
        assert _cache_key("RAG", None, None, None, 5) != _cache_key("GNN", None, None, None, 5)

    def test_key_differs_by_year_start(self):
        from tools.arxiv_search import _cache_key
        assert _cache_key("RAG", 2022, None, None, 5) != _cache_key("RAG", 2023, None, None, 5)

    def test_key_differs_by_categories(self):
        from tools.arxiv_search import _cache_key
        assert _cache_key("RAG", None, None, "cs.LG", 5) != _cache_key("RAG", None, None, None, 5)

    def test_key_has_correct_prefix(self):
        from tools.arxiv_search import _cache_key, CACHE_KEY_PREFIX
        assert _cache_key("x", None, None, None, 5).startswith(CACHE_KEY_PREFIX)

    def test_key_prefix_different_from_semantic_scholar(self):
        """Ensure arXiv and SS keys can coexist in the same Redis db."""
        from tools.arxiv_search import CACHE_KEY_PREFIX as ARXIV_PREFIX
        from tools.semantic_scholar import CACHE_KEY_PREFIX as SS_PREFIX
        assert ARXIV_PREFIX != SS_PREFIX

    def test_key_matches_expected_md5(self):
        from tools.arxiv_search import _cache_key, CACHE_KEY_PREFIX
        raw    = "rag|2022|None|None|3"
        digest = hashlib.md5(raw.encode()).hexdigest()
        assert _cache_key("rag", 2022, None, None, 3) == f"{CACHE_KEY_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# 7. LANGCHAIN TOOL INTERFACE
# ---------------------------------------------------------------------------

class TestLangChainTool:

    @patch("tools.arxiv_search._arxiv_client")
    def test_tool_returns_string(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import arxiv_tool
        result = arxiv_tool.invoke({"query": "transformers", "limit": 1})
        assert isinstance(result, str)

    @patch("tools.arxiv_search._arxiv_client")
    def test_tool_output_contains_key_fields(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import arxiv_tool
        output = arxiv_tool.invoke({"query": "transformers", "limit": 1})
        assert "Attention Is All You Need" in output
        assert "1706.03762v5" in output          # arXiv ID
        assert "10.48550/arXiv.1706.03762" in output  # DOI
        assert "arxiv.org/pdf" in output         # PDF URL
        assert "cs.CL" in output                 # category

    @patch("tools.arxiv_search._arxiv_client")
    def test_tool_truncates_long_author_list(self, mock_client):
        mock_client.results.return_value = iter([FAKE_RESULT])
        from tools.arxiv_search import arxiv_tool
        output = arxiv_tool.invoke({"query": "transformers", "limit": 1})
        assert "et al." in output   # FAKE_RESULT has 5 authors

    @patch("tools.arxiv_search._arxiv_client")
    def test_tool_no_results_message(self, mock_client):
        mock_client.results.return_value = iter([])
        from tools.arxiv_search import arxiv_tool
        output = arxiv_tool.invoke({"query": "xyzzy", "limit": 1})
        assert "No results found" in output

    def test_tool_name_and_description(self):
        from tools.arxiv_search import arxiv_tool
        assert arxiv_tool.name == "arxiv_search"
        assert "arXiv" in arxiv_tool.description
        assert "cs.LG" in arxiv_tool.description   # category example in desc