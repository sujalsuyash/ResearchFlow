"""
tests/test_semantic_scholar.py

Full unit test suite for tools/semantic_scholar.py.
No network access and no Redis required — everything is mocked.
"""

import json
import sys
import os
import hashlib
import pytest
from unittest.mock import patch, MagicMock, call

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so imports work from any cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FAKE_PAPER = {
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on complex recurrent...",
    "authors": [{"name": "Vaswani"}, {"name": "Shazeer"}, {"name": "Parmar"},
                {"name": "Uszkoreit"}, {"name": "Jones"}],
    "year": 2017,
    "citationCount": 90000,
    "externalIds": {"DOI": "10.48550/arXiv.1706.03762", "ArXiv": "1706.03762"},
    "url": "https://www.semanticscholar.org/paper/abc123",
    "fieldsOfStudy": ["Computer Science"],
    "tldr": None,
}

FAKE_API_RESPONSE = {"data": [FAKE_PAPER]}

NO_ABSTRACT_PAPER = {**FAKE_PAPER, "abstract": None, "tldr": {"text": "A transformer model."}}
NO_DOI_PAPER      = {**FAKE_PAPER, "externalIds": {}, "url": None}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def force_no_redis(monkeypatch):
    """Every test runs with Redis disabled unless it explicitly patches _redis."""
    monkeypatch.setattr("tools.semantic_scholar._redis", None)


@pytest.fixture(autouse=True)
def clean_fallback_cache():
    """Wipe the in-process fallback cache before and after every test."""
    from tools.semantic_scholar import clear_cache
    clear_cache()
    yield
    clear_cache()


def _make_mock_http(response_data: dict) -> MagicMock:
    """Return a mock httpx.Client whose .get() returns a 200 with response_data."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp
    return mock_client


# ============================================================
# 1. PARSING
# ============================================================

class TestParsing:

    @patch("tools.semantic_scholar.httpx.Client")
    def test_all_fields_mapped_correctly(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        from tools.semantic_scholar import search_semantic_scholar
        results = search_semantic_scholar("attention", limit=1)

        assert len(results) == 1
        p = results[0]
        assert p["title"]          == "Attention Is All You Need"
        assert p["year"]           == 2017
        assert p["citation_count"] == 90000
        assert p["doi"]            == "10.48550/arXiv.1706.03762"
        assert p["arxiv_id"]       == "1706.03762"
        assert p["url"]            == "https://www.semanticscholar.org/paper/abc123"
        assert p["fields_of_study"]== ["Computer Science"]
        assert p["source"]         == "Semantic Scholar"
        assert "Vaswani" in p["authors"]

    @patch("tools.semantic_scholar.httpx.Client")
    def test_tldr_used_when_abstract_missing(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http({"data": [NO_ABSTRACT_PAPER]})
        from tools.semantic_scholar import search_semantic_scholar
        results = search_semantic_scholar("transformers", limit=1)
        assert results[0]["abstract"] == "A transformer model."

    @patch("tools.semantic_scholar.httpx.Client")
    def test_placeholder_when_both_abstract_and_tldr_missing(self, mock_client_cls):
        bare = {**FAKE_PAPER, "abstract": None, "tldr": None}
        mock_client_cls.return_value = _make_mock_http({"data": [bare]})
        from tools.semantic_scholar import search_semantic_scholar
        results = search_semantic_scholar("bare paper", limit=1)
        assert results[0]["abstract"] == "No abstract available."

    @patch("tools.semantic_scholar.httpx.Client")
    def test_doi_fallback_url_when_url_missing(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http({"data": [NO_DOI_PAPER]})
        from tools.semantic_scholar import search_semantic_scholar
        results = search_semantic_scholar("no doi", limit=1)
        # No DOI and no URL → url should be None
        assert results[0]["url"] is None
        assert results[0]["doi"] is None

    @patch("tools.semantic_scholar.httpx.Client")
    def test_empty_api_response_returns_empty_list(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http({"data": []})
        from tools.semantic_scholar import search_semantic_scholar
        results = search_semantic_scholar("nothing here", limit=5)
        assert results == []


# ============================================================
# 2. FALLBACK CACHE
# ============================================================

class TestFallbackCache:

    @patch("tools.semantic_scholar.httpx.Client")
    def test_second_call_hits_cache_not_api(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        from tools.semantic_scholar import search_semantic_scholar
        r1 = search_semantic_scholar("cache test", limit=1)
        r2 = search_semantic_scholar("cache test", limit=1)
        assert r1 == r2
        assert mock_client_cls.call_count == 1   # network hit only once

    @patch("tools.semantic_scholar.httpx.Client")
    def test_different_limit_is_separate_cache_entry(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        from tools.semantic_scholar import search_semantic_scholar, cache_stats
        search_semantic_scholar("isolation", limit=3)
        search_semantic_scholar("isolation", limit=5)
        assert cache_stats()["entries"] == 2     # two distinct keys

    @patch("tools.semantic_scholar.httpx.Client")
    def test_clear_cache_empties_fallback(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        from tools.semantic_scholar import search_semantic_scholar, clear_cache, cache_stats
        search_semantic_scholar("to be cleared", limit=1)
        assert cache_stats()["entries"] == 1
        clear_cache()
        assert cache_stats()["entries"] == 0


# ============================================================
# 3. REDIS CACHE
# ============================================================

class TestRedisCache:

    @patch("tools.semantic_scholar.httpx.Client")
    def test_redis_hit_skips_api(self, mock_client_cls):
        cached = [{"title": "Cached Paper", "doi": "10.1/x", "source": "Semantic Scholar"}]
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached)

        with patch("tools.semantic_scholar._redis", mock_redis):
            from tools.semantic_scholar import search_semantic_scholar
            results = search_semantic_scholar("redis hit", limit=1)

        assert results == cached
        mock_client_cls.assert_not_called()
        mock_redis.get.assert_called_once()

    @patch("tools.semantic_scholar.httpx.Client")
    def test_redis_miss_calls_api_and_writes_with_ttl(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        mock_redis = MagicMock()
        mock_redis.get.return_value = None       # cache miss

        with patch("tools.semantic_scholar._redis", mock_redis):
            from tools.semantic_scholar import search_semantic_scholar, CACHE_TTL_SECONDS
            results = search_semantic_scholar("redis miss", limit=1)

        assert results[0]["title"] == "Attention Is All You Need"
        mock_redis.setex.assert_called_once()
        key, ttl, payload = mock_redis.setex.call_args[0]
        assert ttl == CACHE_TTL_SECONDS                    # 604800
        assert json.loads(payload)[0]["title"] == "Attention Is All You Need"

    @patch("tools.semantic_scholar.httpx.Client")
    def test_redis_get_failure_falls_through_to_api(self, mock_client_cls):
        """If Redis raises on GET, we must still return API results."""
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("connection reset")

        with patch("tools.semantic_scholar._redis", mock_redis):
            from tools.semantic_scholar import search_semantic_scholar
            results = search_semantic_scholar("redis error read", limit=1)

        assert len(results) == 1
        mock_client_cls.assert_called_once()

    @patch("tools.semantic_scholar.httpx.Client")
    def test_redis_set_failure_does_not_crash(self, mock_client_cls):
        """If Redis raises on SET, results still returned to caller."""
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.setex.side_effect = Exception("write timeout")

        with patch("tools.semantic_scholar._redis", mock_redis):
            from tools.semantic_scholar import search_semantic_scholar
            results = search_semantic_scholar("redis error write", limit=1)

        assert len(results) == 1                # result still returned

    def test_cache_stats_reports_redis_backend(self):
        mock_redis = MagicMock()
        mock_redis.scan.return_value = (0, ["researchflow:ss:abc123"])
        mock_redis.ttl.return_value = 500000

        with patch("tools.semantic_scholar._redis", mock_redis):
            from tools.semantic_scholar import cache_stats, CACHE_TTL_SECONDS
            stats = cache_stats()

        assert stats["backend"]      == "redis"
        assert stats["entries"]      == 1
        assert stats["ttl_seconds"]  == CACHE_TTL_SECONDS
        assert stats["sample_key_ttl_remaining"] == 500000


# ============================================================
# 4. RETRY / RATE LIMITING
# ============================================================

class TestRetry:

    @patch("tools.semantic_scholar.time.sleep")
    @patch("tools.semantic_scholar.httpx.Client")
    def test_429_triggers_exponential_backoff_then_succeeds(self, mock_client_cls, mock_sleep):
        import httpx

        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {}
        rate_exc = httpx.HTTPStatusError("429", request=MagicMock(), response=rate_resp)

        ok_resp = MagicMock()
        ok_resp.json.return_value = FAKE_API_RESPONSE
        ok_resp.raise_for_status = MagicMock()

        # Fail twice, succeed on third attempt
        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__ = MagicMock(return_value=False)
        http.get.side_effect = [rate_exc, rate_exc, ok_resp]
        mock_client_cls.return_value = http

        from tools.semantic_scholar import search_semantic_scholar, RETRY_BASE_DELAY
        results = search_semantic_scholar("backoff test", limit=1)

        assert len(results) == 1
        # First backoff = 10s, second = 20s
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list if c[0][0] >= 1]
        assert RETRY_BASE_DELAY in sleep_calls       # 10s sleep happened
        assert RETRY_BASE_DELAY * 2 in sleep_calls   # 20s sleep happened

    @patch("tools.semantic_scholar.time.sleep")
    @patch("tools.semantic_scholar.httpx.Client")
    def test_retry_after_header_is_honoured(self, mock_client_cls, mock_sleep):
        import httpx

        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "30"}
        rate_exc = httpx.HTTPStatusError("429", request=MagicMock(), response=rate_resp)

        ok_resp = MagicMock()
        ok_resp.json.return_value = FAKE_API_RESPONSE
        ok_resp.raise_for_status = MagicMock()

        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__ = MagicMock(return_value=False)
        http.get.side_effect = [rate_exc, ok_resp]
        mock_client_cls.return_value = http

        from tools.semantic_scholar import search_semantic_scholar
        search_semantic_scholar("retry-after test", limit=1)

        mock_sleep.assert_any_call(30)   # used server's value, not exponential

    @patch("tools.semantic_scholar.time.sleep")
    @patch("tools.semantic_scholar.httpx.Client")
    def test_persistent_429_raises_after_all_retries(self, mock_client_cls, mock_sleep):
        import httpx

        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {}
        rate_exc = httpx.HTTPStatusError("429", request=MagicMock(), response=rate_resp)

        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__ = MagicMock(return_value=False)
        http.get.side_effect = rate_exc   # always 429
        mock_client_cls.return_value = http

        from tools.semantic_scholar import search_semantic_scholar
        with pytest.raises(httpx.HTTPStatusError):
            search_semantic_scholar("always 429", limit=1)

    @patch("tools.semantic_scholar.time.sleep")
    @patch("tools.semantic_scholar.httpx.Client")
    def test_non_429_http_error_raises_immediately(self, mock_client_cls, mock_sleep):
        """A 500 should raise right away with no retries."""
        import httpx

        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.headers = {}
        server_err = httpx.HTTPStatusError("500", request=MagicMock(), response=err_resp)

        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__ = MagicMock(return_value=False)
        http.get.side_effect = server_err
        mock_client_cls.return_value = http

        from tools.semantic_scholar import search_semantic_scholar
        with pytest.raises(httpx.HTTPStatusError):
            search_semantic_scholar("server error", limit=1)

        assert http.get.call_count == 1  # no retry attempted


# ============================================================
# 5. CACHE KEY CORRECTNESS
# ============================================================

class TestCacheKey:

    def test_key_is_deterministic(self):
        from tools.semantic_scholar import _cache_key
        k1 = _cache_key("RAG", 2022, None, None, 5)
        k2 = _cache_key("RAG", 2022, None, None, 5)
        assert k1 == k2

    def test_key_changes_with_year_start(self):
        from tools.semantic_scholar import _cache_key
        assert _cache_key("RAG", 2022, None, None, 5) != _cache_key("RAG", 2023, None, None, 5)

    def test_key_changes_with_limit(self):
        from tools.semantic_scholar import _cache_key
        assert _cache_key("RAG", None, None, None, 3) != _cache_key("RAG", None, None, None, 5)

    def test_key_changes_with_field(self):
        from tools.semantic_scholar import _cache_key
        assert _cache_key("RAG", None, None, "Medicine", 5) != _cache_key("RAG", None, None, None, 5)

    def test_key_has_correct_prefix(self):
        from tools.semantic_scholar import _cache_key, CACHE_KEY_PREFIX
        key = _cache_key("anything", None, None, None, 5)
        assert key.startswith(CACHE_KEY_PREFIX)

    def test_key_matches_expected_md5(self):
        """Pin the exact key so a future refactor can't silently break it."""
        from tools.semantic_scholar import _cache_key, CACHE_KEY_PREFIX
        raw     = "retrieval augmented generation language models|2022|None|None|3"
        digest  = hashlib.md5(raw.encode()).hexdigest()
        expected = f"{CACHE_KEY_PREFIX}{digest}"
        assert _cache_key("retrieval augmented generation language models", 2022, None, None, 3) == expected


# ============================================================
# 6. LANGCHAIN TOOL INTERFACE
# ============================================================

class TestLangChainTool:

    @patch("tools.semantic_scholar.httpx.Client")
    def test_tool_returns_string(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        from tools.semantic_scholar import semantic_scholar_tool
        result = semantic_scholar_tool.invoke({"query": "transformers", "limit": 1})
        assert isinstance(result, str)

    @patch("tools.semantic_scholar.httpx.Client")
    def test_tool_output_contains_title_and_doi(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        from tools.semantic_scholar import semantic_scholar_tool
        result = semantic_scholar_tool.invoke({"query": "transformers", "limit": 1})
        assert "Attention Is All You Need" in result
        assert "10.48550/arXiv.1706.03762" in result

    @patch("tools.semantic_scholar.httpx.Client")
    def test_tool_truncates_long_author_list(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http(FAKE_API_RESPONSE)
        from tools.semantic_scholar import semantic_scholar_tool
        result = semantic_scholar_tool.invoke({"query": "transformers", "limit": 1})
        # FAKE_PAPER has 5 authors — should show 3 + "et al."
        assert "et al." in result

    @patch("tools.semantic_scholar.httpx.Client")
    def test_tool_returns_no_results_message_on_empty(self, mock_client_cls):
        mock_client_cls.return_value = _make_mock_http({"data": []})
        from tools.semantic_scholar import semantic_scholar_tool
        result = semantic_scholar_tool.invoke({"query": "xyzzy impossible query", "limit": 1})
        assert "No results found" in result

    def test_tool_name_and_description(self):
        from tools.semantic_scholar import semantic_scholar_tool
        assert semantic_scholar_tool.name == "semantic_scholar_search"
        assert "200M" in semantic_scholar_tool.description