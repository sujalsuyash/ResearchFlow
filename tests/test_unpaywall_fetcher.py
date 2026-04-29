"""
tests/test_unpaywall_fetcher.py

Comprehensive unit test suite for tools/unpaywall_fetcher.py.
No network access and no Redis required — everything is mocked.

Mock strategy
-------------
- httpx.Client is patched at the module level so _throttled_get's real retry
  loop executes during 429 tests (per the architectural requirement).
- _throttled_get itself is NEVER mocked — the retry, backoff, and 404-raise
  logic must be exercised through real code paths.
- _redis is patched per-test class to test Redis vs. fallback branches.
"""

import hashlib
import json
import os
import sys
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

# Project root on sys.path so "from tools.xxx import ..." works from any cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Shared fake API responses
# ---------------------------------------------------------------------------

FAKE_OA_RESPONSE = {
    "doi":    "10.1038/nature12373",
    "title":  "Quantum entanglement at a macroscopic level",
    "is_oa":  True,
    "oa_status": "gold",
    "best_oa_location": {
        "url_for_pdf":          "https://example.com/paper.pdf",
        "url_for_landing_page": "https://example.com/paper",
        "host_type":            "publisher",
        "license":              "cc-by",
    },
}

FAKE_CLOSED_RESPONSE = {
    "doi":               "10.9999/closed.paper",
    "title":             "Paywalled Physics",
    "is_oa":             False,
    "oa_status":         "closed",
    "best_oa_location":  None,       # None is the real Unpaywall value for closed papers
}

FAKE_GREEN_NO_PDF = {
    "doi":    "10.5555/green.no.pdf",
    "title":  "Green Paper Without PDF",
    "is_oa":  True,
    "oa_status": "green",
    "best_oa_location": {
        "url_for_pdf":          None,
        "url_for_landing_page": "https://repository.example.com/paper",
    },
}

TEST_DOI = "10.1038/nature12373"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_mock(response_data: dict) -> MagicMock:
    """
    Return a mock for httpx.Client().__enter__() whose .get() returns a
    successful response with response_data as the JSON body.
    """
    ok_resp = MagicMock()
    ok_resp.json.return_value = response_data
    ok_resp.raise_for_status = MagicMock()   # no-op — simulates 200

    http = MagicMock()
    http.__enter__ = MagicMock(return_value=http)
    http.__exit__  = MagicMock(return_value=False)
    http.get.return_value = ok_resp
    return http


def _make_http_error(status_code: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers     = headers or {}
    return httpx.HTTPStatusError(str(status_code), request=MagicMock(), response=resp)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def force_no_redis(monkeypatch):
    """Default to Redis-unavailable for every test."""
    monkeypatch.setattr("tools.unpaywall_fetcher._redis", None)


@pytest.fixture(autouse=True)
def clean_cache():
    from tools.unpaywall_fetcher import clear_cache
    clear_cache()
    yield
    clear_cache()


# ============================================================
# 1. RESPONSE PARSING
# ============================================================

class TestParsing:

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_all_fields_extracted_correctly(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall

        result = fetch_unpaywall(TEST_DOI)

        assert result["doi"]         == TEST_DOI
        assert result["title"]       == "Quantum entanglement at a macroscopic level"
        assert result["is_oa"]       is True
        assert result["oa_status"]   == "gold"
        assert result["pdf_url"]     == "https://example.com/paper.pdf"
        assert result["landing_url"] == "https://example.com/paper"
        assert "error"               not in result

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_closed_access_paper_returns_none_urls(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_CLOSED_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall

        result = fetch_unpaywall("10.9999/closed.paper")

        assert result["is_oa"]       is False
        assert result["oa_status"]   == "closed"
        assert result["pdf_url"]     is None
        assert result["landing_url"] is None

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_green_oa_with_no_pdf_url(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_GREEN_NO_PDF)
        from tools.unpaywall_fetcher import fetch_unpaywall

        result = fetch_unpaywall("10.5555/green.no.pdf")

        assert result["is_oa"]       is True
        assert result["oa_status"]   == "green"
        assert result["pdf_url"]     is None
        assert result["landing_url"] == "https://repository.example.com/paper"

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_is_oa_coerced_to_bool(self, mock_client_cls):
        """Unpaywall sometimes returns 0/1 — ensure we always get a proper bool."""
        truthy = {**FAKE_OA_RESPONSE, "is_oa": 1}
        mock_client_cls.return_value = _make_http_mock(truthy)
        from tools.unpaywall_fetcher import fetch_unpaywall

        result = fetch_unpaywall(TEST_DOI)
        assert result["is_oa"] is True
        assert type(result["is_oa"]) is bool

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_doi_whitespace_is_stripped(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall

        result = fetch_unpaywall("  10.1038/nature12373  ")
        assert result["doi"] == TEST_DOI    # cache key and result use stripped DOI


# ============================================================
# 2. 404 GRACEFUL DEGRADATION
# ============================================================

class TestNotFound:

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_404_returns_not_found_dict_not_raise(self, mock_client_cls):
        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        http.get.side_effect = _make_http_error(404)
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall
        result = fetch_unpaywall("10.9999/does.not.exist")

        assert result["is_oa"]     is False
        assert result["oa_status"] == "not_found"
        assert result["pdf_url"]   is None
        assert "error"             in result
        assert "not found"         in result["error"].lower()

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_404_does_not_write_to_cache(self, mock_client_cls):
        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        http.get.side_effect = _make_http_error(404)
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall, cache_stats
        fetch_unpaywall("10.9999/does.not.exist")

        assert cache_stats()["entries"] == 0

    def test_empty_doi_returns_not_found_without_api_call(self):
        from tools.unpaywall_fetcher import fetch_unpaywall
        result = fetch_unpaywall("")
        assert result["oa_status"] == "not_found"
        assert result["is_oa"]     is False

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_500_raises_immediately(self, mock_client_cls):
        """Non-404 HTTP errors must bubble up — they are not handled gracefully."""
        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        http.get.side_effect = _make_http_error(500)
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            fetch_unpaywall(TEST_DOI)
        assert exc_info.value.response.status_code == 500


# ============================================================
# 3. 429 RETRY & EXPONENTIAL BACKOFF
#    httpx.Client is mocked directly so _throttled_get's real loop runs
# ============================================================

class TestRetry:

    @patch("tools.unpaywall_fetcher.time.sleep")
    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_429_retries_and_eventually_succeeds(self, mock_client_cls, mock_sleep):
        ok_resp = MagicMock()
        ok_resp.json.return_value = FAKE_OA_RESPONSE
        ok_resp.raise_for_status  = MagicMock()

        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        # Fail twice with 429, succeed on third attempt
        http.get.side_effect = [
            _make_http_error(429),
            _make_http_error(429),
            ok_resp,
        ]
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall, RETRY_BASE_DELAY
        result = fetch_unpaywall(TEST_DOI)

        assert result["is_oa"]  is True
        assert mock_client_cls.call_count == 3   # three httpx.Client() instantiations

        # Filter out sub-second token-bucket sleeps; only retry delays remain
        retry_sleeps = [c[0][0] for c in mock_sleep.call_args_list if c[0][0] >= 1]
        assert RETRY_BASE_DELAY in retry_sleeps           # 10s first backoff
        assert RETRY_BASE_DELAY * 2 in retry_sleeps       # 20s second backoff

    @patch("tools.unpaywall_fetcher.time.sleep")
    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_retry_after_header_overrides_exponential(self, mock_client_cls, mock_sleep):
        ok_resp = MagicMock()
        ok_resp.json.return_value = FAKE_OA_RESPONSE
        ok_resp.raise_for_status  = MagicMock()

        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        http.get.side_effect = [
            _make_http_error(429, headers={"Retry-After": "45"}),
            ok_resp,
        ]
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall
        fetch_unpaywall(TEST_DOI)

        mock_sleep.assert_any_call(45)   # server-specified delay honoured

    @patch("tools.unpaywall_fetcher.time.sleep")
    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_persistent_429_raises_after_all_retries_exhausted(self, mock_client_cls, mock_sleep):
        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        http.get.side_effect = _make_http_error(429)   # always 429
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall, RETRY_ATTEMPTS
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            fetch_unpaywall(TEST_DOI)

        assert exc_info.value.response.status_code == 429
        assert mock_client_cls.call_count == RETRY_ATTEMPTS

    @patch("tools.unpaywall_fetcher.time.sleep")
    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_non_429_non_404_raises_without_retry(self, mock_client_cls, mock_sleep):
        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        http.get.side_effect = _make_http_error(403)   # Forbidden — not retried
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall
        with pytest.raises(httpx.HTTPStatusError):
            fetch_unpaywall(TEST_DOI)

        assert mock_client_cls.call_count == 1   # raised immediately, no retry

    @patch("tools.unpaywall_fetcher.time.sleep")
    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_backoff_delay_is_capped_at_max(self, mock_client_cls, mock_sleep):
        """Delay must never exceed RETRY_MAX_DELAY regardless of attempt count."""
        ok_resp = MagicMock()
        ok_resp.json.return_value = FAKE_OA_RESPONSE
        ok_resp.raise_for_status  = MagicMock()

        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        # 3 failures then success (exactly RETRY_ATTEMPTS = 4 total calls)
        http.get.side_effect = [
            _make_http_error(429),
            _make_http_error(429),
            _make_http_error(429),
            ok_resp,
        ]
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import fetch_unpaywall, RETRY_MAX_DELAY
        fetch_unpaywall(TEST_DOI)

        retry_sleeps = [c[0][0] for c in mock_sleep.call_args_list if c[0][0] >= 1]
        assert all(s <= RETRY_MAX_DELAY for s in retry_sleeps)


# ============================================================
# 4. FALLBACK CACHE
# ============================================================

class TestFallbackCache:

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_second_call_hits_fallback_cache_not_api(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall

        r1 = fetch_unpaywall(TEST_DOI)
        r2 = fetch_unpaywall(TEST_DOI)

        assert r1 == r2
        assert mock_client_cls.call_count == 1   # API hit only once

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_different_dois_are_separate_cache_entries(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall, cache_stats

        fetch_unpaywall("10.1111/aaa")
        mock_client_cls.return_value = _make_http_mock(FAKE_CLOSED_RESPONSE)
        fetch_unpaywall("10.2222/bbb")

        assert cache_stats()["entries"] == 2

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_clear_cache_empties_fallback(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall, clear_cache, cache_stats

        fetch_unpaywall(TEST_DOI)
        assert cache_stats()["entries"] == 1
        clear_cache()
        assert cache_stats()["entries"] == 0

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_cache_stores_deep_equal_result(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall

        r1 = fetch_unpaywall(TEST_DOI)
        r2 = fetch_unpaywall(TEST_DOI)    # from cache
        assert r1 == r2


# ============================================================
# 5. REDIS CACHE
# ============================================================

class TestRedisCache:

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_redis_hit_skips_api(self, mock_client_cls):
        cached_result = {
            "doi": TEST_DOI, "title": "Cached", "is_oa": True,
            "oa_status": "gold", "pdf_url": "https://cached.pdf", "landing_url": None,
        }
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached_result)

        with patch("tools.unpaywall_fetcher._redis", mock_redis):
            from tools.unpaywall_fetcher import fetch_unpaywall
            result = fetch_unpaywall(TEST_DOI)

        assert result == cached_result
        mock_client_cls.assert_not_called()
        mock_redis.get.assert_called_once()

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_redis_miss_calls_api_and_writes_with_correct_ttl(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        mock_redis = MagicMock()
        mock_redis.get.return_value = None    # cache miss

        with patch("tools.unpaywall_fetcher._redis", mock_redis):
            from tools.unpaywall_fetcher import fetch_unpaywall, CACHE_TTL_SECONDS
            fetch_unpaywall(TEST_DOI)

        mock_redis.setex.assert_called_once()
        _, ttl, payload = mock_redis.setex.call_args[0]
        assert ttl == CACHE_TTL_SECONDS                         # 604800
        assert json.loads(payload)["doi"] == TEST_DOI
        assert json.loads(payload)["is_oa"] is True

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_redis_get_failure_falls_through_to_api(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("connection reset")

        with patch("tools.unpaywall_fetcher._redis", mock_redis):
            from tools.unpaywall_fetcher import fetch_unpaywall
            result = fetch_unpaywall(TEST_DOI)

        assert result["is_oa"] is True
        mock_client_cls.assert_called_once()

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_redis_set_failure_still_returns_result(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        mock_redis = MagicMock()
        mock_redis.get.return_value  = None
        mock_redis.setex.side_effect = Exception("write timeout")

        with patch("tools.unpaywall_fetcher._redis", mock_redis):
            from tools.unpaywall_fetcher import fetch_unpaywall
            result = fetch_unpaywall(TEST_DOI)

        assert result["is_oa"] is True    # result still returned despite SET failure

    def test_cache_stats_reports_redis_backend(self):
        mock_redis = MagicMock()
        mock_redis.scan.return_value = (0, ["researchflow:unpaywall:abc123"])
        mock_redis.ttl.return_value  = 300000

        with patch("tools.unpaywall_fetcher._redis", mock_redis):
            from tools.unpaywall_fetcher import cache_stats, CACHE_TTL_SECONDS
            stats = cache_stats()

        assert stats["backend"]                  == "redis"
        assert stats["entries"]                  == 1
        assert stats["ttl_seconds"]              == CACHE_TTL_SECONDS
        assert stats["sample_key_ttl_remaining"] == 300000


# ============================================================
# 6. CACHE KEY CORRECTNESS
# ============================================================

class TestCacheKey:

    def test_key_is_deterministic(self):
        from tools.unpaywall_fetcher import _cache_key
        assert _cache_key(TEST_DOI) == _cache_key(TEST_DOI)

    def test_different_dois_produce_different_keys(self):
        from tools.unpaywall_fetcher import _cache_key
        assert _cache_key("10.1111/aaa") != _cache_key("10.2222/bbb")

    def test_key_has_correct_prefix(self):
        from tools.unpaywall_fetcher import _cache_key, CACHE_KEY_PREFIX
        assert _cache_key(TEST_DOI).startswith(CACHE_KEY_PREFIX)

    def test_key_matches_expected_md5(self):
        from tools.unpaywall_fetcher import _cache_key, CACHE_KEY_PREFIX
        expected = CACHE_KEY_PREFIX + hashlib.md5(TEST_DOI.encode()).hexdigest()
        assert _cache_key(TEST_DOI) == expected

    def test_key_prefix_unique_across_tools(self):
        """All four tool prefixes must be distinct to coexist in the same Redis db."""
        from tools.unpaywall_fetcher   import CACHE_KEY_PREFIX as UW
        from tools.semantic_scholar    import CACHE_KEY_PREFIX as SS
        from tools.arxiv_search        import CACHE_KEY_PREFIX as AX
        prefixes = {UW, SS, AX}
        assert len(prefixes) == 3    # all distinct


# ============================================================
# 7. LANGCHAIN TOOL INTERFACE
# ============================================================

class TestLangChainTool:

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_tool_returns_string(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import unpaywall_tool
        result = unpaywall_tool.invoke({"doi": TEST_DOI})
        assert isinstance(result, str)

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_tool_output_contains_doi_and_status(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import unpaywall_tool
        output = unpaywall_tool.invoke({"doi": TEST_DOI})
        assert TEST_DOI                          in output
        assert "gold"                            in output.lower()
        assert "https://example.com/paper.pdf"  in output

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_tool_output_contains_pdf_url(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import unpaywall_tool
        output = unpaywall_tool.invoke({"doi": TEST_DOI})
        assert "https://example.com/paper.pdf" in output

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_tool_closed_access_does_not_show_pdf(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_CLOSED_RESPONSE)
        from tools.unpaywall_fetcher import unpaywall_tool
        output = unpaywall_tool.invoke({"doi": "10.9999/closed.paper"})
        assert "Not available" in output or "closed" in output.lower()

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_tool_not_found_returns_graceful_message(self, mock_client_cls):
        http = MagicMock()
        http.__enter__ = MagicMock(return_value=http)
        http.__exit__  = MagicMock(return_value=False)
        http.get.side_effect = _make_http_error(404)
        mock_client_cls.return_value = http

        from tools.unpaywall_fetcher import unpaywall_tool
        output = unpaywall_tool.invoke({"doi": "10.9999/ghost"})
        assert "Not found" in output or "not found" in output.lower()
        assert "❌" in output

    def test_tool_name_and_description(self):
        from tools.unpaywall_fetcher import unpaywall_tool
        assert unpaywall_tool.name == "unpaywall_lookup"
        assert "DOI"     in unpaywall_tool.description
        assert "PDF"     in unpaywall_tool.description
        assert "arXiv"   in unpaywall_tool.description   # skip-arXiv hint is present


# ============================================================
# 8. URL EXTRACTION PRIORITY
# ============================================================

class TestUrlExtraction:

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_pdf_url_takes_priority_when_both_present(self, mock_client_cls):
        """pdf_url must be populated even when landing_url also exists."""
        mock_client_cls.return_value = _make_http_mock(FAKE_OA_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall
        result = fetch_unpaywall(TEST_DOI)
        assert result["pdf_url"]     == "https://example.com/paper.pdf"
        assert result["landing_url"] == "https://example.com/paper"

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_landing_url_present_when_pdf_missing(self, mock_client_cls):
        """When url_for_pdf is None, landing_url should still be populated."""
        mock_client_cls.return_value = _make_http_mock(FAKE_GREEN_NO_PDF)
        from tools.unpaywall_fetcher import fetch_unpaywall
        result = fetch_unpaywall("10.5555/green.no.pdf")
        assert result["pdf_url"]     is None
        assert result["landing_url"] == "https://repository.example.com/paper"

    @patch("tools.unpaywall_fetcher.httpx.Client")
    def test_both_urls_none_for_closed_paper(self, mock_client_cls):
        mock_client_cls.return_value = _make_http_mock(FAKE_CLOSED_RESPONSE)
        from tools.unpaywall_fetcher import fetch_unpaywall
        result = fetch_unpaywall("10.9999/closed.paper")
        assert result["pdf_url"]     is None
        assert result["landing_url"] is None