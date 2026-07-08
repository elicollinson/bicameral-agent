"""Tests for the Brave Web Search provider (issue #100).

All transport is mocked (offline/CI-safe) except the final live smoke test,
which is skipped unless ``BRAVE_API_KEY`` is set.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
from unittest.mock import patch

import pytest

from bicameral_agent.brave_search import BraveSearchProvider, _RateLimiter
from bicameral_agent.gap_scanner import SearchProvider, SearchResult
from bicameral_agent.llm_output import count_degradations


def _payload(num_results: int = 2, extra_snippets: bool = False) -> dict:
    results = []
    for i in range(num_results):
        item = {
            "title": f"Result {i}",
            "url": f"https://example.com/{i}",
            "description": f"Description {i}.",
        }
        if extra_snippets:
            item["extra_snippets"] = [f"Extra A{i}.", f"Extra B{i}.", f"Extra C{i}."]
        results.append(item)
    return {"web": {"results": results}}


class _FakeResponse:
    """Minimal stand-in for urllib.request.urlopen's context manager."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.search.brave.com/res/v1/web/search",
        code,
        "error",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )


def _provider(**kwargs) -> BraveSearchProvider:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("min_request_interval_s", 0.0)
    return BraveSearchProvider(**kwargs)


class TestConstruction:
    def test_satisfies_search_provider_protocol(self):
        assert isinstance(_provider(), SearchProvider)

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "env-key")
        provider = BraveSearchProvider(min_request_interval_s=0.0)
        assert provider._api_key == "env-key"

    def test_explicit_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "env-key")
        assert _provider(api_key="explicit")._api_key == "explicit"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            BraveSearchProvider()


class TestRequestShape:
    def test_url_headers_and_params(self):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(_payload())) as mock_open:
            _provider().search("fusion ignition NIF", max_results=5)

        request = mock_open.call_args.args[0]
        url = urllib.parse.urlparse(request.full_url)
        assert url.scheme == "https"
        assert url.netloc == "api.search.brave.com"
        assert url.path == "/res/v1/web/search"
        params = urllib.parse.parse_qs(url.query)
        assert params["q"] == ["fusion ignition NIF"]
        assert params["count"] == ["5"]
        assert params["extra_snippets"] == ["true"]
        assert request.get_header("X-subscription-token") == "test-key"
        assert mock_open.call_args.kwargs["timeout"] == pytest.approx(10.0)

    def test_count_clamped_to_brave_cap(self):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(_payload())) as mock_open:
            _provider().search("q", max_results=50)
        url = urllib.parse.urlparse(mock_open.call_args.args[0].full_url)
        assert urllib.parse.parse_qs(url.query)["count"] == ["20"]


class TestResponseMapping:
    def test_maps_web_results(self):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(_payload(2))):
            results = _provider().search("q", max_results=3)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].title == "Result 0"
        assert results[0].snippet == "Description 0."
        assert results[0].source == "https://example.com/0"
        # Brave returns rank order without scores; rank is encoded descending.
        assert results[0].relevance_score > results[1].relevance_score

    def test_snippet_includes_capped_extra_snippets(self):
        payload = _payload(1, extra_snippets=True)
        with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
            results = _provider().search("q")
        assert results[0].snippet == "Description 0. Extra A0. Extra B0."

    def test_respects_max_results(self):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(_payload(5))):
            results = _provider().search("q", max_results=2)
        assert len(results) == 2

    def test_empty_and_malformed_payloads_yield_no_results(self):
        for payload in ({}, {"web": {}}, {"web": {"results": [42, {"title": "no snippet"}]}}):
            with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
                assert _provider().search("q") == []


class TestRetryAndDegradation:
    def test_retries_transient_429_then_succeeds(self):
        side_effects = [_http_error(429), _FakeResponse(_payload(1))]
        with patch("urllib.request.urlopen", side_effect=side_effects) as mock_open, \
                patch("bicameral_agent.brave_search.time.sleep") as mock_sleep:
            results = _provider().search("q")
        assert len(results) == 1
        assert mock_open.call_count == 2
        assert mock_sleep.called  # backoff before the retry

    def test_non_retryable_http_error_degrades_without_retry(self, caplog):
        with patch("urllib.request.urlopen", side_effect=_http_error(401)) as mock_open, \
                caplog.at_level("WARNING"):
            results = _provider().search("q")
        assert results == []
        assert mock_open.call_count == 1
        assert any("degrading" in r.message for r in caplog.records)

    def test_outage_degrades_to_empty_and_counts_degradation(self, caplog):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ) as mock_open, patch("bicameral_agent.brave_search.time.sleep"), \
                caplog.at_level("WARNING"), count_degradations() as counter:
            results = _provider().search("q")
        assert results == []
        assert mock_open.call_count == 3  # all attempts exhausted
        assert counter.counts["BraveSearchProvider"] == 1
        assert any("degrading" in r.message for r in caplog.records)


class TestThrottle:
    def test_zero_interval_is_noop(self):
        limiter = _RateLimiter(0.0)
        start = time.monotonic()
        for _ in range(100):
            limiter.wait()
        assert time.monotonic() - start < 0.5

    def test_concurrent_searches_are_spaced(self):
        """N threads calling search() hit the transport >= interval apart."""
        interval = 0.05
        provider = _provider(min_request_interval_s=interval)
        send_times: list[float] = []
        times_lock = threading.Lock()

        def fake_urlopen(request, timeout):
            with times_lock:
                send_times.append(time.monotonic())
            return _FakeResponse(_payload(1))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            threads = [
                threading.Thread(target=provider.search, args=("q",))
                for _ in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(send_times) == 5
        send_times.sort()
        gaps = [b - a for a, b in zip(send_times, send_times[1:])]
        # Allow small scheduling jitter below the nominal interval.
        assert all(gap >= interval * 0.8 for gap in gaps), gaps


@pytest.mark.skipif(
    not os.environ.get("BRAVE_API_KEY"),
    reason="BRAVE_API_KEY not set; skipping live Brave Web Search smoke test",
)
class TestLiveSmoke:
    def test_live_search_returns_results(self):
        provider = BraveSearchProvider()
        results = provider.search("Python programming language", max_results=2)
        assert 1 <= len(results) <= 2
        for result in results:
            assert result.title
            assert result.snippet
            assert result.source.startswith("http")
