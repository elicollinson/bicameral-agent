"""Brave Web Search backend for the gap scanner's SearchProvider protocol (issue #100).

The default ``MockSearchProvider`` keyword-matches 20 canned snippets, so the
subconscious can only rearrange the answerer's own knowledge. This provider
performs real web search, the first configuration where the tool arm can hold
information the answerer lacks (FRAMES-style multi-hop retrieval tasks).

API contract (verified against Brave's Web Search API docs, 2026-07-08):
``GET https://api.search.brave.com/res/v1/web/search?q=...&count=N`` with the
key in the ``X-Subscription-Token`` header; results arrive rank-ordered in
``web.results[]`` with ``title``/``description``/``url`` and, when requested
via ``extra_snippets=true`` (available on the free AI plan), an
``extra_snippets`` array of additional page excerpts. ``count`` caps at 20.

Transport conventions match the repo: stdlib ``urllib`` (like
``ollama_cloud.py``), retry on 429/5xx/network errors with exponential backoff
(mirroring ``eval_datasets.hf_fetch``), and graceful degradation -- a search
outage returns ``[]`` with a warning and a degradation count (issue #47
philosophy) rather than killing the episode.

A client-side throttle spaces requests at the free tier's ~1 req/s. It is
thread-safe for concurrent episodes (``--parallel-episodes 10``): each caller
atomically reserves the next send slot under a lock, then sleeps outside it,
so callers queue up at 1 req/s without blocking each other during network I/O.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from bicameral_agent.gap_scanner import SearchResult
from bicameral_agent.llm_output import report_degradation

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_USER_AGENT = "bicameral-agent/0.1 (research eval; issue-100)"

# Tighter than hf_fetch's 4x30s: the scanner runs inside a live episode turn,
# so a dead search backend must cost seconds, not minutes, before degrading.
_TIMEOUT_S = 10.0
_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 1.0
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

_MAX_COUNT = 20  # Brave's per-request result cap
_MAX_EXTRA_SNIPPETS = 2  # keep snippets QueueItem-sized


class BraveSearchError(Exception):
    """A Brave search request failed (non-retryably, or after all retries)."""


class _RateLimiter:
    """Thread-safe minimum-interval request spacer.

    ``wait()`` atomically reserves the next available send slot under a lock
    and sleeps outside it, so N concurrent callers serialize their requests
    at one per ``min_interval_s`` without holding the lock during the sleep
    or the subsequent network I/O.
    """

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._next_slot = 0.0  # monotonic time of the next free send slot

    def wait(self) -> None:
        """Block until this caller's reserved send slot arrives."""
        if self._min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(self._next_slot, now)
            self._next_slot = slot + self._min_interval_s
        if slot > now:
            time.sleep(slot - now)


class BraveSearchProvider:
    """Real web search via the Brave Web Search API.

    Conforms to the ``SearchProvider`` protocol. Thread-safe: the only
    mutable state is the rate limiter, which is lock-guarded.
    """

    def __init__(
        self,
        api_key: str | None = None,
        min_request_interval_s: float = 1.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required: pass api_key= or set BRAVE_API_KEY env var"
            )
        self._api_key = resolved_key
        self._throttle = _RateLimiter(min_request_interval_s)

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        """Search the web, degrading to ``[]`` on failure (never raises)."""
        try:
            payload = self._fetch(query, max_results)
        except BraveSearchError as err:
            # A search outage must not kill the episode: warn, count the
            # degradation, and let the scanner proceed with no results.
            logger.warning(
                "BraveSearchProvider: search failed for %r; degrading to no "
                "results: %s",
                query,
                err,
            )
            report_degradation("BraveSearchProvider")
            return []
        return _map_results(payload, max_results)

    def _fetch(self, query: str, max_results: int) -> dict[str, Any]:
        """One throttled GET with hf_fetch-style retry on transient errors."""
        params = urllib.parse.urlencode(
            {
                "q": query,
                "count": min(max(max_results, 1), _MAX_COUNT),
                "extra_snippets": "true",
            }
        )
        url = f"{_ENDPOINT}?{params}"
        last_err: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                time.sleep(_RETRY_BASE_DELAY_S * 2 ** (attempt - 1))
            self._throttle.wait()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": _USER_AGENT,
                    "X-Subscription-Token": self._api_key,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                if err.code not in _RETRYABLE_HTTP_CODES:
                    raise BraveSearchError(
                        f"HTTP {err.code} from Brave search: {err.reason}"
                    ) from err
                last_err = err
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionResetError,
                json.JSONDecodeError,
            ) as err:
                last_err = err
        raise BraveSearchError(
            f"giving up after {_MAX_ATTEMPTS} attempts: {last_err}"
        ) from last_err


def _map_results(payload: dict[str, Any], max_results: int) -> list[SearchResult]:
    """Map a Brave response payload to ``SearchResult`` items.

    Brave returns results rank-ordered without numeric scores, so
    ``relevance_score`` encodes the rank (1.0, 0.9, ...) to preserve the
    ordering downstream; the scanner's LLM ranking pass reassigns real
    relevance scores before anything reaches the queue.
    """
    raw = (payload.get("web") or {}).get("results") or []
    results: list[SearchResult] = []
    for item in raw:
        if len(results) == max_results:
            break
        if not isinstance(item, dict):
            continue
        description = item.get("description") or ""
        extras = [s for s in (item.get("extra_snippets") or []) if isinstance(s, str)]
        snippet = " ".join([description, *extras[:_MAX_EXTRA_SNIPPETS]]).strip()
        if not snippet:
            continue
        results.append(
            SearchResult(
                title=item.get("title") or "",
                snippet=snippet,
                relevance_score=round(max(0.1, 1.0 - 0.1 * len(results)), 3),
                source=item.get("url") or "",
            )
        )
    return results
