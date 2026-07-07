"""Stdlib-only pager for the Hugging Face datasets-server rows API.

Extracted from ``hard_benchmark.py`` (Issue #56) so every HF-backed dataset
adapter shares one hardened fetch path: retries on rate-limit/5xx/network
errors with exponential backoff, and refuses to treat an error payload as an
empty terminal page (which would silently truncate a benchmark).

No HF ``datasets``/``pandas`` dependency, per repo policy -- plain ``urllib``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "bicameral-agent/0.1 (research eval; issue-42)"

# The HF datasets-server rate-limits routinely; retry transient failures
# with exponential backoff before giving up.
MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 2.0
RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


def http_get_json(url: str) -> dict:
    """GET *url* as JSON, retrying transient (rate-limit / 5xx / network) errors."""
    last_err: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(RETRY_BASE_DELAY_S * 2 ** (attempt - 1))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted hosts)
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            if err.code not in RETRYABLE_HTTP_CODES:
                raise
            last_err = err
        except urllib.error.URLError as err:
            last_err = err
    raise RuntimeError(
        f"Giving up on {url} after {MAX_ATTEMPTS} attempts: {last_err}"
    ) from last_err


def fetch_page(dataset: str, split: str, offset: int, length: int) -> list[dict]:
    """Fetch one page of raw rows from the datasets-server rows API."""
    url = (
        f"{HF_ROWS_ENDPOINT}?dataset={urllib.parse.quote(dataset)}"
        f"&config=default&split={split}&offset={offset}&length={length}"
    )
    payload = http_get_json(url)
    if "rows" not in payload:
        # An error payload (e.g. rate-limit body) must not read as an empty
        # terminal page — that silently truncates the benchmark.
        raise RuntimeError(
            f"Unexpected datasets-server payload for {dataset} at offset {offset}: "
            f"{payload.get('error', payload)!r}"
        )
    return [row["row"] for row in payload["rows"]]
