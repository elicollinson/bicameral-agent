"""Loaders for harder external benchmark datasets (Issue #42).

Maps two external benchmarks into the existing :class:`ResearchQATask` shape to
give the evaluation pool real headroom above the saturated ``research_qa`` pool
(see the RCA in issue #41):

- **FRAMES** (``google/frames-benchmark``, Apache-2.0): multi-hop factual QA
  requiring synthesis across several Wikipedia articles -> mapped to ``hard``.
- **CREPE** (false-presupposition QA, arXiv:2211.17257): questions built on a
  false premise, with the presupposition and its correction annotated ->
  mapped to ``tricky``.

The raw data is **not redistributed** in this repository. :func:`build_hard_benchmark`
pulls subsets from upstream into a local, git-ignored cache; the pure
``*_row_to_task`` mappers are import-safe and unit-tested offline against a
synthetic fixture. See ``docs/hard_benchmark.md`` for sources, licenses, and
attribution, and for the fallback datasets to watch for.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .dataset import ResearchQADataset, ResearchQATask, TaskDifficulty, TaskSplit

FRAMES_DATASET = "google/frames-benchmark"
# tasksource re-host of the false-presupposition CREPE (arXiv:2211.17257);
# carries the presuppositions/corrections annotations we map from.
CREPE_DATASET = "tasksource/CREPE"

_HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
# Anchored to the repo root (this file lives at src/bicameral_agent/) so the
# cache resolves to the same place regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _REPO_ROOT / "data" / "external" / "hard_benchmark.json"
_USER_AGENT = "bicameral-agent/0.1 (research eval; issue-42)"

# The HF datasets-server rate-limits routinely; retry transient failures
# with exponential backoff before giving up.
_MAX_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 2.0
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


# --- pure mappers (offline-testable) ----------------------------------------

def frames_row_to_task(row: dict, index: int) -> ResearchQATask:
    """Map one raw FRAMES row into a ``hard`` ResearchQATask.

    FRAMES ships no rubric, so we synthesize one anchored on the gold answer
    and the multi-hop reasoning the question demands.
    """
    question = (row.get("Prompt") or "").strip()
    answer = (row.get("Answer") or "").strip()
    rubric = (
        f"5: States the correct answer ('{answer}') and shows the multi-hop "
        "reasoning linking the required facts. 4: Correct answer with thin "
        "justification. 3: Partially correct, or correct answer with no "
        "reasoning. 2: Relevant facts but wrong final answer. 1: Incorrect."
    )
    return ResearchQATask(
        task_id=f"frames_hard_{index:03d}",
        difficulty=TaskDifficulty.HARD,
        split=TaskSplit.EVAL,
        question=question,
        gold_answer=answer,
        scoring_rubric=rubric,
    )


def crepe_row_to_task(row: dict, index: int) -> ResearchQATask:
    """Map one raw CREPE false-presupposition row into a ``tricky`` task.

    The annotated presupposition(s) become ``known_assumptions`` and the
    annotated correction(s) become the gold answer.
    """
    question = (row.get("question") or "").strip()
    presups = [p.strip() for p in (row.get("presuppositions") or []) if p and p.strip()]
    corrections = [c.strip() for c in (row.get("corrections") or []) if c and c.strip()]
    gold = " ".join(corrections)
    rubric = (
        "5: Flags that the question rests on a false presupposition AND states "
        "the correction. 4: Identifies the false premise but corrects it "
        "vaguely. 3: Hedges or only partially questions the premise. 2: Answers "
        "as asked without noticing the false premise. 1: Affirms the false "
        "premise."
    )
    return ResearchQATask(
        task_id=f"crepe_tricky_{index:03d}",
        difficulty=TaskDifficulty.TRICKY,
        split=TaskSplit.EVAL,
        question=question,
        gold_answer=gold,
        scoring_rubric=rubric,
        known_assumptions=presups,
    )


# --- upstream fetch ---------------------------------------------------------

def _http_get_json(url: str) -> dict:
    """GET *url* as JSON, retrying transient (rate-limit / 5xx / network) errors."""
    last_err: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(_RETRY_BASE_DELAY_S * 2 ** (attempt - 1))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted hosts)
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            if err.code not in _RETRYABLE_HTTP_CODES:
                raise
            last_err = err
        except urllib.error.URLError as err:
            last_err = err
    raise RuntimeError(
        f"Giving up on {url} after {_MAX_ATTEMPTS} attempts: {last_err}"
    ) from last_err


def _fetch_page(dataset: str, split: str, offset: int, length: int) -> list[dict]:
    url = (
        f"{_HF_ROWS_ENDPOINT}?dataset={urllib.parse.quote(dataset)}"
        f"&config=default&split={split}&offset={offset}&length={length}"
    )
    payload = _http_get_json(url)
    if "rows" not in payload:
        # An error payload (e.g. rate-limit body) must not read as an empty
        # terminal page — that silently truncates the benchmark.
        raise RuntimeError(
            f"Unexpected datasets-server payload for {dataset} at offset {offset}: "
            f"{payload.get('error', payload)!r}"
        )
    return [row["row"] for row in payload["rows"]]


def fetch_frames(limit: int = 100, split: str = "test") -> list[ResearchQATask]:
    """Pull a subset of FRAMES rows and map them to ``hard`` tasks."""
    tasks: list[ResearchQATask] = []
    offset = 0
    while len(tasks) < limit:
        page = _fetch_page(FRAMES_DATASET, split, offset, min(100, limit - len(tasks)))
        if not page:
            break
        for raw in page:
            if (raw.get("Prompt") or "").strip() and (raw.get("Answer") or "").strip():
                tasks.append(frames_row_to_task(raw, len(tasks) + 1))
                if len(tasks) == limit:
                    break
        offset += len(page)
    return tasks


def fetch_crepe(limit: int = 60, split: str = "test") -> list[ResearchQATask]:
    """Scan CREPE and map false-presupposition rows to ``tricky`` tasks."""
    tasks: list[ResearchQATask] = []
    offset = 0
    while len(tasks) < limit:
        page = _fetch_page(CREPE_DATASET, split, offset, 100)
        if not page:
            break
        for raw in page:
            presups = [p for p in (raw.get("presuppositions") or []) if p and p.strip()]
            corrections = [c for c in (raw.get("corrections") or []) if c and c.strip()]
            if presups and corrections:
                tasks.append(crepe_row_to_task(raw, len(tasks) + 1))
                if len(tasks) == limit:
                    break
        offset += len(page)
    return tasks


def build_hard_benchmark(
    frames_n: int = 100,
    crepe_n: int = 60,
    cache_path: Path = _DEFAULT_CACHE,
) -> list[ResearchQATask]:
    """Fetch subsets from upstream and write a normalized JSON cache.

    The cache file is git-ignored; it is the artifact :func:`load_hard_benchmark`
    reads. Returns the combined task list.

    Raises:
        RuntimeError: If either fetch yields fewer tasks than requested, so a
            partial upstream response cannot produce a silent short benchmark.
    """
    frames = fetch_frames(frames_n)
    crepe = fetch_crepe(crepe_n)
    if len(frames) != frames_n or len(crepe) != crepe_n:
        raise RuntimeError(
            f"Fetched {len(frames)}/{frames_n} FRAMES and {len(crepe)}/{crepe_n} "
            "CREPE tasks; refusing to write a short benchmark cache."
        )
    tasks = frames + crepe
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([t.model_dump(mode="json") for t in tasks], indent=2),
        encoding="utf-8",
    )
    return tasks


# --- read path --------------------------------------------------------------

def load_hard_benchmark(cache_path: Path = _DEFAULT_CACHE) -> ResearchQADataset:
    """Load the cached hard-benchmark tasks into a :class:`ResearchQADataset`.

    Raises ``FileNotFoundError`` with fetch instructions if the cache is absent
    (the data is not committed to the repo).
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Hard benchmark cache not found at {cache_path}. The data is not "
            "committed to the repo -- run `python scripts/fetch_hard_benchmark.py` "
            "first. See docs/hard_benchmark.md for sources and attribution."
        )
    return ResearchQADataset.from_path(cache_path)
