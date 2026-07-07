"""Schema + loader smoke tests for the harder benchmark integration (Issue #42).

Runs fully offline: the upstream mappers are exercised against synthetic raw
rows, and the loader against a committed author-owned fixture. No network and
no externally-licensed data are required.

Since Issue #56 the implementation lives in ``bicameral_agent.eval_datasets``;
everything here imports through the ``bicameral_agent.hard_benchmark`` shim to
prove the original surface still works, while monkeypatches target the modules
that now own the code (``hf_fetch``, ``frames``, ``crepe``).
"""

import urllib.error
from pathlib import Path

import pytest

from bicameral_agent.eval_datasets import crepe as crepe_mod
from bicameral_agent.eval_datasets import frames as frames_mod
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.dataset import ResearchQADataset, TaskDifficulty, TaskSplit
from bicameral_agent.hard_benchmark import (
    _DEFAULT_CACHE,
    build_hard_benchmark,
    crepe_row_to_task,
    fetch_crepe,
    fetch_frames,
    frames_row_to_task,
    load_hard_benchmark,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hard_benchmark_sample.json"


class TestFramesMapper:
    def test_maps_to_hard_eval_task(self):
        row = {
            "Prompt": "Who succeeded the monarch who reigned during the Great Fire of London?",
            "Answer": "James II",
            "reasoning_types": "Multiple constraints",
        }
        task = frames_row_to_task(row, 1)
        assert task.task_id == "frames_hard_001"
        assert task.difficulty == TaskDifficulty.HARD
        assert task.split == TaskSplit.EVAL
        assert task.gold_answer == "James II"
        assert "James II" in task.scoring_rubric  # rubric anchored on gold answer
        assert task.known_assumptions is None


class TestCrepeMapper:
    def test_maps_to_tricky_with_assumptions(self):
        row = {
            "question": "Why are deleted files unrecoverable from an SSD?",
            "presuppositions": ["Deleted files are unrecoverable from an SSD."],
            "corrections": ["Deleted files often remain recoverable until overwritten."],
            "labels": ["false presupposition"],
        }
        task = crepe_row_to_task(row, 1)
        assert task.task_id == "crepe_tricky_001"
        assert task.difficulty == TaskDifficulty.TRICKY
        assert task.known_assumptions == ["Deleted files are unrecoverable from an SSD."]
        assert task.gold_answer == "Deleted files often remain recoverable until overwritten."

    def test_tricky_without_presupposition_is_rejected(self):
        # The dataset validator requires tricky tasks to carry an assumption;
        # the fetch path filters such rows out, but guard the invariant here.
        row = {"question": "x", "presuppositions": [], "corrections": ["y"]}
        with pytest.raises(ValueError, match="known_assumptions"):
            crepe_row_to_task(row, 1)


def _frames_row(i: int) -> dict:
    return {"Prompt": f"question {i}?", "Answer": f"answer {i}"}


def _crepe_row(i: int, *, valid: bool = True) -> dict:
    return {
        "question": f"question {i}?",
        "presuppositions": [f"presup {i}"] if valid else [],
        "corrections": [f"correction {i}"] if valid else [],
    }


class TestPager:
    """Offline coverage of the fetch/pagination logic via a mocked pager."""

    def test_frames_paginates_until_limit(self, monkeypatch):
        pages = [[_frames_row(i) for i in range(3)], [_frames_row(i) for i in range(3, 6)]]
        calls: list[tuple[int, int]] = []

        def fake_fetch_page(dataset, split, offset, length):
            calls.append((offset, length))
            return pages.pop(0)

        monkeypatch.setattr(hf_fetch, "fetch_page", fake_fetch_page)
        tasks = fetch_frames(limit=5)
        assert len(tasks) == 5
        assert [t.task_id for t in tasks] == [f"frames_hard_{i:03d}" for i in range(1, 6)]
        # Second page requested at the offset the first page ended at.
        assert calls[1][0] == 3

    def test_crepe_filtered_page_then_empty_terminal_page(self, monkeypatch):
        # First page: 2 usable rows among 4; second page empty -> pager stops
        # short of the limit instead of looping forever.
        pages = [
            [_crepe_row(0), _crepe_row(1, valid=False), _crepe_row(2), _crepe_row(3, valid=False)],
            [],
        ]
        monkeypatch.setattr(hf_fetch, "fetch_page", lambda *a: pages.pop(0))
        tasks = fetch_crepe(limit=10)
        assert len(tasks) == 2
        assert all(t.difficulty == TaskDifficulty.TRICKY for t in tasks)
        assert not pages  # both pages consumed

    def test_error_payload_raises_instead_of_empty_page(self, monkeypatch):
        monkeypatch.setattr(
            hf_fetch, "http_get_json", lambda url: {"error": "rate limited"}
        )
        with pytest.raises(RuntimeError, match="rate limited"):
            hf_fetch.fetch_page("some/dataset", "test", 0, 100)

    def test_http_get_json_retries_transient_errors(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(hf_fetch.time, "sleep", sleeps.append)
        attempts = iter([
            urllib.error.HTTPError("u", 429, "rate limited", None, None),
            urllib.error.URLError("conn reset"),
        ])

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"rows": []}'

        def fake_urlopen(req, timeout):
            try:
                raise next(attempts)
            except StopIteration:
                return FakeResponse()

        monkeypatch.setattr(hf_fetch.urllib.request, "urlopen", fake_urlopen)
        assert hf_fetch.http_get_json("http://x") == {"rows": []}
        assert len(sleeps) == 2  # backed off before each retry

    def test_http_get_json_gives_up_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr(hf_fetch.time, "sleep", lambda s: None)

        def always_503(req, timeout):
            raise urllib.error.HTTPError("u", 503, "unavailable", None, None)

        monkeypatch.setattr(hf_fetch.urllib.request, "urlopen", always_503)
        with pytest.raises(RuntimeError, match="Giving up"):
            hf_fetch.http_get_json("http://x")

    def test_non_retryable_http_error_raises_immediately(self, monkeypatch):
        def not_found(req, timeout):
            raise urllib.error.HTTPError("u", 404, "not found", None, None)

        monkeypatch.setattr(hf_fetch.urllib.request, "urlopen", not_found)
        with pytest.raises(urllib.error.HTTPError):
            hf_fetch.http_get_json("http://x")


class TestBuildHardBenchmark:
    def test_short_fetch_refuses_to_write_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            frames_mod, "fetch_frames", lambda n: [frames_row_to_task(_frames_row(1), 1)]
        )
        monkeypatch.setattr(crepe_mod, "fetch_crepe", lambda n: [])
        cache = tmp_path / "cache.json"
        with pytest.raises(RuntimeError, match="short benchmark"):
            build_hard_benchmark(frames_n=5, crepe_n=5, cache_path=cache)
        assert not cache.exists()

    def test_full_fetch_writes_cache(self, monkeypatch, tmp_path):
        frames = [frames_row_to_task(_frames_row(i), i) for i in range(1, 3)]
        crepe = [crepe_row_to_task(_crepe_row(i), i) for i in range(1, 3)]
        monkeypatch.setattr(frames_mod, "fetch_frames", lambda n: frames)
        monkeypatch.setattr(crepe_mod, "fetch_crepe", lambda n: crepe)
        cache = tmp_path / "cache.json"
        tasks = build_hard_benchmark(frames_n=2, crepe_n=2, cache_path=cache)
        assert len(tasks) == 4
        assert len(load_hard_benchmark(cache)) == 4

    def test_default_cache_is_repo_root_anchored(self):
        repo_root = Path(__file__).resolve().parents[1]
        assert _DEFAULT_CACHE == (
            repo_root / "data" / "external" / "hard_benchmark.json"
        )


class TestLoader:
    def test_loads_fixture_via_from_path(self):
        ds = ResearchQADataset.from_path(FIXTURE)
        assert len(ds) == 3
        assert len(ds.by_difficulty(TaskDifficulty.HARD)) == 2
        assert len(ds.by_difficulty(TaskDifficulty.TRICKY)) == 1
        for task in ds:
            assert task.question
            assert task.gold_answer
            assert task.scoring_rubric
            assert task.split == TaskSplit.EVAL

    def test_load_hard_benchmark_missing_cache_is_actionable(self, tmp_path):
        missing = tmp_path / "nope.json"
        with pytest.raises(FileNotFoundError, match="fetch_dataset"):
            load_hard_benchmark(missing)
