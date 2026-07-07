"""Schema + loader smoke tests for the harder benchmark integration (Issue #42).

Runs fully offline: the upstream mappers are exercised against synthetic raw
rows, and the loader against a committed author-owned fixture. No network and
no externally-licensed data are required.
"""

from pathlib import Path

import pytest

from bicameral_agent.dataset import ResearchQADataset, TaskDifficulty, TaskSplit
from bicameral_agent.hard_benchmark import (
    crepe_row_to_task,
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
        with pytest.raises(FileNotFoundError, match="fetch_hard_benchmark"):
            load_hard_benchmark(missing)
