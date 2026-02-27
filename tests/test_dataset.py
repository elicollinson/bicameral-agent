"""Tests for the research QA evaluation task dataset (Issue #11)."""

import pytest
from pydantic import ValidationError

from bicameral_agent.dataset import (
    ResearchQADataset,
    ResearchQATask,
    TaskDifficulty,
    TaskSplit,
)


@pytest.fixture
def dataset():
    """Load the full dataset from bundled JSON."""
    return ResearchQADataset()


class TestDatasetCounts:
    """Verify task count distribution matches spec."""

    def test_total_count(self, dataset):
        assert len(dataset) == 130

    def test_typical_count(self, dataset):
        assert len(dataset.by_difficulty(TaskDifficulty.TYPICAL)) == 50

    def test_hard_count(self, dataset):
        assert len(dataset.by_difficulty(TaskDifficulty.HARD)) == 25

    def test_tricky_count(self, dataset):
        assert len(dataset.by_difficulty(TaskDifficulty.TRICKY)) == 25

    def test_tool_test_count(self, dataset):
        assert len(dataset.by_difficulty(TaskDifficulty.TOOL_TEST)) == 30

    def test_eval_split_count(self, dataset):
        assert len(dataset.eval_tasks()) == 100

    def test_tool_test_split_count(self, dataset):
        assert len(dataset.tool_test_tasks()) == 30

    def test_tool_test_with_gaps(self, dataset):
        tool_tests = dataset.tool_test_tasks()
        with_gaps = [t for t in tool_tests if t.known_gaps is not None]
        assert len(with_gaps) == 15

    def test_tool_test_without_gaps(self, dataset):
        tool_tests = dataset.tool_test_tasks()
        without_gaps = [t for t in tool_tests if t.known_gaps is None]
        assert len(without_gaps) == 15


class TestNoOverlap:
    """Eval and tool_test splits have no overlapping task_ids."""

    def test_no_id_overlap(self, dataset):
        eval_ids = {t.task_id for t in dataset.eval_tasks()}
        tool_ids = {t.task_id for t in dataset.tool_test_tasks()}
        assert eval_ids.isdisjoint(tool_ids)

    def test_all_ids_unique(self, dataset):
        all_ids = [t.task_id for t in dataset]
        assert len(all_ids) == len(set(all_ids))


class TestSchemaValidation:
    """All fields populated, schema constraints hold."""

    def test_all_fields_populated(self, dataset):
        for task in dataset:
            assert task.task_id
            assert task.question
            assert task.gold_answer
            assert task.scoring_rubric
            assert task.difficulty in TaskDifficulty
            assert task.split in TaskSplit

    def test_tricky_tasks_have_assumptions(self, dataset):
        for task in dataset.by_difficulty(TaskDifficulty.TRICKY):
            assert task.known_assumptions is not None
            assert len(task.known_assumptions) >= 1

    def test_gap_annotated_tasks_have_at_least_one(self, dataset):
        for task in dataset.with_gaps():
            assert len(task.known_gaps) >= 1

    def test_task_id_format(self, dataset):
        for task in dataset:
            parts = task.task_id.rsplit("_", 1)
            assert len(parts) == 2
            assert parts[1].isdigit()


class TestSchemaRejection:
    """Pydantic validation rejects malformed tasks."""

    def test_tricky_without_assumptions_rejected(self):
        with pytest.raises(ValidationError):
            ResearchQATask(
                task_id="tricky_999",
                difficulty=TaskDifficulty.TRICKY,
                split=TaskSplit.EVAL,
                question="Why?",
                gold_answer="Because.",
                known_assumptions=None,
                scoring_rubric="5: Good. 1: Bad.",
            )

    def test_empty_gaps_list_rejected(self):
        with pytest.raises(ValidationError):
            ResearchQATask(
                task_id="test_999",
                difficulty=TaskDifficulty.TYPICAL,
                split=TaskSplit.EVAL,
                question="Why?",
                gold_answer="Because.",
                known_gaps=[],
                scoring_rubric="5: Good. 1: Bad.",
            )

    def test_invalid_difficulty_rejected(self):
        with pytest.raises(ValidationError):
            ResearchQATask(
                task_id="bad_001",
                difficulty="impossible",
                split=TaskSplit.EVAL,
                question="Q",
                gold_answer="A",
                scoring_rubric="rubric",
            )


class TestLoaderFiltering:
    """Loader filtering methods return correct subsets."""

    def test_by_difficulty_returns_only_matching(self, dataset):
        for diff in TaskDifficulty:
            tasks = dataset.by_difficulty(diff)
            assert all(t.difficulty == diff for t in tasks)

    def test_by_split_returns_only_matching(self, dataset):
        for split in TaskSplit:
            tasks = dataset.by_split(split)
            assert all(t.split == split for t in tasks)

    def test_with_gaps_all_have_gaps(self, dataset):
        for task in dataset.with_gaps():
            assert task.known_gaps is not None

    def test_without_gaps_none_have_gaps(self, dataset):
        for task in dataset.without_gaps():
            assert task.known_gaps is None

    def test_with_assumptions_all_have_assumptions(self, dataset):
        for task in dataset.with_assumptions():
            assert task.known_assumptions is not None

    def test_gaps_plus_no_gaps_equals_total(self, dataset):
        assert len(dataset.with_gaps()) + len(dataset.without_gaps()) == len(dataset)


class TestLoaderConstruction:
    """Dataset can be constructed from task list or defaults to file."""

    def test_from_task_list(self):
        task = ResearchQATask(
            task_id="test_001",
            difficulty=TaskDifficulty.TYPICAL,
            split=TaskSplit.EVAL,
            question="Q?",
            gold_answer="A.",
            scoring_rubric="5: Complete. 1: Missing.",
        )
        ds = ResearchQADataset(tasks=[task])
        assert len(ds) == 1
        assert ds.tasks[0].task_id == "test_001"

    def test_default_loads_from_file(self):
        ds = ResearchQADataset()
        assert len(ds) == 130

    def test_iteration(self, dataset):
        count = sum(1 for _ in dataset)
        assert count == 130


class TestSplitConsistency:
    """Difficulty and split are consistent."""

    def test_eval_tasks_are_typical_hard_or_tricky(self, dataset):
        for task in dataset.eval_tasks():
            assert task.difficulty in {
                TaskDifficulty.TYPICAL,
                TaskDifficulty.HARD,
                TaskDifficulty.TRICKY,
            }

    def test_tool_test_tasks_have_tool_test_difficulty(self, dataset):
        for task in dataset.tool_test_tasks():
            assert task.difficulty == TaskDifficulty.TOOL_TEST
