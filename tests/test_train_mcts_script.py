"""Regression tests for scripts/train_mcts.py's live-collection wiring.

The offline (--episodes-parquet) path never touches task selection, so a
``dataset.tasks()`` TypeError hid until the first real training launch
(``ResearchQADataset.tasks`` is a property, unlike the ``eval_tasks()`` /
``tool_test_tasks()`` methods beside it). These tests drive the live-mode
branch of ``main()`` without any network by mocking the runner and
trainer.

The module requires torch (the script constructs the policy/transition
nets before task selection) and is skipped when unavailable.
"""

from __future__ import annotations

# ruff: noqa: E402  (imports below the importorskip guard are intentional)

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from bicameral_agent.dataset import ResearchQADataset


def _load_script():
    """Import scripts/train_mcts.py (not a package) by path."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "train_mcts.py"
    spec = importlib.util.spec_from_file_location("train_mcts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDatasetTaskAccess:
    def test_tasks_is_a_property_returning_all_tasks(self):
        """Guard the property-vs-method split on ResearchQADataset.

        ``tasks`` is a property (calling it raises ``TypeError: 'list'
        object is not callable``) while its siblings ``eval_tasks()`` /
        ``tool_test_tasks()`` are methods — the exact confusion behind
        the live-path bug.
        """
        assert isinstance(ResearchQADataset.tasks, property)
        dataset = ResearchQADataset()
        assert dataset.tasks == list(dataset)


class TestLiveCollectionPath:
    def test_main_reaches_trainer_with_disjoint_task_pools(self, tmp_path):
        """Live mode builds train/eval task pools without network."""
        script = _load_script()

        cost_tracker = MagicMock()
        cost_tracker.get_total.return_value = SimpleNamespace(total=0.0, call_count=0)

        with (
            patch.object(
                script, "build_runner", return_value=(MagicMock(), cost_tracker)
            ) as mock_build_runner,
            patch.object(script, "MCTSTrainer") as MockTrainer,
        ):
            rc = script.main(
                [
                    "--output-dir", str(tmp_path / "run"),
                    "--iterations", "0",
                    "--eval-tasks", "2",
                    "--policy-hidden-dim", "16",
                    "--transition-hidden-dim", "16",
                    "--quiet",
                ]
            )

        assert rc == 0
        assert mock_build_runner.called
        kwargs = MockTrainer.call_args.kwargs
        train_tasks = kwargs["train_tasks"]
        eval_tasks = kwargs["eval_tasks"]
        assert len(eval_tasks) == 2
        assert len(train_tasks) > 0

        train_ids = {t.task_id for t in train_tasks}
        eval_ids = {t.task_id for t in eval_tasks}
        assert train_ids.isdisjoint(eval_ids)
        # Collection pool == every dataset task not held out for eval.
        all_ids = {t.task_id for t in ResearchQADataset()}
        assert train_ids | eval_ids == all_ids
