"""Research QA evaluation task dataset.

Provides the ResearchQATask schema and ResearchQADataset loader for the
130-task evaluation dataset spanning typical, hard, tricky, and tool_test
categories.
"""

from __future__ import annotations

import enum
import json
from importlib.resources import files

from pydantic import BaseModel, Field, model_validator


_DATA_PATH = files("bicameral_agent.data").joinpath("research_qa.json")


class TaskDifficulty(str, enum.Enum):
    """Difficulty classification for research QA tasks.

    - TYPICAL: Straightforward research questions with clear answers.
    - HARD: Require multiple research steps, synthesis across sources.
    - TRICKY: Contain common assumption traps (presuppose something false).
    - TOOL_TEST: Tasks designed for Phase 1 tool testing.
    """

    TYPICAL = "typical"
    HARD = "hard"
    TRICKY = "tricky"
    TOOL_TEST = "tool_test"


class TaskSplit(str, enum.Enum):
    """Dataset split membership.

    - EVAL: Held-out evaluation tasks (100 tasks).
    - TOOL_TEST: Tasks for Phase 1 tool testing (30 tasks).
    """

    EVAL = "eval"
    TOOL_TEST = "tool_test"


class ResearchQATask(BaseModel):
    """A single research QA evaluation task."""

    task_id: str
    difficulty: TaskDifficulty
    split: TaskSplit
    question: str
    gold_answer: str
    known_gaps: list[str] | None = Field(default=None)
    known_assumptions: list[str] | None = Field(default=None)
    scoring_rubric: str

    @model_validator(mode="after")
    def check_tricky_has_assumptions(self) -> ResearchQATask:
        """Tricky tasks must have at least one known assumption."""
        if self.difficulty == TaskDifficulty.TRICKY:
            if not self.known_assumptions:
                msg = f"Tricky task {self.task_id} must have at least one known_assumptions entry"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_gaps_nonempty(self) -> ResearchQATask:
        """If known_gaps is present, it must be non-empty (use None for no gaps)."""
        if self.known_gaps is not None and len(self.known_gaps) == 0:
            msg = f"Task {self.task_id} has known_gaps=[] -- use None for no gaps"
            raise ValueError(msg)
        return self


class ResearchQADataset:
    """Loader and filter interface for the research QA evaluation dataset.

    Loads 130 tasks from the bundled JSON file and provides methods to
    filter by difficulty, split, and annotation status.
    """

    def __init__(self, tasks: list[ResearchQATask] | None = None) -> None:
        if tasks is not None:
            self._tasks = list(tasks)
        else:
            self._tasks = self._load_from_package()

    @staticmethod
    def _load_from_package() -> list[ResearchQATask]:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        return [ResearchQATask.model_validate(item) for item in raw]

    @property
    def tasks(self) -> list[ResearchQATask]:
        """All tasks in the dataset."""
        return list(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    def by_difficulty(self, difficulty: TaskDifficulty) -> list[ResearchQATask]:
        """Return tasks matching the given difficulty."""
        return [t for t in self._tasks if t.difficulty == difficulty]

    def by_split(self, split: TaskSplit) -> list[ResearchQATask]:
        """Return tasks in the given split."""
        return [t for t in self._tasks if t.split == split]

    def with_gaps(self) -> list[ResearchQATask]:
        """Return tasks annotated with known research gaps."""
        return [t for t in self._tasks if t.known_gaps is not None]

    def without_gaps(self) -> list[ResearchQATask]:
        """Return tasks without known research gaps."""
        return [t for t in self._tasks if t.known_gaps is None]

    def with_assumptions(self) -> list[ResearchQATask]:
        """Return tasks annotated with known false assumptions."""
        return [t for t in self._tasks if t.known_assumptions is not None]

    def eval_tasks(self) -> list[ResearchQATask]:
        """Shorthand for by_split(TaskSplit.EVAL)."""
        return self.by_split(TaskSplit.EVAL)

    def tool_test_tasks(self) -> list[ResearchQATask]:
        """Shorthand for by_split(TaskSplit.TOOL_TEST)."""
        return self.by_split(TaskSplit.TOOL_TEST)
