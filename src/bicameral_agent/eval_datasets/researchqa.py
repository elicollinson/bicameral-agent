"""ResearchQA adapter: rubric-native long-form research QA, ``hard`` tier.

``realliyifei/ResearchQA`` (MIT): research questions mined from scholarly
surveys, each annotated with per-question rubric items but **no single gold
answer** -- tasks carry ``gold_answer=""`` plus ``rubric_items`` (the pairing
the dataset schema validator enforces). Upstream rubric items are unweighted,
so each maps to 1.0 points.
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import (
    ResearchQATask,
    RubricItem,
    TaskDifficulty,
    TaskSplit,
)
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

RESEARCHQA_DATASET = "realliyifei/ResearchQA"
RESEARCHQA_SPLIT = "test"


def researchqa_row_to_task(row: dict, index: int) -> ResearchQATask:
    """Map one ResearchQA row into a ``hard`` rubric-scored task."""
    rubric_items = [
        RubricItem(criterion=str(r.get("rubric_item") or "").strip(), points=1.0)
        for r in (row.get("rubric") or [])
        if str(r.get("rubric_item") or "").strip()
    ]
    return ResearchQATask(
        task_id=f"researchqa_hard_{index:03d}",
        difficulty=TaskDifficulty.HARD,
        split=TaskSplit.EVAL,
        question=str(row.get("query") or "").strip(),
        gold_answer="",
        scoring_rubric=(
            f"Scored by coverage of {len(rubric_items)} rubric items "
            "(see rubric_items); no single gold answer exists."
        ),
        rubric_items=rubric_items,
    )


def _researchqa_row_valid(row: dict) -> bool:
    has_rubric = any(
        str(r.get("rubric_item") or "").strip() for r in (row.get("rubric") or [])
    )
    return bool(str(row.get("query") or "").strip() and has_rubric)


def fetch_researchqa(limit: int = 100) -> list[ResearchQATask]:
    """Pull a subset of ResearchQA rows and map them to rubric tasks."""
    return hf_fetch.fetch_mapped_tasks(
        RESEARCHQA_DATASET,
        RESEARCHQA_SPLIT,
        limit,
        researchqa_row_to_task,
        is_valid=_researchqa_row_valid,
    )


class ResearchQA(EvalDataset):
    """ResearchQA rubric-native long-form research QA (``hard`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="researchqa",
        source=RESEARCHQA_DATASET,
        license="MIT",
        citation=(
            "ResearchQA: Evaluating Scholarly Question Answering at Scale "
            "(arXiv:2509.00496)"
        ),
    )
    default_metric: ClassVar[str] = "rubric_coverage"
    supported_metrics: ClassVar[tuple[str, ...]] = ("rubric_coverage",)
    default_limit = 100

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_researchqa(limit)
