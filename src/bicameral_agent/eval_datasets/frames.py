"""FRAMES adapter: multi-hop factual QA mapped to the ``hard`` tier.

``google/frames-benchmark`` (Apache-2.0): questions requiring synthesis across
several Wikipedia articles. See ``docs/hard_benchmark.md`` for the field
mapping, licensing, and attribution.
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

FRAMES_DATASET = "google/frames-benchmark"


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


def _frames_row_valid(row: dict) -> bool:
    return bool(
        (row.get("Prompt") or "").strip() and (row.get("Answer") or "").strip()
    )


def fetch_frames(limit: int = 100, split: str = "test") -> list[ResearchQATask]:
    """Pull a subset of FRAMES rows and map them to ``hard`` tasks."""
    return hf_fetch.fetch_mapped_tasks(
        FRAMES_DATASET, split, limit, frames_row_to_task, is_valid=_frames_row_valid
    )


class Frames(EvalDataset):
    """FRAMES multi-hop QA (``hard`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="frames",
        source=FRAMES_DATASET,
        license="Apache-2.0",
        citation=(
            "Fact, Fetch, and Reason: A Unified Evaluation of "
            "Retrieval-Augmented Generation (arXiv:2409.12941)"
        ),
    )
    default_metric: ClassVar[str] = "llm_judge"
    supported_metrics: ClassVar[tuple[str, ...]] = ("llm_judge", "lexical")
    default_limit = 100

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_frames(limit)
