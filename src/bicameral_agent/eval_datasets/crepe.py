"""CREPE adapter: false-presupposition QA mapped to the ``tricky`` tier.

Pulled via the ``tasksource/CREPE`` re-host of the false-presupposition CREPE
(arXiv:2211.17257), which carries the presuppositions/corrections annotations
we map from. Upstream declares no license, so the data is fetch-only -- never
redistributed. See ``docs/hard_benchmark.md``.
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

CREPE_DATASET = "tasksource/CREPE"


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


def _crepe_row_valid(row: dict) -> bool:
    presups = [p for p in (row.get("presuppositions") or []) if p and p.strip()]
    corrections = [c for c in (row.get("corrections") or []) if c and c.strip()]
    return bool(presups and corrections)


def fetch_crepe(limit: int = 60, split: str = "test") -> list[ResearchQATask]:
    """Scan CREPE and map false-presupposition rows to ``tricky`` tasks."""
    # Fixed 100-row pages: most CREPE rows are filtered out, so asking for
    # only the remaining count would multiply requests.
    return hf_fetch.fetch_mapped_tasks(
        CREPE_DATASET,
        split,
        limit,
        crepe_row_to_task,
        is_valid=_crepe_row_valid,
        page_length=100,
    )


class Crepe(EvalDataset):
    """CREPE false-presupposition QA (``tricky`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="crepe",
        source=CREPE_DATASET,
        license="None declared (NOASSERTION) -- fetch-only, do not redistribute",
        citation=(
            "CREPE: Open-Domain Question Answering with False Presuppositions, "
            "Yu et al., ACL 2023 (arXiv:2211.17257)"
        ),
    )
    default_metric: ClassVar[str] = "llm_judge"
    supported_metrics: ClassVar[tuple[str, ...]] = ("llm_judge", "lexical")
    default_limit = 60

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_crepe(limit)
