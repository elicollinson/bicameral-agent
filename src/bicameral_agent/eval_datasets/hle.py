"""Humanity's Last Exam (HLE) adapter: frontier-difficulty QA, ``hard`` tier.

``cais/hle`` (MIT, **gated**): expert-written questions at the frontier of
human knowledge. The dataset requires accepting the terms on Hugging Face and
an ``HF_TOKEN`` environment variable for the fetch (the pager sends it as a
bearer token). Multi-modal rows (non-empty ``image``) are filtered out --
the harness is text-only.

Row schema (per the public dataset card): ``id``, ``question``, ``answer``,
``answer_type`` (exactMatch / multipleChoice), ``image``, ``category``.
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

HLE_DATASET = "cais/hle"
HLE_SPLIT = "test"


def hle_row_to_task(row: dict, index: int) -> ResearchQATask:
    """Map one text-only HLE row into a ``hard`` ResearchQATask."""
    answer = str(row.get("answer") or "").strip()
    rubric = (
        f"5: States the correct answer ('{answer}') with sound reasoning. "
        "3: Correct answer, weak justification. 1: Incorrect or fabricated."
    )
    return ResearchQATask(
        task_id=f"hle_hard_{index:03d}",
        difficulty=TaskDifficulty.HARD,
        split=TaskSplit.EVAL,
        question=str(row.get("question") or "").strip(),
        gold_answer=answer,
        scoring_rubric=rubric,
    )


def fetch_hle(limit: int = 100) -> list[ResearchQATask]:
    """Pull text-only HLE rows (requires HF_TOKEN; the dataset is gated)."""
    tasks: list[ResearchQATask] = []
    offset = 0
    while len(tasks) < limit:
        page = hf_fetch.fetch_page(HLE_DATASET, HLE_SPLIT, offset, 100)
        if not page:
            break
        for raw in page:
            if raw.get("image"):  # text-only harness: skip multi-modal rows
                continue
            if str(raw.get("question") or "").strip() and str(
                raw.get("answer") or ""
            ).strip():
                tasks.append(hle_row_to_task(raw, len(tasks) + 1))
                if len(tasks) == limit:
                    break
        offset += len(page)
    return tasks


class Hle(EvalDataset):
    """Humanity's Last Exam text-only subset (``hard`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="hle",
        source=HLE_DATASET,
        license="MIT (gated: accept terms on Hugging Face; needs HF_TOKEN)",
        citation="Humanity's Last Exam (arXiv:2501.14249), CAIS & Scale AI",
        requires_hf_token=True,
    )
    default_metric: ClassVar[str] = "llm_judge"
    supported_metrics: ClassVar[tuple[str, ...]] = (
        "llm_judge",
        "exact_match",
        "lexical",
    )
    default_limit = 100

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_hle(limit)
