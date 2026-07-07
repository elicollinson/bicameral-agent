"""SimpleQA Verified adapter: short-form factuality mapped to ``typical``.

``google/simpleqa-verified`` (MIT): 1,000 human-verified short-answer factual
questions from OpenAI's SimpleQA. Gold answers are short strings, so the
deterministic ``exact_match`` verifier is the default; the LLM judge remains
available for graded credit.
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

SIMPLEQA_DATASET = "google/simpleqa-verified"
SIMPLEQA_CONFIG = "simpleqa_verified"
SIMPLEQA_SPLIT = "eval"


def simpleqa_row_to_task(row: dict, index: int) -> ResearchQATask:
    """Map one SimpleQA Verified row into a ``typical`` ResearchQATask."""
    question = (row.get("problem") or "").strip()
    answer = (row.get("answer") or "").strip()
    rubric = (
        f"5: States the correct short answer ('{answer}') exactly. "
        "3: States it with hedging or extraneous alternatives. 1: Wrong, "
        "missing, or fabricated."
    )
    return ResearchQATask(
        task_id=f"simpleqa_typical_{index:03d}",
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question=question,
        gold_answer=answer,
        scoring_rubric=rubric,
    )


def fetch_simpleqa_verified(limit: int = 100) -> list[ResearchQATask]:
    """Pull a subset of SimpleQA Verified rows and map them to tasks."""
    tasks: list[ResearchQATask] = []
    offset = 0
    while len(tasks) < limit:
        page = hf_fetch.fetch_page(
            SIMPLEQA_DATASET,
            SIMPLEQA_SPLIT,
            offset,
            min(100, limit - len(tasks)),
            config=SIMPLEQA_CONFIG,
        )
        if not page:
            break
        for raw in page:
            if (raw.get("problem") or "").strip() and (raw.get("answer") or "").strip():
                tasks.append(simpleqa_row_to_task(raw, len(tasks) + 1))
                if len(tasks) == limit:
                    break
        offset += len(page)
    return tasks


class SimpleQAVerified(EvalDataset):
    """SimpleQA Verified short-form factuality (``typical`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="simpleqa_verified",
        source=SIMPLEQA_DATASET,
        license="MIT",
        citation=(
            "SimpleQA Verified (Google DeepMind, 2025); based on Measuring "
            "short-form factuality in large language models (arXiv:2411.04368)"
        ),
    )
    default_metric: ClassVar[str] = "exact_match"
    supported_metrics: ClassVar[tuple[str, ...]] = (
        "exact_match",
        "llm_judge",
        "lexical",
    )
    default_limit = 100

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_simpleqa_verified(limit)
