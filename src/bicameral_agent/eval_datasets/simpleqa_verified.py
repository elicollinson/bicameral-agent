"""SimpleQA Verified adapter: short-form factuality mapped to ``typical``.

``google/simpleqa-verified`` (MIT): 1,000 human-verified short-answer factual
questions from OpenAI's SimpleQA. Gold answers are short strings, so the
deterministic ``exact_match`` verifier is the default; the official SimpleQA
3-way autorater (``llm_autorater``) and the LLM judge are the graded
alternatives.
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


def _simpleqa_row_valid(row: dict) -> bool:
    return bool(
        (row.get("problem") or "").strip() and (row.get("answer") or "").strip()
    )


def fetch_simpleqa_verified(limit: int = 100) -> list[ResearchQATask]:
    """Pull a subset of SimpleQA Verified rows and map them to tasks."""
    return hf_fetch.fetch_mapped_tasks(
        SIMPLEQA_DATASET,
        SIMPLEQA_SPLIT,
        limit,
        simpleqa_row_to_task,
        config=SIMPLEQA_CONFIG,
        is_valid=_simpleqa_row_valid,
    )


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
        "llm_autorater",
        "llm_judge",
        "lexical",
    )
    default_limit = 100

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_simpleqa_verified(limit)
