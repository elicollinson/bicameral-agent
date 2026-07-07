"""SuperGPQA adapter: graduate-level multiple choice mapped to ``hard``.

``m-a-p/SuperGPQA`` (ODC-BY, attribution required): multiple-choice questions
across 285 graduate disciplines. Options are embedded into the question text
(the answerer only sees ``question``) and also kept in ``choices`` for the
deterministic ``multiple_choice`` verifier; the gold answer is the letter.
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

SUPERGPQA_DATASET = "m-a-p/SuperGPQA"
SUPERGPQA_SPLIT = "train"


def format_choices(options: list[str]) -> str:
    """Render options as an A./B./... block for the question text."""
    return "\n".join(
        f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
    )


def supergpqa_row_to_task(row: dict, index: int) -> ResearchQATask:
    """Map one SuperGPQA row into a ``hard`` multiple-choice task."""
    options = [str(o) for o in (row.get("options") or [])]
    letter = (row.get("answer_letter") or "").strip().upper()
    question = (
        f"{(row.get('question') or '').strip()}\n\n"
        f"Options:\n{format_choices(options)}\n\n"
        "Answer with the letter of the correct option."
    )
    rubric = (
        f"5: Selects option {letter} with sound reasoning. 3: Selects option "
        f"{letter} without justification. 1: Selects any other option or none."
    )
    return ResearchQATask(
        task_id=f"supergpqa_hard_{index:03d}",
        difficulty=TaskDifficulty.HARD,
        split=TaskSplit.EVAL,
        question=question,
        gold_answer=letter,
        scoring_rubric=rubric,
        choices=options,
    )


def _supergpqa_row_valid(row: dict) -> bool:
    return bool(
        (row.get("question") or "").strip()
        and row.get("options")
        and (row.get("answer_letter") or "").strip()
    )


def fetch_supergpqa(limit: int = 100) -> list[ResearchQATask]:
    """Pull a subset of SuperGPQA rows and map them to tasks."""
    return hf_fetch.fetch_mapped_tasks(
        SUPERGPQA_DATASET,
        SUPERGPQA_SPLIT,
        limit,
        supergpqa_row_to_task,
        is_valid=_supergpqa_row_valid,
    )


class SuperGPQA(EvalDataset):
    """SuperGPQA graduate-level multiple choice (``hard`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="supergpqa",
        source=SUPERGPQA_DATASET,
        license="ODC-BY (attribution required)",
        citation=(
            "SuperGPQA: Scaling LLM Evaluation across 285 Graduate "
            "Disciplines (arXiv:2502.14739)"
        ),
    )
    default_metric: ClassVar[str] = "multiple_choice"
    supported_metrics: ClassVar[tuple[str, ...]] = ("multiple_choice", "llm_judge")
    default_limit = 100

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_supergpqa(limit)
