"""BIG-Bench Extra Hard (BBEH) adapter: hard reasoning mapped to ``hard``.

``google-deepmind/bbeh`` on GitHub (CC-BY-4.0): each subtask ships a
``task.json`` with ``{"examples": [{"input", "target"}]}``. We sample
round-robin across subtasks so a small pull still covers the reasoning mix,
fetched as raw JSON over urllib (no HF dependency).
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

BBEH_SOURCE = "https://github.com/google-deepmind/bbeh"
BBEH_RAW_URL = (
    "https://raw.githubusercontent.com/google-deepmind/bbeh/main/"
    "bbeh/benchmark_tasks/{subtask}/task.json"
)

# The repo's benchmark_tasks/ directories as of 2026-07 (fixed here so fetch
# order -- and therefore task ids -- is deterministic).
BBEH_SUBTASKS: tuple[str, ...] = (
    "bbeh_boardgame_qa",
    "bbeh_boolean_expressions",
    "bbeh_buggy_tables",
    "bbeh_causal_understanding",
    "bbeh_disambiguation_qa",
    "bbeh_dyck_languages",
    "bbeh_geometric_shapes",
    "bbeh_hyperbaton",
    "bbeh_linguini",
    "bbeh_movie_recommendation",
    "bbeh_multistep_arithmetic",
    "bbeh_nycc",
    "bbeh_object_counting",
    "bbeh_object_properties",
    "bbeh_sarc_triples",
    "bbeh_shuffled_objects",
    "bbeh_spatial_reasoning",
    "bbeh_sportqa",
    "bbeh_temporal_sequence",
    "bbeh_time_arithmetic",
    "bbeh_web_of_lies",
    "bbeh_word_sorting",
    "bbeh_zebra_puzzles",
)


def bbeh_example_to_task(example: dict, index: int) -> ResearchQATask:
    """Map one BBEH example into a ``hard`` ResearchQATask."""
    target = str(example.get("target") or "").strip()
    rubric = (
        f"5: States the exact correct answer ('{target}') with valid "
        "reasoning. 3: Correct answer, flawed or missing reasoning. "
        "1: Incorrect answer."
    )
    return ResearchQATask(
        task_id=f"bbeh_hard_{index:03d}",
        difficulty=TaskDifficulty.HARD,
        split=TaskSplit.EVAL,
        question=str(example.get("input") or "").strip(),
        gold_answer=target,
        scoring_rubric=rubric,
    )


def fetch_bbeh(limit: int = 100) -> list[ResearchQATask]:
    """Sample up to *limit* examples round-robin across BBEH subtasks."""
    per_subtask = -(-limit // len(BBEH_SUBTASKS))  # ceil
    tasks: list[ResearchQATask] = []
    for subtask in BBEH_SUBTASKS:
        payload = hf_fetch.http_get_json(BBEH_RAW_URL.format(subtask=subtask))
        for example in (payload.get("examples") or [])[:per_subtask]:
            if str(example.get("input") or "").strip() and str(
                example.get("target") or ""
            ).strip():
                tasks.append(bbeh_example_to_task(example, len(tasks) + 1))
                if len(tasks) == limit:
                    return tasks
    return tasks


class Bbeh(EvalDataset):
    """BIG-Bench Extra Hard reasoning tasks (``hard`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="bbeh",
        source=BBEH_SOURCE,
        license="CC-BY-4.0",
        citation=(
            "BIG-Bench Extra Hard (arXiv:2502.19187), Google DeepMind"
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
        return fetch_bbeh(limit)
