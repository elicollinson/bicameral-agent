"""HealthBench Hard adapter: rubric-graded health conversations, ``hard`` tier.

The 1,000-example "hard" subset published with OpenAI's simple-evals (MIT),
downloaded as JSONL from the openaipublic blob referenced by
``openai/simple-evals``. Each example carries a weighted rubric (points may be
negative for penalty items) which maps onto ``rubric_items`` for the
``rubric_coverage`` verifier; the published ideal completion (when present)
becomes the gold answer so the LLM judge stays usable.
"""

from __future__ import annotations

import json
from typing import ClassVar

from bicameral_agent.dataset import (
    ResearchQATask,
    RubricItem,
    TaskDifficulty,
    TaskSplit,
)
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset

HEALTHBENCH_SOURCE = "https://github.com/openai/simple-evals (HealthBench hard subset)"
HEALTHBENCH_HARD_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/"
    "healthbench/hard_2025-05-08-21-00-10.jsonl"
)


def healthbench_record_to_task(record: dict, index: int) -> ResearchQATask:
    """Map one HealthBench Hard JSONL record into a ``hard`` rubric task.

    Multi-turn prompts are flattened into a single question transcript (the
    answerer sees ``question`` only); single-turn prompts stay bare.
    """
    prompt = record.get("prompt") or []
    if len(prompt) == 1:
        question = str(prompt[0].get("content") or "").strip()
    else:
        question = "\n\n".join(
            f"{m.get('role', 'user')}: {str(m.get('content') or '').strip()}"
            for m in prompt
        )
    rubric_items = [
        RubricItem(criterion=str(r.get("criterion") or ""), points=float(r.get("points", 0)))
        for r in (record.get("rubrics") or [])
    ]
    ideal = (record.get("ideal_completions_data") or {}).get("ideal_completion") or ""
    return ResearchQATask(
        task_id=f"healthbench_hard_{index:03d}",
        difficulty=TaskDifficulty.HARD,
        split=TaskSplit.EVAL,
        question=question,
        gold_answer=str(ideal).strip(),
        scoring_rubric=(
            f"Scored by weighted coverage of {len(rubric_items)} rubric "
            "criteria (see rubric_items; negative-point items are penalties)."
        ),
        rubric_items=rubric_items,
    )


def fetch_healthbench_hard(limit: int = 100) -> list[ResearchQATask]:
    """Download the hard-subset JSONL and map the first *limit* records."""
    text = hf_fetch.http_get_text(HEALTHBENCH_HARD_URL)
    tasks: list[ResearchQATask] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not (record.get("prompt") and record.get("rubrics")):
            continue
        tasks.append(healthbench_record_to_task(record, len(tasks) + 1))
        if len(tasks) == limit:
            break
    return tasks


class HealthBenchHard(EvalDataset):
    """HealthBench Hard rubric-graded health QA (``hard`` tier)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="healthbench_hard",
        source=HEALTHBENCH_SOURCE,
        license="MIT",
        citation=(
            "HealthBench: Evaluating Large Language Models Towards Improved "
            "Human Health (arXiv:2505.08775), OpenAI"
        ),
    )
    default_metric: ClassVar[str] = "rubric_coverage"
    supported_metrics: ClassVar[tuple[str, ...]] = ("rubric_coverage", "llm_judge")
    default_limit = 100

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        return fetch_healthbench_hard(limit)
