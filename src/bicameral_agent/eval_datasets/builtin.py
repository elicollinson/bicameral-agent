"""Adapter for the packaged built-in research QA pool.

Wraps the bundled 130-task ``research_qa.json`` so the built-in pool goes
through the same factory as external benchmarks. ``--dataset builtin`` is
exactly today's behavior: nothing is fetched or cached.
"""

from __future__ import annotations

from typing import ClassVar

from bicameral_agent.dataset import ResearchQADataset, ResearchQATask
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset


class BuiltinPool(EvalDataset):
    """The packaged research QA evaluation pool."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="builtin",
        source="bicameral_agent/data/research_qa.json (bundled)",
        license="Project-owned",
        citation="",
    )
    default_metric: ClassVar[str] = "llm_judge"
    supported_metrics: ClassVar[tuple[str, ...]] = ("llm_judge", "lexical")
    default_limit = 130

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        """No network: the pool ships inside the package."""
        return self.load().tasks[:limit]

    def build(self, limit: int | None = None) -> list[ResearchQATask]:
        """No-op fetch: bundled data needs no external cache."""
        return self.load().tasks

    def load(self) -> ResearchQADataset:
        return ResearchQADataset()
