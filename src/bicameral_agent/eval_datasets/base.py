"""Adapter base class for plug-and-play evaluation datasets (Issue #56).

Each external benchmark is one small :class:`EvalDataset` subclass that maps
upstream rows into the existing :class:`ResearchQATask` shape. The lifecycle is
uniform across datasets:

- ``build(limit)`` fetches from upstream and writes a normalized JSON cache
  under ``data/external/<name>.json`` (git-ignored; no data is redistributed).
- ``load()`` reads that cache into a :class:`ResearchQADataset`, raising an
  actionable ``FileNotFoundError`` if it has not been fetched yet.
- ``default_metric`` names the verifier the dataset should be scored with
  (see :mod:`bicameral_agent.verifiers`), overridable within
  ``supported_metrics``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from bicameral_agent.dataset import ResearchQADataset, ResearchQATask

# Anchored to the repo root (this file lives at src/bicameral_agent/eval_datasets/)
# so caches resolve to the same place regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DATA_DIR = _REPO_ROOT / "data" / "external"


class DatasetMeta(BaseModel):
    """Provenance and licensing record for an evaluation dataset."""

    name: str
    source: str
    """HF repo id, GitHub URL, or packaged-data path."""
    license: str
    citation: str
    requires_hf_token: bool = False


class EvalDataset(ABC):
    """Base adapter: fetch-at-build into a git-ignored cache, load as tasks."""

    meta: ClassVar[DatasetMeta]
    default_metric: ClassVar[str]
    """Verifier-registry key (see bicameral_agent.verifiers.build_verifier)."""
    supported_metrics: ClassVar[tuple[str, ...]]
    default_limit: int
    """Tasks fetched by ``build()`` when no explicit limit is given."""

    def __init__(self, cache_path: Path | str | None = None) -> None:
        self._cache_override = Path(cache_path) if cache_path is not None else None

    def cache_path(self) -> Path:
        """Location of the local JSON cache for this dataset."""
        if self._cache_override is not None:
            return self._cache_override
        return EXTERNAL_DATA_DIR / f"{self.meta.name}.json"

    @abstractmethod
    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        """Pull up to *limit* tasks from upstream (network)."""

    def build(self, limit: int | None = None) -> list[ResearchQATask]:
        """Fetch from upstream and write the normalized JSON cache.

        Raises:
            RuntimeError: If the fetch yields fewer tasks than requested, so a
                partial upstream response cannot produce a silent short benchmark.
        """
        n = limit if limit is not None else self.default_limit
        tasks = self.fetch_tasks(n)
        if len(tasks) != n:
            raise RuntimeError(
                f"Fetched {len(tasks)}/{n} {self.meta.name} tasks; refusing to "
                "write a short benchmark cache."
            )
        path = self.cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([t.model_dump(mode="json") for t in tasks], indent=2),
            encoding="utf-8",
        )
        return tasks

    def load(self) -> ResearchQADataset:
        """Load the cached tasks into a :class:`ResearchQADataset`.

        Raises ``FileNotFoundError`` with fetch instructions if the cache is
        absent (external data is not committed to the repo).
        """
        path = self.cache_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset cache for {self.meta.name!r} not found at {path}. "
                "External data is not committed to the repo -- run "
                f"`python scripts/fetch_dataset.py --dataset {self.meta.name}` first."
            )
        return ResearchQADataset.from_path(path)
