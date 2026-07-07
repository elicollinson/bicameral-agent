"""Composite hard-benchmark adapter: FRAMES (hard) + CREPE (tricky) combined.

Preserves the Issue #42 artifact: the same ``data/external/hard_benchmark.json``
cache file, so previously fetched caches remain valid. The public
``build_hard_benchmark`` / ``load_hard_benchmark`` functions back the
``bicameral_agent.hard_benchmark`` compatibility shim.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from bicameral_agent.dataset import ResearchQADataset, ResearchQATask
from bicameral_agent.eval_datasets import crepe as crepe_mod
from bicameral_agent.eval_datasets import frames as frames_mod
from bicameral_agent.eval_datasets.base import EXTERNAL_DATA_DIR, DatasetMeta, EvalDataset

DEFAULT_CACHE = EXTERNAL_DATA_DIR / "hard_benchmark.json"


class HardBenchmark(EvalDataset):
    """FRAMES + CREPE combined pool (mixed hard/tricky difficulties)."""

    meta: ClassVar[DatasetMeta] = DatasetMeta(
        name="hard_benchmark",
        source=f"{frames_mod.FRAMES_DATASET} + {crepe_mod.CREPE_DATASET}",
        license=(
            "FRAMES: Apache-2.0; CREPE: none declared -- fetch-only, "
            "do not redistribute"
        ),
        citation="arXiv:2409.12941 (FRAMES); arXiv:2211.17257 (CREPE)",
    )
    default_metric: ClassVar[str] = "llm_judge"
    supported_metrics: ClassVar[tuple[str, ...]] = ("llm_judge", "lexical")

    def __init__(
        self,
        frames_n: int = 100,
        crepe_n: int = 60,
        cache_path: Path | str | None = None,
    ) -> None:
        super().__init__(cache_path)
        self.frames_n = frames_n
        self.crepe_n = crepe_n
        self.default_limit = frames_n + crepe_n

    def fetch_tasks(self, limit: int) -> list[ResearchQATask]:
        if limit != self.frames_n + self.crepe_n:
            raise ValueError(
                "hard_benchmark sizing is set via the frames_n/crepe_n options, "
                f"not a flat limit (got limit={limit}, "
                f"frames_n={self.frames_n} + crepe_n={self.crepe_n})"
            )
        frames = frames_mod.fetch_frames(self.frames_n)
        crepe = crepe_mod.fetch_crepe(self.crepe_n)
        if len(frames) != self.frames_n or len(crepe) != self.crepe_n:
            raise RuntimeError(
                f"Fetched {len(frames)}/{self.frames_n} FRAMES and "
                f"{len(crepe)}/{self.crepe_n} CREPE tasks; refusing to write a "
                "short benchmark cache."
            )
        return frames + crepe


def build_hard_benchmark(
    frames_n: int = 100,
    crepe_n: int = 60,
    cache_path: Path = DEFAULT_CACHE,
) -> list[ResearchQATask]:
    """Fetch subsets from upstream and write a normalized JSON cache.

    The cache file is git-ignored; it is the artifact :func:`load_hard_benchmark`
    reads. Returns the combined task list.

    Raises:
        RuntimeError: If either fetch yields fewer tasks than requested, so a
            partial upstream response cannot produce a silent short benchmark.
    """
    return HardBenchmark(frames_n=frames_n, crepe_n=crepe_n, cache_path=cache_path).build()


def load_hard_benchmark(cache_path: Path = DEFAULT_CACHE) -> ResearchQADataset:
    """Load the cached hard-benchmark tasks into a :class:`ResearchQADataset`."""
    return HardBenchmark(cache_path=cache_path).load()
