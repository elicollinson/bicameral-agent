"""Evaluation-dataset factory: pick a benchmark by name (Issue #56).

Mirrors ``model_client.build_client``: a plain dict registry, no decorators or
entry points. Adding a benchmark = one small adapter module + one registry
line. The package is named ``eval_datasets`` to avoid colliding with HF's
``datasets``.
"""

from __future__ import annotations

from bicameral_agent.eval_datasets.abstentionbench import AbstentionBench
from bicameral_agent.eval_datasets.base import DatasetMeta, EvalDataset
from bicameral_agent.eval_datasets.bbeh import Bbeh
from bicameral_agent.eval_datasets.builtin import BuiltinPool
from bicameral_agent.eval_datasets.crepe import Crepe
from bicameral_agent.eval_datasets.frames import Frames
from bicameral_agent.eval_datasets.hard import HardBenchmark
from bicameral_agent.eval_datasets.healthbench_hard import HealthBenchHard
from bicameral_agent.eval_datasets.hle import Hle
from bicameral_agent.eval_datasets.researchqa import ResearchQA
from bicameral_agent.eval_datasets.simpleqa_verified import SimpleQAVerified
from bicameral_agent.eval_datasets.supergpqa import SuperGPQA

__all__ = [
    "AbstentionBench",
    "Bbeh",
    "BuiltinPool",
    "Crepe",
    "DatasetMeta",
    "EvalDataset",
    "Frames",
    "HardBenchmark",
    "HealthBenchHard",
    "Hle",
    "ResearchQA",
    "SimpleQAVerified",
    "SuperGPQA",
    "build_dataset",
    "dataset_names",
    "resolve_metric",
]

_REGISTRY: dict[str, type[EvalDataset]] = {
    "builtin": BuiltinPool,
    "frames": Frames,
    "crepe": Crepe,
    "hard_benchmark": HardBenchmark,
    "simpleqa_verified": SimpleQAVerified,
    "supergpqa": SuperGPQA,
    "bbeh": Bbeh,
    "healthbench_hard": HealthBenchHard,
    "researchqa": ResearchQA,
    "abstentionbench": AbstentionBench,
    "hle": Hle,
}


def dataset_names() -> list[str]:
    """Names of all registered datasets."""
    return sorted(_REGISTRY)


def build_dataset(name: str = "builtin", **opts) -> EvalDataset:
    """Construct the evaluation dataset registered under *name*.

    Args:
        name: Registry key (see :func:`dataset_names`).
        **opts: Forwarded to the adapter constructor (e.g. ``cache_path``,
            or ``frames_n``/``crepe_n`` for ``hard_benchmark``).

    Raises:
        ValueError: If *name* is not registered.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}; known datasets: {sorted(_REGISTRY)}"
        ) from None
    return cls(**opts)


def resolve_metric(dataset: EvalDataset, override: str | None = None) -> str:
    """Resolve the verification metric for *dataset*.

    Returns the dataset's ``default_metric`` unless *override* is given, in
    which case it is validated against the dataset's ``supported_metrics``.
    """
    if override is None:
        return dataset.default_metric
    if override not in dataset.supported_metrics:
        raise ValueError(
            f"Metric {override!r} is not supported by dataset "
            f"{dataset.meta.name!r}; supported: {list(dataset.supported_metrics)}"
        )
    return override
