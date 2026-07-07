"""Back-compat shim: the hard benchmark moved into ``eval_datasets`` (Issue #56).

The FRAMES/CREPE mappers, the datasets-server pager, and the fetch/load
lifecycle now live in :mod:`bicameral_agent.eval_datasets` (``frames``,
``crepe``, ``hf_fetch``, ``hard``). This module re-exports the original public
surface so existing callers (``scripts/fetch_hard_benchmark.py``, docs) keep
working; new code should use ``build_dataset("hard_benchmark")``.
"""

from __future__ import annotations

from bicameral_agent.eval_datasets.crepe import CREPE_DATASET, crepe_row_to_task, fetch_crepe
from bicameral_agent.eval_datasets.frames import FRAMES_DATASET, fetch_frames, frames_row_to_task
from bicameral_agent.eval_datasets.hard import (
    DEFAULT_CACHE as _DEFAULT_CACHE,
)
from bicameral_agent.eval_datasets.hard import (
    build_hard_benchmark,
    load_hard_benchmark,
)

__all__ = [
    "CREPE_DATASET",
    "FRAMES_DATASET",
    "_DEFAULT_CACHE",
    "build_hard_benchmark",
    "crepe_row_to_task",
    "fetch_crepe",
    "fetch_frames",
    "frames_row_to_task",
    "load_hard_benchmark",
]
