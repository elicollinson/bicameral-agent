"""Append-only storage and loading for training examples.

Storage layout
--------------

Each :meth:`TrainingDataStore.save_examples` call writes one new
immutable *chunk* directory under the store root::

    root/
        chunk-000000/
            states.npy        (N, STATE_DIM) float32
            actions.npy       (N,)  int64
            rewards.npy       (N,)  float64
            next_states.npy   (N, STATE_DIM) float32
            dones.npy         (N,)  bool
            returns.npy       (N,)  float64
            meta.json         sidecar: iteration, count, per-example
                              provenance (episode_id, decision_index),
                              optional chunk-level metadata (controller
                              type, episode quality, ...)
        chunk-000001/
            ...

Existing chunks are never rewritten, so incremental addition of new
iterations is cheap and append-only. Arrays are opened with
``mmap_mode="r"`` so loading is fast and lazy even for large stores.
``meta.json`` is written last: a chunk directory without it is treated
as incomplete and ignored by loaders.

Scalar float columns (rewards, returns) are stored as float64 to
round-trip Python floats exactly; state vectors are float32, matching
what :class:`~bicameral_agent.training_pipeline.TrainingDataPipeline`
produces.

Torch is an optional dependency: this module imports without torch.
Only the ``Dataset``-returning methods (:meth:`load_all`,
:meth:`load_iteration`, :meth:`load_filtered`, :meth:`split`) require
it. Datasets yield ``(state, action, reward, next_state, done,
discounted_return)`` tuples with the same ordering and dtypes as
``TrainingDataPipeline.to_torch_dataset``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bicameral_agent.training_pipeline import STATE_DIM, TrainingExample

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import Dataset

_META_FILENAME = "meta.json"
_CHUNK_PREFIX = "chunk-"

# Column name -> on-disk dtype.
_COLUMN_DTYPES: dict[str, type] = {
    "states": np.float32,
    "actions": np.int64,
    "rewards": np.float64,
    "next_states": np.float32,
    "dones": np.bool_,
    "returns": np.float64,
}


class _Chunk:
    """One immutable saved batch: memory-mapped columns plus sidecar metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with (path / _META_FILENAME).open(encoding="utf-8") as f:
            self.meta: dict[str, Any] = json.load(f)
        self.iteration: int = int(self.meta["iteration"])
        self.count: int = int(self.meta["count"])
        self.arrays: dict[str, np.ndarray] = {
            name: np.load(path / f"{name}.npy", mmap_mode="r") for name in _COLUMN_DTYPES
        }

    def example_record(self, i: int) -> dict[str, Any]:
        """Per-example metadata record used for filtering.

        Chunk-level metadata (e.g. controller type, episode quality) is
        merged with per-example provenance fields.
        """
        record: dict[str, Any] = dict(self.meta.get("metadata") or {})
        record["iteration"] = self.iteration
        record["episode_id"] = self.meta["episode_ids"][i]
        record["decision_index"] = self.meta["decision_indices"][i]
        record["done"] = bool(self.arrays["dones"][i])
        return record


# Lazily created torch Dataset subclass (keeps torch optional at import).
_DATASET_CLS: type | None = None


def _get_dataset_cls() -> type:
    global _DATASET_CLS
    if _DATASET_CLS is not None:
        return _DATASET_CLS

    import torch
    from torch.utils.data import Dataset

    class MemmapExampleDataset(Dataset):
        """Dataset over memory-mapped chunks, optionally restricted to indices.

        Items are ``(state, action, reward, next_state, done,
        discounted_return)`` tensors, matching
        ``TrainingDataPipeline.to_torch_dataset``.
        """

        def __init__(self, chunks: list[_Chunk], indices: np.ndarray | None = None) -> None:
            self._chunks = chunks
            self._offsets = np.cumsum([0] + [c.count for c in chunks])
            total = int(self._offsets[-1])
            if indices is None:
                self._indices = np.arange(total, dtype=np.int64)
            else:
                self._indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return len(self._indices)

        def __getitem__(self, idx: int):
            g = int(self._indices[idx])
            ci = int(np.searchsorted(self._offsets, g, side="right")) - 1
            local = g - int(self._offsets[ci])
            arrays = self._chunks[ci].arrays
            return (
                torch.from_numpy(np.array(arrays["states"][local])),
                torch.tensor(int(arrays["actions"][local]), dtype=torch.long),
                torch.tensor(float(arrays["rewards"][local]), dtype=torch.float32),
                torch.from_numpy(np.array(arrays["next_states"][local])),
                torch.tensor(bool(arrays["dones"][local]), dtype=torch.bool),
                torch.tensor(float(arrays["returns"][local]), dtype=torch.float32),
            )

    _DATASET_CLS = MemmapExampleDataset
    return _DATASET_CLS


class TrainingDataStore:
    """Append-only on-disk store for :class:`TrainingExample` records.

    Parameters
    ----------
    root:
        Directory holding the store. Created if it does not exist.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_examples(
        self,
        examples: list[TrainingExample],
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append ``examples`` as a new immutable chunk tagged ``iteration``.

        Existing chunks are never touched. ``metadata`` is an optional
        chunk-level dict (e.g. ``{"controller": "heuristic",
        "quality": 0.9}``) stored in the sidecar and made available to
        :meth:`load_filtered` predicates.
        """
        if iteration < 0:
            msg = f"iteration must be >= 0, got {iteration}"
            raise ValueError(msg)
        if not examples:
            msg = "cannot save an empty list of examples"
            raise ValueError(msg)

        states = np.stack([e.state for e in examples]).astype(np.float32, copy=False)
        next_states = np.stack([e.next_state for e in examples]).astype(np.float32, copy=False)
        if states.shape[1] != STATE_DIM or next_states.shape[1] != STATE_DIM:
            msg = (
                f"state vectors must have {STATE_DIM} dims, got "
                f"{states.shape[1]} / {next_states.shape[1]}"
            )
            raise ValueError(msg)

        columns: dict[str, np.ndarray] = {
            "states": states,
            "actions": np.array([e.action for e in examples], dtype=np.int64),
            "rewards": np.array([e.reward for e in examples], dtype=np.float64),
            "next_states": next_states,
            "dones": np.array([e.done for e in examples], dtype=np.bool_),
            "returns": np.array([e.discounted_return for e in examples], dtype=np.float64),
        }

        chunk_dir = self._root / f"{_CHUNK_PREFIX}{self._next_chunk_id():06d}"
        chunk_dir.mkdir()
        for name, arr in columns.items():
            np.save(chunk_dir / f"{name}.npy", arr)

        meta = {
            "iteration": iteration,
            "count": len(examples),
            "state_dim": STATE_DIM,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata or {}),
            "episode_ids": [e.episode_id for e in examples],
            "decision_indices": [e.decision_index for e in examples],
        }
        # Written last: a chunk without meta.json is ignored by loaders,
        # so a crash mid-save cannot corrupt reads.
        with (chunk_dir / _META_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> Dataset:
        """Load all examples across all iterations as a PyTorch Dataset."""
        return _get_dataset_cls()(self._chunks())

    def load_iteration(self, n: int) -> Dataset:
        """Load examples saved under iteration ``n`` as a PyTorch Dataset."""
        return _get_dataset_cls()([c for c in self._chunks() if c.iteration == n])

    def load_filtered(self, predicate: Callable[[dict[str, Any]], bool]) -> Dataset:
        """Load examples whose metadata record satisfies ``predicate``.

        The predicate receives one dict per example containing
        ``iteration``, ``episode_id``, ``decision_index``, ``done`` and
        any chunk-level metadata passed to :meth:`save_examples`.
        """
        chunks = self._chunks()
        indices: list[int] = []
        offset = 0
        for chunk in chunks:
            for i in range(chunk.count):
                if predicate(chunk.example_record(i)):
                    indices.append(offset + i)
            offset += chunk.count
        return _get_dataset_cls()(chunks, np.array(indices, dtype=np.int64))

    def split(self, train_ratio: float = 0.8, seed: int = 0) -> tuple[Dataset, Dataset]:
        """Deterministic (train, val) split over all stored examples.

        The same store contents, ``train_ratio`` and ``seed`` always
        produce the identical split.
        """
        if not 0.0 < train_ratio < 1.0:
            msg = f"train_ratio must be in (0, 1), got {train_ratio}"
            raise ValueError(msg)
        chunks = self._chunks()
        total = sum(c.count for c in chunks)
        perm = np.random.default_rng(seed).permutation(total)
        n_train = int(total * train_ratio)
        cls = _get_dataset_cls()
        return cls(chunks, perm[:n_train]), cls(chunks, perm[n_train:])

    def load_examples(self, iteration: int | None = None) -> list[TrainingExample]:
        """Materialize stored examples back into :class:`TrainingExample` objects.

        Torch-free exact round-trip; optionally restricted to one iteration.
        """
        out: list[TrainingExample] = []
        for chunk in self._chunks():
            if iteration is not None and chunk.iteration != iteration:
                continue
            arrays = chunk.arrays
            episode_ids = chunk.meta["episode_ids"]
            decision_indices = chunk.meta["decision_indices"]
            for i in range(chunk.count):
                out.append(
                    TrainingExample(
                        state=np.array(arrays["states"][i]),
                        action=int(arrays["actions"][i]),
                        reward=float(arrays["rewards"][i]),
                        next_state=np.array(arrays["next_states"][i]),
                        done=bool(arrays["dones"][i]),
                        discounted_return=float(arrays["returns"][i]),
                        episode_id=episode_ids[i],
                        decision_index=int(decision_indices[i]),
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(c.count for c in self._chunks())

    @property
    def iterations(self) -> list[int]:
        """Sorted list of distinct iteration numbers present in the store."""
        return sorted({c.iteration for c in self._chunks()})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chunks(self) -> list[_Chunk]:
        """All complete chunks, sorted by chunk id (== insertion order)."""
        dirs = sorted(
            p
            for p in self._root.iterdir()
            if p.is_dir() and p.name.startswith(_CHUNK_PREFIX) and (p / _META_FILENAME).exists()
        )
        return [_Chunk(p) for p in dirs]

    def _next_chunk_id(self) -> int:
        ids = [
            int(p.name.removeprefix(_CHUNK_PREFIX))
            for p in self._root.iterdir()
            if p.is_dir() and p.name.startswith(_CHUNK_PREFIX)
        ]
        return max(ids) + 1 if ids else 0
