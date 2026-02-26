"""Text embedding strategies for the state encoder.

Provides a protocol for text embedders and two implementations:

- ``HashEmbedder``: deterministic SHAKE-256 hash producing a 32-dim unit vector.
  No external dependencies; suitable as a fast, reproducible fallback.
- ``FastEmbedEmbedder``: ONNX-based ``all-MiniLM-L6-v2`` via the ``fastembed``
  library (install with ``pip install bicameral-agent[ml]``).  384-dim output
  is truncated to 32 and re-normalized.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding strategies."""

    def embed(self, text: str) -> np.ndarray:
        """Return a 32-dimensional float32 embedding for *text*."""
        ...


class HashEmbedder:
    """Deterministic SHAKE-256 hash embedder producing 32-dim unit vectors.

    Parameters
    ----------
    seed:
        Integer seed mixed into every hash for reproducibility across runs.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def embed(self, text: str) -> np.ndarray:
        """Hash *text* into a 32-dimensional unit vector (float32)."""
        payload = f"{self._seed}:{text}".encode("utf-8")
        raw = hashlib.shake_256(payload).digest(32)  # 32 bytes → 32 uint8 values
        vec = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        vec = vec / 255.0 * 2.0 - 1.0  # map [0, 255] → [-1, 1]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class FastEmbedEmbedder:
    """ONNX-based embedder using fastembed (optional ``[ml]`` extra).

    Lazily loads the model on first call.  The 384-dim output from
    ``all-MiniLM-L6-v2`` is truncated to 32 dimensions and re-normalized.

    Parameters
    ----------
    model_name:
        fastembed model identifier.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, text: str) -> np.ndarray:
        """Embed *text* into a 32-dimensional unit vector (float32)."""
        if self._model is None:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]

            self._model = TextEmbedding(model_name=self._model_name)
        embeddings = list(self._model.embed([text]))
        vec = np.array(embeddings[0], dtype=np.float32)[:32]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def get_default_embedder() -> Embedder:
    """Return the best available embedder.

    Tries ``FastEmbedEmbedder`` first; falls back to ``HashEmbedder``.
    """
    try:
        import fastembed as _fastembed  # noqa: F811, F401
        return FastEmbedEmbedder()
    except ImportError:
        return HashEmbedder()
