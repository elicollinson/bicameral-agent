"""Tests for TrainingDataStore (issue #38).

Torch-dependent tests use ``pytest.importorskip("torch")`` (same
pattern as test_training_pipeline.py) so collection works without the
torch extra.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from bicameral_agent.training_data_store import TrainingDataStore
from bicameral_agent.training_pipeline import STATE_DIM, TrainingExample


def make_examples(
    n: int,
    *,
    seed: int = 0,
    episode_prefix: str = "ep",
) -> list[TrainingExample]:
    """Synthetic examples with the real TrainingExample shape."""
    rng = np.random.default_rng(seed)
    examples: list[TrainingExample] = []
    for i in range(n):
        examples.append(
            TrainingExample(
                state=rng.random(STATE_DIM).astype(np.float32),
                action=int(rng.integers(0, 4)),
                reward=float(rng.normal()),
                next_state=rng.random(STATE_DIM).astype(np.float32),
                done=bool(i % 7 == 6),
                discounted_return=float(rng.normal()),
                episode_id=f"{episode_prefix}-{i // 10}",
                decision_index=i % 10,
            )
        )
    return examples


@pytest.fixture
def store(tmp_path: Path) -> TrainingDataStore:
    return TrainingDataStore(tmp_path / "store")


# ---------------------------------------------------------------------------
# AC1: 10,000 examples round-trip exactly
# ---------------------------------------------------------------------------


def test_round_trip_10k_exact(store: TrainingDataStore) -> None:
    originals = make_examples(10_000, seed=1)
    store.save_examples(originals, iteration=0)

    loaded = store.load_examples()
    assert len(loaded) == 10_000

    np.testing.assert_array_equal(
        np.stack([e.state for e in loaded]),
        np.stack([e.state for e in originals]),
    )
    np.testing.assert_array_equal(
        np.stack([e.next_state for e in loaded]),
        np.stack([e.next_state for e in originals]),
    )
    assert [e.action for e in loaded] == [e.action for e in originals]
    assert [e.reward for e in loaded] == [e.reward for e in originals]
    assert [e.done for e in loaded] == [e.done for e in originals]
    assert [e.discounted_return for e in loaded] == [
        e.discounted_return for e in originals
    ]
    assert [e.episode_id for e in loaded] == [e.episode_id for e in originals]
    assert [e.decision_index for e in loaded] == [e.decision_index for e in originals]


# ---------------------------------------------------------------------------
# AC2: load time for 10,000 examples < 1 second
# ---------------------------------------------------------------------------


def test_load_10k_under_one_second(store: TrainingDataStore) -> None:
    store.save_examples(make_examples(10_000, seed=2), iteration=0)

    t0 = time.perf_counter()
    examples = store.load_examples()
    elapsed = time.perf_counter() - t0
    assert len(examples) == 10_000
    assert elapsed < 1.0, f"load_examples took {elapsed:.2f}s; budget is 1s"


def test_load_all_dataset_10k_under_one_second(store: TrainingDataStore) -> None:
    pytest.importorskip("torch")
    store.save_examples(make_examples(10_000, seed=3), iteration=0)

    t0 = time.perf_counter()
    dataset = store.load_all()
    n = len(dataset)
    first = dataset[0]
    last = dataset[n - 1]
    elapsed = time.perf_counter() - t0
    assert n == 10_000
    assert first[0].shape == (STATE_DIM,)
    assert last[0].shape == (STATE_DIM,)
    assert elapsed < 1.0, f"load_all took {elapsed:.2f}s; budget is 1s"


# ---------------------------------------------------------------------------
# AC3: train/val split is deterministic
# ---------------------------------------------------------------------------


def test_split_deterministic(store: TrainingDataStore, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    store.save_examples(make_examples(1_000, seed=4), iteration=0)

    train_a, val_a = store.split(train_ratio=0.8)
    # A fresh store instance over the same directory must give the same split.
    reopened = TrainingDataStore(tmp_path / "store")
    train_b, val_b = reopened.split(train_ratio=0.8)

    assert len(train_a) == len(train_b) == 800
    assert len(val_a) == len(val_b) == 200
    for i in range(len(train_a)):
        assert torch.equal(train_a[i][0], train_b[i][0])
    for i in range(len(val_a)):
        assert torch.equal(val_a[i][0], val_b[i][0])


def test_split_is_disjoint_and_covers_everything(store: TrainingDataStore) -> None:
    pytest.importorskip("torch")
    originals = make_examples(100, seed=5)
    store.save_examples(originals, iteration=0)

    train, val = store.split(train_ratio=0.8)
    # Use discounted_return as a per-example fingerprint (float64 draws
    # are unique with probability 1).
    split_values = sorted(
        float(item[5]) for dataset in (train, val) for item in dataset
    )
    original_values = sorted(
        np.float32(e.discounted_return) for e in originals
    )
    assert split_values == pytest.approx(original_values)


def test_split_rejects_bad_ratio(store: TrainingDataStore) -> None:
    pytest.importorskip("torch")
    store.save_examples(make_examples(10), iteration=0)
    with pytest.raises(ValueError):
        store.split(train_ratio=0.0)
    with pytest.raises(ValueError):
        store.split(train_ratio=1.0)


# ---------------------------------------------------------------------------
# AC4: incremental addition (5K + 5K -> 10K) without rewriting old chunks
# ---------------------------------------------------------------------------


def test_incremental_addition(store: TrainingDataStore, tmp_path: Path) -> None:
    store.save_examples(make_examples(5_000, seed=6, episode_prefix="a"), iteration=0)

    root = tmp_path / "store"
    first_chunk = sorted(p for p in root.iterdir() if p.is_dir())[0]
    before = {p.name: p.read_bytes() for p in first_chunk.iterdir()}

    store.save_examples(make_examples(5_000, seed=7, episode_prefix="b"), iteration=1)

    assert len(store) == 10_000
    loaded = store.load_examples()
    assert len(loaded) == 10_000
    assert sum(1 for e in loaded if e.episode_id.startswith("a-")) == 5_000
    assert sum(1 for e in loaded if e.episode_id.startswith("b-")) == 5_000

    # The first chunk's files were not rewritten.
    after = {p.name: p.read_bytes() for p in first_chunk.iterdir()}
    assert before == after


# ---------------------------------------------------------------------------
# AC5: filtering by iteration
# ---------------------------------------------------------------------------


def test_load_iteration_filters_correctly(store: TrainingDataStore) -> None:
    store.save_examples(make_examples(30, seed=8, episode_prefix="it1"), iteration=1)
    store.save_examples(make_examples(20, seed=9, episode_prefix="it2"), iteration=2)
    # A second chunk in iteration 1 accumulates.
    store.save_examples(make_examples(10, seed=10, episode_prefix="it1b"), iteration=1)

    assert store.iterations == [1, 2]
    it1 = store.load_examples(iteration=1)
    it2 = store.load_examples(iteration=2)
    assert len(it1) == 40
    assert len(it2) == 20
    assert all(e.episode_id.startswith("it1") for e in it1)
    assert all(e.episode_id.startswith("it2") for e in it2)
    assert store.load_examples(iteration=3) == []


def test_load_iteration_dataset(store: TrainingDataStore) -> None:
    pytest.importorskip("torch")
    store.save_examples(make_examples(30, seed=8), iteration=1)
    store.save_examples(make_examples(20, seed=9), iteration=2)
    assert len(store.load_iteration(1)) == 30
    assert len(store.load_iteration(2)) == 20
    assert len(store.load_iteration(3)) == 0


# ---------------------------------------------------------------------------
# AC6: PyTorch DataLoader batch iteration
# ---------------------------------------------------------------------------


def test_dataloader_iteration(store: TrainingDataStore) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    store.save_examples(make_examples(64, seed=11), iteration=0)
    loader = DataLoader(store.load_all(), batch_size=16, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4

    states, actions, rewards, next_states, dones, returns = batches[0]
    assert states.shape == (16, STATE_DIM)
    assert next_states.shape == (16, STATE_DIM)
    assert states.dtype == torch.float32
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.bool
    assert returns.dtype == torch.float32


# ---------------------------------------------------------------------------
# Metadata filtering (episode quality, controller type, ...)
# ---------------------------------------------------------------------------


def test_load_filtered_by_chunk_metadata(store: TrainingDataStore) -> None:
    pytest.importorskip("torch")
    store.save_examples(
        make_examples(30, seed=12),
        iteration=0,
        metadata={"controller": "heuristic", "quality": 0.9},
    )
    store.save_examples(
        make_examples(20, seed=13),
        iteration=0,
        metadata={"controller": "random", "quality": 0.2},
    )

    heuristic = store.load_filtered(lambda m: m["controller"] == "heuristic")
    assert len(heuristic) == 30
    good = store.load_filtered(lambda m: m["quality"] >= 0.5)
    assert len(good) == 30
    none = store.load_filtered(lambda m: m["quality"] > 1.0)
    assert len(none) == 0


def test_load_filtered_by_example_fields(store: TrainingDataStore) -> None:
    pytest.importorskip("torch")
    store.save_examples(make_examples(50, seed=14, episode_prefix="x"), iteration=0)
    first_episode = store.load_filtered(lambda m: m["episode_id"] == "x-0")
    assert len(first_episode) == 10
    terminal = store.load_filtered(lambda m: m["done"])
    assert len(terminal) == 50 // 7


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


def test_empty_store(store: TrainingDataStore) -> None:
    assert len(store) == 0
    assert store.iterations == []
    assert store.load_examples() == []


def test_empty_store_dataset(store: TrainingDataStore) -> None:
    pytest.importorskip("torch")
    assert len(store.load_all()) == 0


def test_save_empty_list_raises(store: TrainingDataStore) -> None:
    with pytest.raises(ValueError):
        store.save_examples([], iteration=0)


def test_save_negative_iteration_raises(store: TrainingDataStore) -> None:
    with pytest.raises(ValueError):
        store.save_examples(make_examples(1), iteration=-1)


def test_save_wrong_state_dim_raises(store: TrainingDataStore) -> None:
    bad = TrainingExample(
        state=np.zeros(STATE_DIM + 1, dtype=np.float32),
        action=0,
        reward=0.0,
        next_state=np.zeros(STATE_DIM + 1, dtype=np.float32),
        done=False,
        discounted_return=0.0,
        episode_id="bad",
        decision_index=0,
    )
    with pytest.raises(ValueError):
        store.save_examples([bad], iteration=0)


def test_incomplete_chunk_is_ignored(store: TrainingDataStore, tmp_path: Path) -> None:
    store.save_examples(make_examples(5, seed=15), iteration=0)
    # Simulate a crash mid-save: chunk dir without meta.json.
    partial = tmp_path / "store" / "chunk-000001"
    partial.mkdir()
    np.save(partial / "states.npy", np.zeros((3, STATE_DIM), dtype=np.float32))

    assert len(store) == 5
    # The next save must not collide with the partial directory.
    store.save_examples(make_examples(5, seed=16), iteration=1)
    assert len(store) == 10
