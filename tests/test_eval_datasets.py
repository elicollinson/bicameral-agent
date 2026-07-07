"""Tests for the evaluation-dataset factory (Issue #56).

Fully offline: adapters are exercised against monkeypatched pagers and tmp
caches; no network and no externally-licensed data are required.
"""

from pathlib import Path

import pytest

from bicameral_agent.dataset import ResearchQADataset, TaskDifficulty
from bicameral_agent.eval_datasets import (
    BuiltinPool,
    Crepe,
    Frames,
    HardBenchmark,
    _REGISTRY,
    build_dataset,
    dataset_names,
    resolve_metric,
)
from bicameral_agent.eval_datasets import hf_fetch
from bicameral_agent.eval_datasets.base import EXTERNAL_DATA_DIR
from bicameral_agent.verifiers import verifier_names

REPO_ROOT = Path(__file__).resolve().parents[1]


def _frames_page(start: int, n: int) -> list[dict]:
    return [{"Prompt": f"q {i}?", "Answer": f"a {i}"} for i in range(start, start + n)]


class TestRegistry:
    def test_known_names(self):
        assert dataset_names() == ["builtin", "crepe", "frames", "hard_benchmark"]

    def test_build_dataset_dispatches(self):
        assert isinstance(build_dataset("builtin"), BuiltinPool)
        assert isinstance(build_dataset("frames"), Frames)
        assert isinstance(build_dataset("crepe"), Crepe)
        assert isinstance(build_dataset("hard_benchmark"), HardBenchmark)

    def test_unknown_name_lists_known(self):
        with pytest.raises(ValueError, match="builtin"):
            build_dataset("nope")

    def test_opts_forwarded_to_adapter(self):
        ds = build_dataset("hard_benchmark", frames_n=2, crepe_n=3)
        assert ds.default_limit == 5

    def test_every_adapter_declares_meta_and_metrics(self):
        for name, cls in _REGISTRY.items():
            assert cls.meta.name == name  # cache path derives from meta.name
            assert cls.meta.license
            assert cls.default_metric in cls.supported_metrics
            # Every declared metric must be constructible via the verifier registry.
            for metric in cls.supported_metrics:
                assert metric in verifier_names()


class TestResolveMetric:
    def test_defaults_to_dataset_metric(self):
        assert resolve_metric(BuiltinPool()) == "llm_judge"

    def test_valid_override_wins(self):
        assert resolve_metric(BuiltinPool(), "lexical") == "lexical"

    def test_unsupported_override_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_metric(BuiltinPool(), "multiple_choice")


class TestBuiltinPool:
    def test_load_is_the_packaged_pool(self):
        ds = BuiltinPool().load()
        assert isinstance(ds, ResearchQADataset)
        assert len(ds) == len(ResearchQADataset())

    def test_build_writes_no_cache(self, tmp_path):
        cache = tmp_path / "builtin.json"
        tasks = BuiltinPool(cache_path=cache).build()
        assert tasks  # returns the bundled tasks...
        assert not cache.exists()  # ...without materializing a cache file


class TestAdapterLifecycle:
    def test_default_cache_is_repo_root_anchored(self):
        assert EXTERNAL_DATA_DIR == REPO_ROOT / "data" / "external"
        assert Frames().cache_path() == EXTERNAL_DATA_DIR / "frames.json"
        assert Crepe().cache_path() == EXTERNAL_DATA_DIR / "crepe.json"
        assert HardBenchmark().cache_path() == EXTERNAL_DATA_DIR / "hard_benchmark.json"

    def test_build_then_load_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            hf_fetch, "fetch_page", lambda ds, split, offset, length: _frames_page(offset, length)
        )
        cache = tmp_path / "frames.json"
        built = Frames(cache_path=cache).build(limit=3)
        assert len(built) == 3
        loaded = Frames(cache_path=cache).load()
        assert [t.task_id for t in loaded] == [t.task_id for t in built]
        assert all(t.difficulty == TaskDifficulty.HARD for t in loaded)

    def test_short_fetch_refuses_to_write_cache(self, monkeypatch, tmp_path):
        pages = [_frames_page(0, 2), []]
        monkeypatch.setattr(hf_fetch, "fetch_page", lambda *a: pages.pop(0))
        cache = tmp_path / "frames.json"
        with pytest.raises(RuntimeError, match="short benchmark"):
            Frames(cache_path=cache).build(limit=5)
        assert not cache.exists()

    def test_load_missing_cache_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_dataset.py --dataset frames"):
            Frames(cache_path=tmp_path / "nope.json").load()

    def test_hard_benchmark_rejects_flat_limit(self):
        with pytest.raises(ValueError, match="frames_n/crepe_n"):
            HardBenchmark(frames_n=2, crepe_n=2).build(limit=3)
