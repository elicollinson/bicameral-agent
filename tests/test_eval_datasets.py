"""Tests for the evaluation-dataset factory (Issue #56).

Fully offline: adapters are exercised against monkeypatched pagers and tmp
caches; no network and no externally-licensed data are required.
"""

import json
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
from bicameral_agent.eval_datasets import (
    abstentionbench,
    bbeh,
    healthbench_hard,
    hf_fetch,
    hle,
    researchqa,
    simpleqa_verified,
    supergpqa,
)
from bicameral_agent.eval_datasets.base import EXTERNAL_DATA_DIR
from bicameral_agent.verifiers import verifier_names

REPO_ROOT = Path(__file__).resolve().parents[1]


def _frames_page(start: int, n: int) -> list[dict]:
    return [{"Prompt": f"q {i}?", "Answer": f"a {i}"} for i in range(start, start + n)]


class TestRegistry:
    def test_known_names(self):
        assert dataset_names() == [
            "abstentionbench",
            "bbeh",
            "builtin",
            "crepe",
            "frames",
            "hard_benchmark",
            "healthbench_hard",
            "hle",
            "researchqa",
            "simpleqa_verified",
            "supergpqa",
        ]

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


class TestSimpleQAVerified:
    def test_row_mapping(self, monkeypatch):
        rows = [
            {"problem": "Capital of France?", "answer": "Paris", "topic": "Geo"},
            {"problem": "", "answer": "skipped"},  # blank problem filtered
            {"problem": "2+2?", "answer": "4"},
        ]
        monkeypatch.setattr(
            hf_fetch, "fetch_page", lambda *a, **kw: rows if a[2] == 0 else []
        )
        tasks = simpleqa_verified.fetch_simpleqa_verified(limit=2)
        assert [t.task_id for t in tasks] == [
            "simpleqa_typical_001",
            "simpleqa_typical_002",
        ]
        assert tasks[0].gold_answer == "Paris"
        assert all(t.difficulty == TaskDifficulty.TYPICAL for t in tasks)

    def test_uses_named_config(self, monkeypatch):
        seen: dict = {}

        def fake_fetch(dataset, split, offset, length, config="default"):
            seen.update(dataset=dataset, split=split, config=config)
            return []

        monkeypatch.setattr(hf_fetch, "fetch_page", fake_fetch)
        simpleqa_verified.fetch_simpleqa_verified(limit=1)
        assert seen == {
            "dataset": "google/simpleqa-verified",
            "split": "eval",
            "config": "simpleqa_verified",
        }


class TestSuperGPQA:
    def test_row_mapping(self, monkeypatch):
        rows = [
            {
                "question": "Which is prime?",
                "options": ["4", "7", "9"],
                "answer": "7",
                "answer_letter": "b",
            }
        ]
        monkeypatch.setattr(
            hf_fetch, "fetch_page", lambda *a, **kw: rows if a[2] == 0 else []
        )
        tasks = supergpqa.fetch_supergpqa(limit=1)
        task = tasks[0]
        assert task.gold_answer == "B"  # normalized to upper case
        assert task.choices == ["4", "7", "9"]
        assert "A. 4" in task.question and "B. 7" in task.question
        assert task.difficulty == TaskDifficulty.HARD

    def test_rows_missing_fields_skipped(self, monkeypatch):
        rows = [
            {"question": "no options", "options": [], "answer_letter": "A"},
            {"question": "ok", "options": ["x", "y"], "answer_letter": "A"},
        ]
        monkeypatch.setattr(
            hf_fetch, "fetch_page", lambda *a, **kw: rows if a[2] == 0 else []
        )
        tasks = supergpqa.fetch_supergpqa(limit=2)
        assert len(tasks) == 1
        assert tasks[0].question.startswith("ok")


class TestBbeh:
    def test_round_robin_sampling(self, monkeypatch):
        def fake_get(url):
            name = url.rsplit("/", 2)[-2]
            return {
                "examples": [
                    {"input": f"{name} q{i}", "target": f"t{i}"} for i in range(3)
                ]
            }

        monkeypatch.setattr(hf_fetch, "http_get_json", fake_get)
        tasks = bbeh.fetch_bbeh(limit=25)
        assert len(tasks) == 25
        # ceil(25/23) = 2 per subtask; the first two subtasks contribute both
        assert tasks[0].question == "bbeh_boardgame_qa q0"
        assert tasks[1].question == "bbeh_boardgame_qa q1"
        assert tasks[2].question == "bbeh_boolean_expressions q0"
        assert all(t.difficulty == TaskDifficulty.HARD for t in tasks)
        assert tasks[0].gold_answer == "t0"


class TestHealthBenchHard:
    _RECORD = {
        "prompt": [{"role": "user", "content": "Is my dose safe?"}],
        "prompt_id": "abc",
        "rubrics": [
            {"criterion": "Advises consulting a clinician", "points": 5},
            {"criterion": "Recommends doubling the dose", "points": -3},
        ],
        "ideal_completions_data": {"ideal_completion": "Check with your doctor."},
    }

    def test_jsonl_mapping(self, monkeypatch):
        jsonl = json.dumps(self._RECORD) + "\n\n" + json.dumps(self._RECORD)
        monkeypatch.setattr(hf_fetch, "http_get_text", lambda url: jsonl)
        tasks = healthbench_hard.fetch_healthbench_hard(limit=2)
        assert len(tasks) == 2
        task = tasks[0]
        assert task.question == "Is my dose safe?"
        assert task.gold_answer == "Check with your doctor."
        assert [i.points for i in task.rubric_items] == [5.0, -3.0]
        assert task.difficulty == TaskDifficulty.HARD

    def test_multi_turn_prompt_flattened(self, monkeypatch):
        record = dict(
            self._RECORD,
            prompt=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Dose?"},
            ],
        )
        monkeypatch.setattr(
            hf_fetch, "http_get_text", lambda url: json.dumps(record)
        )
        (task,) = healthbench_hard.fetch_healthbench_hard(limit=1)
        assert "user: Hi" in task.question
        assert "assistant: Hello" in task.question


class TestResearchQA:
    def test_rubric_native_mapping(self, monkeypatch):
        rows = [
            {
                "query": "How does X affect Y?",
                "rubric": [
                    {"rubric_item": "Mentions mechanism A", "type": ["Other"]},
                    {"rubric_item": "Compares with Z", "type": ["Comparison"]},
                ],
            },
            {"query": "No rubric", "rubric": []},  # filtered out
        ]
        monkeypatch.setattr(
            hf_fetch, "fetch_page", lambda *a, **kw: rows if a[2] == 0 else []
        )
        tasks = researchqa.fetch_researchqa(limit=1)
        (task,) = tasks
        assert task.gold_answer == ""  # rubric-native: no single gold answer
        assert [i.criterion for i in task.rubric_items] == [
            "Mentions mechanism A",
            "Compares with Z",
        ]
        assert all(i.points == 1.0 for i in task.rubric_items)

    def test_default_metric_is_rubric_coverage(self):
        assert build_dataset("researchqa").default_metric == "rubric_coverage"


class TestAbstentionBench:
    def test_abstain_row_mapping(self, monkeypatch):
        rows = [
            {
                "question": "What is in my pocket?",
                "reference_answers": [],
                "should_abstain": True,
                "metadata_json": "{}",
            },
            {
                "question": "What is 2+2?",
                "reference_answers": ["4"],
                "should_abstain": False,
                "metadata_json": "{}",
            },
        ]
        monkeypatch.setattr(
            hf_fetch, "fetch_page", lambda *a, **kw: rows if a[2] == 0 else []
        )
        tasks = abstentionbench.fetch_abstentionbench(limit=2)
        assert tasks[0].abstention_expected is True
        assert "abstain" in tasks[0].gold_answer  # explicit abstain gold
        assert tasks[1].abstention_expected is False
        assert tasks[1].gold_answer == "4"

    def test_license_is_flagged_non_commercial(self):
        assert "NON-COMMERCIAL" in abstentionbench.AbstentionBench.meta.license


class TestHle:
    def test_multimodal_rows_filtered(self, monkeypatch):
        rows = [
            {"question": "Text-only?", "answer": "Yes", "image": ""},
            {"question": "Has image", "answer": "No", "image": "base64..."},
            {"question": "Also text", "answer": "Sure", "image": None},
        ]
        monkeypatch.setattr(
            hf_fetch, "fetch_page", lambda *a, **kw: rows if a[2] == 0 else []
        )
        tasks = hle.fetch_hle(limit=3)
        assert [t.question for t in tasks] == ["Text-only?", "Also text"]

    def test_requires_hf_token_flagged(self):
        assert hle.Hle.meta.requires_hf_token is True


class TestHfFetchAuth:
    def test_token_sent_to_datasets_server(self, monkeypatch):
        seen: dict = {}

        def fake_json(url, headers=None):
            seen["headers"] = headers
            return {"rows": []}

        monkeypatch.setattr(hf_fetch, "http_get_json", fake_json)
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        hf_fetch.fetch_page("cais/hle", "test", 0, 10)
        assert seen["headers"] == {"Authorization": "Bearer hf_secret"}

    def test_no_token_no_header(self, monkeypatch):
        seen: dict = {}

        def fake_json(url, headers=None):
            seen["headers"] = headers
            return {"rows": []}

        monkeypatch.setattr(hf_fetch, "http_get_json", fake_json)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        hf_fetch.fetch_page("google/frames-benchmark", "test", 0, 10)
        assert seen["headers"] is None
