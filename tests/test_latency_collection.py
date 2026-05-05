"""Tests for the latency data collection harness (Issue #35)."""

from __future__ import annotations

import json
import os
from typing import Callable

import pyarrow.parquet as pq
import pytest

from bicameral_agent.gemini import GeminiResponse
from bicameral_agent.heuristic_controller import TOOL_IDS, Action
from bicameral_agent.latency_collection import (
    CONSCIOUS_LOOP_TOOL_ID,
    DEFAULT_CONV_LENGTHS,
    LatencyCollector,
    LatencyObservation,
    ToolObservation,
    load_observations,
    recompute_predictions,
    save_observations,
    synthesize_conversation,
)
from bicameral_agent.tool_latency import ToolLatencyModel


# ---------------------------------------------------------------------------
# Fake GeminiClient
# ---------------------------------------------------------------------------


class _FakeGeminiClient:
    """Stand-in for GeminiClient that returns scripted responses.

    Calls the registered ``on_completion`` callback after each response, so the
    LatencyCollector behaves exactly as it would with the real client.
    """

    def __init__(self, responses: list[GeminiResponse], on_completion: Callable | None = None) -> None:
        self._responses = list(responses)
        self._on_completion = on_completion
        self.call_count = 0

    @property
    def model(self) -> str:
        return "fake-gemini"

    def generate(self, *_args, **_kwargs) -> GeminiResponse:
        if not self._responses:
            raise RuntimeError("FakeGeminiClient ran out of scripted responses")
        resp = self._responses.pop(0)
        self.call_count += 1
        if self._on_completion is not None:
            self._on_completion(resp.input_tokens, resp.output_tokens, resp.duration_ms)
        return resp


def _refresher_response(drift: bool = False, *, input_tokens: int = 600, output_tokens: int = 80, duration_ms: float = 230.0) -> GeminiResponse:
    payload = {
        "drift_detected": drift,
        "drifts": [] if not drift else [
            {"category": "scope_creep", "description": "off-topic"}
        ],
        "reminder": None if not drift else "Refocus on the original task.",
    }
    return GeminiResponse(
        content=json.dumps(payload),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        finish_reason="STOP",
    )


def _gap_no_gaps_response(*, input_tokens: int = 700, output_tokens: int = 90, duration_ms: float = 250.0) -> GeminiResponse:
    return GeminiResponse(
        content=json.dumps({"has_gaps": False, "gaps": []}),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        finish_reason="STOP",
    )


def _auditor_no_assumptions_response(*, input_tokens: int = 550, output_tokens: int = 70, duration_ms: float = 210.0) -> GeminiResponse:
    return GeminiResponse(
        content=json.dumps({"assumptions": []}),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        finish_reason="STOP",
    )


# ---------------------------------------------------------------------------
# synthesize_conversation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", DEFAULT_CONV_LENGTHS)
def test_synthesize_conversation_hits_target_bucket(target: int) -> None:
    history = synthesize_conversation(target, seed=0)
    total = sum(m.token_count for m in history)
    # Within ±15% of the target — relaxed because we always finish a message.
    assert 0.85 * target <= total <= 1.15 * target, f"got {total} for target {target}"


def test_synthesize_conversation_alternates_roles() -> None:
    history = synthesize_conversation(1000, seed=42)
    roles = [m.role for m in history]
    for i in range(len(roles) - 1):
        assert roles[i] != roles[i + 1]


def test_synthesize_conversation_is_deterministic() -> None:
    a = synthesize_conversation(2000, seed=7)
    b = synthesize_conversation(2000, seed=7)
    assert [m.content for m in a] == [m.content for m in b]


# ---------------------------------------------------------------------------
# Coverage_within_pct (light tests; full tests live in test_latency_report)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LatencyCollector
# ---------------------------------------------------------------------------


def _build_collector_with_fake(responses: list[GeminiResponse]) -> tuple[LatencyCollector, _FakeGeminiClient, ToolLatencyModel]:
    model = ToolLatencyModel()
    collector = LatencyCollector(tool_latency_model=model)
    client = _FakeGeminiClient(responses, on_completion=collector.on_completion)
    collector.bind_client(client)
    return collector, client, model


def test_collect_tool_records_observation_per_call() -> None:
    """A single Refresher call should yield exactly one LatencyObservation."""
    collector, client, _ = _build_collector_with_fake([_refresher_response()])
    collector.collect_tool(TOOL_IDS[Action.REFRESHER], conv_length=1000, run_index=0)

    assert client.call_count == 1
    assert len(collector.api_observations) == 1
    obs = collector.api_observations[0]
    assert obs.tool_id == TOOL_IDS[Action.REFRESHER]
    assert obs.input_tokens == 600
    assert obs.output_tokens == 80
    assert obs.actual_duration_ms == 230.0
    # Predictions populated from the cold-start prior — must be > 0.
    assert obs.predicted_mean_ms > 0
    assert obs.predicted_p25_ms <= obs.predicted_mean_ms <= obs.predicted_p75_ms


def test_collect_tool_emits_tool_observation() -> None:
    collector, _, _ = _build_collector_with_fake([_auditor_no_assumptions_response()])
    collector.collect_tool(TOOL_IDS[Action.AUDITOR], conv_length=2000, run_index=1)

    assert len(collector.tool_observations) == 1
    tool_obs = collector.tool_observations[0]
    assert tool_obs.tool_id == TOOL_IDS[Action.AUDITOR]
    assert tool_obs.conversation_length_bucket == 2000
    assert tool_obs.run_index == 1
    assert tool_obs.actual_num_calls == 1
    assert tool_obs.actual_input_tokens == 550
    assert tool_obs.actual_output_tokens == 70
    assert tool_obs.actual_total_duration_ms > 0
    assert tool_obs.predicted_num_calls >= 1


def test_collect_tool_feeds_observations_into_latency_model() -> None:
    """After 3 collections, the underlying APILatencyModel should have 3 observations."""
    responses = [_refresher_response() for _ in range(3)]
    collector, _, model = _build_collector_with_fake(responses)
    for run in range(3):
        collector.collect_tool(TOOL_IDS[Action.REFRESHER], conv_length=1000, run_index=run)
    assert model.latency_model.observation_count == 3


def test_collect_tool_unknown_tool_id_raises() -> None:
    collector, _, _ = _build_collector_with_fake([_refresher_response()])
    with pytest.raises(ValueError, match="Unknown tool_id"):
        collector.collect_tool("nonexistent_tool", conv_length=1000, run_index=0)


def test_collect_tool_without_client_raises() -> None:
    model = ToolLatencyModel()
    collector = LatencyCollector(tool_latency_model=model)
    with pytest.raises(RuntimeError, match="bind_client"):
        collector.collect_tool(TOOL_IDS[Action.REFRESHER], conv_length=1000, run_index=0)


def test_collect_conscious_loop_records_observation() -> None:
    """Conscious loop call records one API observation under a dedicated tool id."""
    response = GeminiResponse(
        content="A short summary.",
        input_tokens=1200,
        output_tokens=15,
        duration_ms=180.0,
        finish_reason="STOP",
    )
    collector, _, _ = _build_collector_with_fake([response])
    collector.collect_conscious_loop(conv_length=1000, run_index=0)

    cl_obs = [o for o in collector.api_observations if o.tool_id == CONSCIOUS_LOOP_TOOL_ID]
    assert len(cl_obs) == 1
    assert cl_obs[0].input_tokens == 1200
    assert cl_obs[0].sub_call_label == "conscious_loop_turn"


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_observations_round_trip(tmp_path) -> None:
    api_obs = [
        LatencyObservation(
            tool_id="research_gap_scanner",
            sub_call_label="gap_identification",
            conversation_length_bucket=1000,
            run_index=0,
            input_tokens=600,
            output_tokens=80,
            actual_duration_ms=240.0,
            predicted_mean_ms=300.0,
            predicted_p25_ms=200.0,
            predicted_p75_ms=400.0,
            timestamp_ms=1_700_000_000,
        ),
        LatencyObservation(
            tool_id="conscious_loop",
            sub_call_label="conscious_loop_turn",
            conversation_length_bucket=2000,
            run_index=0,
            input_tokens=1300,
            output_tokens=20,
            actual_duration_ms=200.0,
            predicted_mean_ms=250.0,
            predicted_p25_ms=160.0,
            predicted_p75_ms=340.0,
            timestamp_ms=1_700_000_001,
        ),
    ]
    tool_obs = [
        ToolObservation(
            tool_id="research_gap_scanner",
            conversation_length_bucket=1000,
            run_index=0,
            actual_conversation_tokens=995,
            conversation_turn_count=10,
            predicted_input_tokens=1500,
            predicted_output_tokens=400,
            predicted_num_calls=2,
            actual_input_tokens=600,
            actual_output_tokens=80,
            actual_num_calls=1,
            actual_total_duration_ms=250.0,
            timestamp_ms=1_700_000_002,
        ),
    ]

    api_path, tool_path = save_observations(tmp_path, api_obs, tool_obs)
    assert os.path.exists(api_path)
    assert os.path.exists(tool_path)

    loaded_api, loaded_tool = load_observations(tmp_path)
    assert loaded_api == api_obs
    assert loaded_tool == tool_obs

    # Schema sanity: parquet file should contain all dataclass fields.
    table = pq.read_table(api_path)
    assert set(table.schema.names) == set(LatencyObservation.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# recompute_predictions
# ---------------------------------------------------------------------------


def test_recompute_predictions_uses_trained_model() -> None:
    """recompute_predictions overwrites pre-training predictions with fresh ones."""
    responses = [_refresher_response(input_tokens=600, output_tokens=80, duration_ms=240.0) for _ in range(8)]
    collector, _, model = _build_collector_with_fake(responses)
    for run in range(8):
        collector.collect_tool(TOOL_IDS[Action.REFRESHER], conv_length=1000, run_index=run)

    # Pre-training predictions were made by an untrained model — they should
    # differ from what the now-trained model produces.
    pre = collector.api_observations[0]
    final_api, final_tool = recompute_predictions(
        collector.api_observations, collector.tool_observations, model,
    )
    assert len(final_api) == 8
    assert len(final_tool) == 8
    new_pred = final_api[0].predicted_mean_ms
    fresh = model.latency_model.predict(pre.input_tokens, pre.output_tokens)
    assert new_pred == pytest.approx(fresh.mean_ms)


# ---------------------------------------------------------------------------
# Real-API smoke test (skipped when GEMINI_API_KEY is not set)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping real-API integration test.",
)
@pytest.mark.skipif(
    os.environ.get("BICAMERAL_SKIP_REAL_API") == "1",
    reason="BICAMERAL_SKIP_REAL_API=1; skipping real-API integration test.",
)
def test_real_gemini_collection_smoke() -> None:
    """One real Gemini call to confirm the harness wires up correctly.

    Set ``BICAMERAL_SKIP_REAL_API=1`` to disable this test even when an API
    key is configured (e.g. in CI without billing).
    """
    from bicameral_agent.gemini import GeminiClient

    model = ToolLatencyModel()
    collector = LatencyCollector(tool_latency_model=model)
    client = GeminiClient(on_completion=collector.on_completion)
    collector.bind_client(client)

    collector.collect_tool(TOOL_IDS[Action.REFRESHER], conv_length=1000, run_index=0)

    assert len(collector.api_observations) >= 1
    assert collector.api_observations[0].actual_duration_ms > 0
    assert collector.api_observations[0].input_tokens > 0
    assert model.latency_model.observation_count >= 1
