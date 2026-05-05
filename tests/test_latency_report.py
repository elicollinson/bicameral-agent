"""Tests for latency_report (Issue #35)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bicameral_agent.latency_collection import LatencyObservation, ToolObservation
from bicameral_agent.latency_report import (
    coverage_within_pct,
    format_text_report,
    layer1_coverage,
    layer2_coverage,
    save_scatter_plot,
)


def _api_obs(predicted: float, actual: float, *, tool_id: str = "research_gap_scanner", bucket: int = 1000) -> LatencyObservation:
    return LatencyObservation(
        tool_id=tool_id,
        sub_call_label="x",
        conversation_length_bucket=bucket,
        run_index=0,
        input_tokens=1000,
        output_tokens=100,
        actual_duration_ms=actual,
        predicted_mean_ms=predicted,
        predicted_p25_ms=predicted * 0.7,
        predicted_p75_ms=predicted * 1.3,
        timestamp_ms=1_700_000_000,
    )


def _tool_obs(predicted_out: int, actual_out: int, *, tool_id: str = "research_gap_scanner", bucket: int = 1000) -> ToolObservation:
    return ToolObservation(
        tool_id=tool_id,
        conversation_length_bucket=bucket,
        run_index=0,
        actual_conversation_tokens=bucket,
        conversation_turn_count=10,
        predicted_input_tokens=2000,
        predicted_output_tokens=predicted_out,
        predicted_num_calls=2,
        actual_input_tokens=1500,
        actual_output_tokens=actual_out,
        actual_num_calls=2,
        actual_total_duration_ms=300.0,
        timestamp_ms=1_700_000_000,
    )


def test_coverage_within_pct_basic() -> None:
    # 7 of 10 within ±25% of actual=100 → 0.7. Outliers: 130, 60, 50.
    actuals = [100.0] * 10
    predicteds = [100, 110, 124, 80, 50, 90, 105, 130, 60, 100]
    cov = coverage_within_pct(predicteds, actuals, 0.25)
    assert cov == pytest.approx(0.7)


def test_coverage_within_pct_skips_zero_actual() -> None:
    cov = coverage_within_pct([100, 100], [0, 100], 0.25)
    assert cov == pytest.approx(1.0)


def test_coverage_within_pct_empty() -> None:
    assert coverage_within_pct([], [], 0.25) == 0.0


def test_coverage_within_pct_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        coverage_within_pct([1.0], [1.0, 2.0], 0.1)


def test_layer2_coverage_uses_predicted_mean() -> None:
    obs = [
        _api_obs(predicted=100, actual=100),
        _api_obs(predicted=110, actual=100),
        _api_obs(predicted=140, actual=100),
    ]
    cov = layer2_coverage(obs, pct=0.25)
    assert cov == pytest.approx(2 / 3)


def test_layer1_coverage_uses_output_tokens() -> None:
    obs = [
        _tool_obs(predicted_out=100, actual_out=100),
        _tool_obs(predicted_out=110, actual_out=100),
        _tool_obs(predicted_out=200, actual_out=100),
    ]
    cov = layer1_coverage(obs, pct=0.30)
    assert cov == pytest.approx(2 / 3)


def test_format_text_report_smoke() -> None:
    api_obs = [_api_obs(predicted=100, actual=100, bucket=b) for b in (1000, 2000)]
    tool_obs = [_tool_obs(predicted_out=100, actual_out=100, bucket=b) for b in (1000, 2000)]
    report = format_text_report(api_obs, tool_obs)
    assert "Latency Data Collection Report" in report
    assert "Layer 2" in report
    assert "Layer 1" in report
    assert "Acceptance criteria" in report
    assert "research_gap_scanner" in report
    assert "1000" in report
    assert "2000" in report


def test_format_text_report_handles_empty_lists() -> None:
    report = format_text_report([], [])
    assert "Total API observations:  0" in report
    assert "AC1 ≥90 API observations: 0" in report


def test_save_scatter_plot_fallback_to_tsv(tmp_path, monkeypatch) -> None:
    """When matplotlib is unavailable, the fallback writes a .tsv with the data."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    api_obs = [
        _api_obs(predicted=100, actual=120),
        _api_obs(predicted=200, actual=210),
    ]
    out = save_scatter_plot(api_obs, tmp_path / "scatter.png")
    assert out.endswith(".tsv")
    content = Path(out).read_text()
    assert "predicted_ms" in content
    assert "120" in content
    assert "210" in content


def test_save_scatter_plot_uses_matplotlib_when_available(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    _ = matplotlib  # silence unused

    api_obs = [
        _api_obs(predicted=100, actual=120),
        _api_obs(predicted=200, actual=210, tool_id="assumption_auditor"),
    ]
    out = save_scatter_plot(api_obs, tmp_path / "scatter.png")
    assert out.endswith(".png")
    assert Path(out).stat().st_size > 0
