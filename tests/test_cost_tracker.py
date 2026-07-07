"""Tests for cost tracking and budget enforcement."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from bicameral_agent.cost_tracker import (
    MODEL_PRICING,
    CostBudgetExceeded,
    CostReport,
    CostTrackedClient,
    CostTracker,
)

_MODEL = "gemini-3.1-flash-lite-preview"
_PRICING = MODEL_PRICING[_MODEL]


class TestRecordCall:
    """Test cost calculation for recorded API calls."""

    def test_single_call_cost_math(self):
        tracker = CostTracker()
        tracker.record_call(1000, 500, _MODEL)
        report = tracker.get_total()

        expected_input = 1000 * _PRICING.input_cost_per_token
        expected_output = 500 * _PRICING.output_cost_per_token

        assert report.input_cost == pytest.approx(expected_input)
        assert report.output_cost == pytest.approx(expected_output)
        assert report.total == pytest.approx(expected_input + expected_output)
        assert report.call_count == 1

    def test_ten_calls_match_manual_calculation(self):
        tracker = CostTracker()
        for _ in range(10):
            tracker.record_call(2000, 1000, _MODEL)
        report = tracker.get_total()

        expected_input = 10 * 2000 * _PRICING.input_cost_per_token
        expected_output = 10 * 1000 * _PRICING.output_cost_per_token

        assert report.input_cost == pytest.approx(expected_input)
        assert report.output_cost == pytest.approx(expected_output)
        assert report.total == pytest.approx(expected_input + expected_output)
        assert report.call_count == 10

    def test_unknown_model_raises(self):
        tracker = CostTracker()
        with pytest.raises(ValueError, match="Unknown model"):
            tracker.record_call(100, 100, "not-a-real-model")


class TestGetTotal:
    """Test session-level cost reporting."""

    def test_empty_returns_zeros(self):
        tracker = CostTracker()
        report = tracker.get_total()
        assert report == CostReport(
            input_cost=0.0, output_cost=0.0, total=0.0, call_count=0
        )

    def test_accumulates_across_calls(self):
        tracker = CostTracker()
        tracker.record_call(100, 200, _MODEL)
        tracker.record_call(300, 400, _MODEL)
        report = tracker.get_total()

        expected_input = (100 + 300) * _PRICING.input_cost_per_token
        expected_output = (200 + 400) * _PRICING.output_cost_per_token
        assert report.total == pytest.approx(expected_input + expected_output)
        assert report.call_count == 2


class TestGetEpisodeCost:
    """Test episode-level cost reporting."""

    def test_tracks_separately_from_session(self):
        tracker = CostTracker()
        tracker.record_call(100, 200, _MODEL)
        tracker.reset_episode()
        tracker.record_call(300, 400, _MODEL)

        episode = tracker.get_episode_cost()
        session = tracker.get_total()

        # Episode only has the second call
        assert episode.call_count == 1
        expected_ep_input = 300 * _PRICING.input_cost_per_token
        assert episode.input_cost == pytest.approx(expected_ep_input)

        # Session has both
        assert session.call_count == 2


class TestResetEpisode:
    """Test episode reset returns report and zeros accumulators."""

    def test_returns_report_and_zeros_episode(self):
        tracker = CostTracker()
        tracker.record_call(1000, 500, _MODEL)
        report = tracker.reset_episode()

        assert report.call_count == 1
        assert report.total > 0

        # Episode is now zero
        after = tracker.get_episode_cost()
        assert after == CostReport(
            input_cost=0.0, output_cost=0.0, total=0.0, call_count=0
        )

    def test_preserves_session_total(self):
        tracker = CostTracker()
        tracker.record_call(1000, 500, _MODEL)
        before_session = tracker.get_total()
        tracker.reset_episode()

        assert tracker.get_total() == before_session


class TestSetBudget:
    """Test session-level budget enforcement."""

    def test_budget_exceeded_raises(self):
        tracker = CostTracker()
        tracker.set_budget(0.01)
        # Record enough calls to exceed $0.01
        # 10k output tokens * $3/1M = $0.03
        tracker.record_call(0, 10_000, _MODEL)
        with pytest.raises(CostBudgetExceeded):
            tracker.check_budget()

    def test_no_budget_no_exception(self):
        tracker = CostTracker()
        tracker.record_call(0, 100_000, _MODEL)
        tracker.check_budget()  # should not raise


class TestSetEpisodeBudget:
    """Test episode-level budget enforcement."""

    def test_episode_budget_enforced(self):
        tracker = CostTracker()
        tracker.set_episode_budget(0.001)
        tracker.record_call(0, 10_000, _MODEL)
        with pytest.raises(CostBudgetExceeded, match="Episode cost"):
            tracker.check_budget()

    def test_resets_with_episode(self):
        tracker = CostTracker()
        tracker.set_episode_budget(0.001)
        tracker.record_call(0, 10_000, _MODEL)
        tracker.reset_episode()
        # After reset, episode cost is zero so budget is no longer exceeded
        tracker.check_budget()  # should not raise


class TestThreadSafety:
    """Test concurrent access to CostTracker."""

    def test_concurrent_record_call(self):
        tracker = CostTracker()
        num_threads = 10
        calls_per_thread = 100

        def worker():
            for _ in range(calls_per_thread):
                tracker.record_call(100, 200, _MODEL)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        report = tracker.get_total()
        expected_calls = num_threads * calls_per_thread
        assert report.call_count == expected_calls

        expected_input = expected_calls * 100 * _PRICING.input_cost_per_token
        expected_output = expected_calls * 200 * _PRICING.output_cost_per_token
        assert report.input_cost == pytest.approx(expected_input)
        assert report.output_cost == pytest.approx(expected_output)


class TestCostTrackedClient:
    """Test the CostTrackedClient wrapper."""

    def _make_mock_client(self):
        client = MagicMock()
        client.model = _MODEL
        response = MagicMock()
        response.input_tokens = 500
        response.output_tokens = 200
        client.generate.return_value = response
        return client, response

    def test_check_budget_before_generate(self):
        client, _ = self._make_mock_client()
        tracker = CostTracker()
        tracker.set_budget(0.0)  # zero budget — immediate exceed
        # Need at least one recorded call to be "at" budget
        # Actually, 0.0 >= 0.0 is true, so check_budget raises immediately
        wrapped = CostTrackedClient(client, tracker)
        with pytest.raises(CostBudgetExceeded):
            wrapped.generate([])
        client.generate.assert_not_called()

    def test_record_call_after_generate(self):
        client, _ = self._make_mock_client()
        tracker = CostTracker()
        wrapped = CostTrackedClient(client, tracker)
        wrapped.generate([])

        report = tracker.get_total()
        assert report.call_count == 1
        assert report.input_cost == pytest.approx(
            500 * _PRICING.input_cost_per_token
        )

    def test_budget_exceeded_prevents_api_call(self):
        client, _ = self._make_mock_client()
        tracker = CostTracker()
        tracker.set_budget(0.001)
        wrapped = CostTrackedClient(client, tracker)

        # First call should succeed
        wrapped.generate([])

        # Record enough to exceed budget
        tracker.record_call(0, 100_000, _MODEL)

        with pytest.raises(CostBudgetExceeded):
            wrapped.generate([])
        # generate was only called once (the first time)
        assert client.generate.call_count == 1

    def test_model_property(self):
        client, _ = self._make_mock_client()
        tracker = CostTracker()
        wrapped = CostTrackedClient(client, tracker)
        assert wrapped.model == _MODEL


class _StubClient:
    """Minimal ModelClient stand-in with a fixed model tag and provider."""

    def __init__(self, model: str, provider: str | None = None) -> None:
        self.model = model
        if provider is not None:
            self.provider = provider

    def generate(self, *args, **kwargs):
        response = MagicMock()
        response.input_tokens = 1000
        response.output_tokens = 500
        return response


class TestCostTrackedClientFailFast:
    """Pricing is validated at construction, never after a paid call (issue #52)."""

    def test_unknown_model_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Unknown model"):
            CostTrackedClient(_StubClient("not-a-real-model"), CostTracker())

    def test_flat_rate_provider_unknown_tag_records_zero_cost(self):
        tracker = CostTracker()
        wrapped = CostTrackedClient(
            _StubClient("qwen8:8b-cloud", provider="ollama"), tracker
        )
        wrapped.generate([])
        report = tracker.get_total()
        assert report.call_count == 1
        assert report.total == 0.0


class TestCostConfig:
    """Test CostConfig loading and adapter method."""

    def test_config_loads_from_toml(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[cost]\nsession_budget = 5.0\nepisode_budget = 1.0\n'
            '[model]\nname = "gemini-3.1-flash-lite-preview"\n'
        )
        from bicameral_agent.config import HyperConfig

        cfg = HyperConfig.from_toml(toml_file)
        assert cfg.cost.session_budget == 5.0
        assert cfg.cost.episode_budget == 1.0

    def test_env_overrides(self):
        from bicameral_agent.config import HyperConfig

        cfg = HyperConfig()
        with patch.dict(
            "os.environ",
            {"BICAMERAL_COST__SESSION_BUDGET": "2.5"},
        ):
            cfg2 = cfg.with_env_overrides()
        assert cfg2.cost.session_budget == 2.5

    def test_adapter_method(self):
        from bicameral_agent.config import HyperConfig

        cfg = HyperConfig(
            cost={"session_budget": 10.0, "episode_budget": 2.0}
        )
        tracker = cfg.to_cost_tracker()
        assert isinstance(tracker, CostTracker)

        # Verify budgets were applied: record a huge cost and check
        tracker.record_call(0, 100_000_000, _MODEL)
        with pytest.raises(CostBudgetExceeded, match="Session cost"):
            tracker.check_budget()

    def test_default_no_budgets(self):
        from bicameral_agent.config import HyperConfig

        cfg = HyperConfig()
        assert cfg.cost.session_budget is None
        assert cfg.cost.episode_budget is None
        tracker = cfg.to_cost_tracker()
        tracker.record_call(0, 100_000_000, _MODEL)
        tracker.check_budget()  # should not raise
