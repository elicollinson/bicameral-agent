"""Tests for the composite tool latency model (Issue #9)."""

import threading
import time

import numpy as np
import pytest

from bicameral_agent.latency import APILatencyModel, LatencyEstimate
from bicameral_agent.token_estimator import ContextFeatures, TokenEstimate, TokenEstimator
from bicameral_agent.tool_latency import (
    CostEstimate,
    SubCallPrediction,
    ToolLatencyModel,
    ToolPrediction,
)

_TOOL_IDS = ["research_gap_scanner", "assumption_auditor", "context_refresher"]

_CONV_SIZES = [
    ContextFeatures(conversation_length_tokens=500, conversation_turn_count=3),
    ContextFeatures(conversation_length_tokens=2000, conversation_turn_count=10),
    ContextFeatures(conversation_length_tokens=4000, conversation_turn_count=15),
    ContextFeatures(conversation_length_tokens=6000, conversation_turn_count=20),
    ContextFeatures(conversation_length_tokens=12000, conversation_turn_count=40),
]


class TestComposition:
    """AC1: predict_tool_duration correctly composes both layers."""

    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    @pytest.mark.parametrize(
        "ctx", _CONV_SIZES, ids=lambda c: f"conv={c.conversation_length_tokens}"
    )
    def test_duration_positive_and_ordered(self, tool_id, ctx):
        """Duration is positive and p25 <= mean <= p75."""
        model = ToolLatencyModel()
        est = model.predict_tool_duration(tool_id, ctx)
        assert est.mean_ms > 0
        assert est.p25_ms <= est.mean_ms <= est.p75_ms

    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    def test_sub_calls_latencies_sum_to_aggregate_mean(self, tool_id):
        """The sum of sub-call means equals the aggregate mean."""
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=4000, conversation_turn_count=15)
        pred = model.predict(tool_id, ctx)
        sub_mean_sum = sum(sc.latency.mean_ms for sc in pred.sub_calls)
        assert abs(pred.latency.mean_ms - sub_mean_sum) < 1e-6

    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    def test_sub_call_tokens_match_aggregate(self, tool_id):
        """Sum of sub-call input tokens matches aggregate TokenEstimate."""
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=4000, conversation_turn_count=15)
        pred = model.predict(tool_id, ctx)
        total_input = sum(sc.input_tokens for sc in pred.sub_calls)
        assert total_input == pred.token_estimate.input_tokens


class TestScannerDecomposition:
    """AC2: Scanner with predicted 2 gaps decomposes correctly."""

    def test_scanner_2_gaps_structure(self):
        """At conv=4000, gaps=2, so 4 sub-calls: gap + 2 search + synthesis."""
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=4000, conversation_turn_count=15)
        pred = model.predict("research_gap_scanner", ctx)

        assert pred.token_estimate.num_calls == 4
        assert len(pred.sub_calls) == 4
        assert pred.sub_calls[0].label == "gap_identification"
        assert pred.sub_calls[1].label == "search_1"
        assert pred.sub_calls[2].label == "search_2"
        assert pred.sub_calls[3].label == "synthesis"

    def test_scanner_2_gaps_input_tokens(self):
        """Verify per-call input token counts match the formulas."""
        model = ToolLatencyModel()
        conv = 4000
        ctx = ContextFeatures(conversation_length_tokens=conv, conversation_turn_count=15)
        pred = model.predict("research_gap_scanner", ctx)

        assert pred.sub_calls[0].input_tokens == 500 + conv
        assert pred.sub_calls[1].input_tokens == 2000
        assert pred.sub_calls[2].input_tokens == 2000
        assert pred.sub_calls[3].input_tokens == 500 + conv + 2 * 2000

    def test_scanner_2_gaps_mean_is_sum(self):
        """Total mean latency = sum of 4 individual call means."""
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=4000, conversation_turn_count=15)
        pred = model.predict("research_gap_scanner", ctx)

        expected_mean = sum(sc.latency.mean_ms for sc in pred.sub_calls)
        assert abs(pred.latency.mean_ms - expected_mean) < 1e-6

    def test_scanner_varying_gaps(self):
        """Scanner sub-call count scales with conversation length."""
        model = ToolLatencyModel()
        small = ContextFeatures(conversation_length_tokens=500, conversation_turn_count=3)
        large = ContextFeatures(
            conversation_length_tokens=12000, conversation_turn_count=40
        )
        pred_small = model.predict("research_gap_scanner", small)
        pred_large = model.predict("research_gap_scanner", large)
        assert len(pred_large.sub_calls) >= len(pred_small.sub_calls)


class TestCostEstimates:
    """AC3: Cost estimates match manual calculation from Gemini pricing."""

    _INPUT_RATE = 0.50 / 1_000_000
    _OUTPUT_RATE = 3.00 / 1_000_000

    @pytest.mark.parametrize("conv", [500, 4000, 12000])
    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    def test_cost_matches_manual(self, tool_id, conv):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=conv, conversation_turn_count=10)
        pred = model.predict(tool_id, ctx)
        cost = pred.cost

        expected_input = pred.token_estimate.input_tokens * self._INPUT_RATE
        expected_output = pred.token_estimate.output_tokens * self._OUTPUT_RATE
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-12
        assert abs(cost.output_cost - expected_output) < 1e-12
        assert abs(cost.total - expected_total) < 1e-12

    def test_cost_positive(self):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=2000, conversation_turn_count=10)
        cost = model.predict_cost("assumption_auditor", ctx)
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.total > 0
        assert abs(cost.total - cost.input_cost - cost.output_cost) < 1e-12

    def test_predict_cost_matches_predict(self):
        """predict_cost() returns same result as predict().cost."""
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=4000, conversation_turn_count=15)
        full = model.predict("assumption_auditor", ctx)
        cost_only = model.predict_cost("assumption_auditor", ctx)
        assert full.cost.input_cost == cost_only.input_cost
        assert full.cost.output_cost == cost_only.output_cost
        assert full.cost.total == cost_only.total


class TestPerformance:
    """AC4: Full prediction (both layers) completes in < 5ms."""

    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    def test_predict_speed(self, tool_id):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=6000, conversation_turn_count=20)

        # Warm up
        for _ in range(100):
            model.predict(tool_id, ctx)

        n_calls = 5_000
        start = time.perf_counter()
        for _ in range(n_calls):
            model.predict(tool_id, ctx)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_calls) * 1000
        assert avg_ms < 5.0, f"Average predict() took {avg_ms:.3f}ms, expected < 5ms"


class TestRankOrdering:
    """AC5: Model correctly predicts rank ordering of tool latencies."""

    def test_rank_ordering_on_synthetic_data(self, tool_latency_model):
        """Given synthetic latency training, model ranks tools correctly >= 80%."""
        rng = np.random.default_rng(99)

        correct = 0
        n_test = 50
        for _ in range(n_test):
            conv = int(rng.integers(1000, 8000))
            ctx = ContextFeatures(
                conversation_length_tokens=conv,
                conversation_turn_count=conv // 200,
            )
            refresher = tool_latency_model.predict_tool_duration("context_refresher", ctx)
            scanner = tool_latency_model.predict_tool_duration("research_gap_scanner", ctx)
            if scanner.mean_ms > refresher.mean_ms:
                correct += 1

        assert correct / n_test >= 0.80


class TestObserveDelegation:
    """AC6: observe methods correctly delegate to sub-models."""

    def test_observe_delegates_to_latency_model(self):
        latency_model = APILatencyModel()
        model = ToolLatencyModel(latency_model=latency_model)
        assert latency_model.observation_count == 0
        model.observe(1000, 500, 2000.0)
        assert latency_model.observation_count == 1

    def test_observe_tool_delegates_to_token_estimator(self):
        token_est = TokenEstimator()
        model = ToolLatencyModel(token_estimator=token_est)
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)

        before = token_est.estimate("assumption_auditor", ctx)
        model.observe_tool("assumption_auditor", ctx, 1000)
        after = token_est.estimate("assumption_auditor", ctx)

        assert after.output_tokens != before.output_tokens
        assert after.output_tokens == 1000

    def test_observe_affects_subsequent_predictions(self):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)

        before = model.predict("assumption_auditor", ctx)

        for _ in range(10):
            model.observe_tool("assumption_auditor", ctx, 2000)

        after = model.predict("assumption_auditor", ctx)
        assert after.cost.output_cost > before.cost.output_cost


class TestDataClasses:
    def test_cost_estimate_frozen(self):
        ce = CostEstimate(input_cost=0.001, output_cost=0.002, total=0.003)
        with pytest.raises(AttributeError):
            ce.total = 0.0

    def test_sub_call_prediction_frozen(self):
        sc = SubCallPrediction(
            label="test",
            input_tokens=100,
            output_tokens=200,
            latency=LatencyEstimate(mean_ms=100.0, p25_ms=80.0, p75_ms=120.0),
        )
        with pytest.raises(AttributeError):
            sc.label = "changed"

    def test_tool_prediction_frozen(self):
        pred = ToolPrediction(
            tool_id="test",
            latency=LatencyEstimate(mean_ms=100.0, p25_ms=80.0, p75_ms=120.0),
            cost=CostEstimate(input_cost=0.001, output_cost=0.002, total=0.003),
            token_estimate=TokenEstimate(input_tokens=100, output_tokens=200, num_calls=1),
            sub_calls=(),
        )
        with pytest.raises(AttributeError):
            pred.tool_id = "changed"


class TestSingleCallTools:
    """Single-call tools have exactly one sub-call."""

    @pytest.mark.parametrize("tool_id", ["assumption_auditor", "context_refresher"])
    def test_single_sub_call(self, tool_id):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)
        pred = model.predict(tool_id, ctx)
        assert len(pred.sub_calls) == 1
        assert pred.sub_calls[0].label == tool_id


class TestUnknownTool:
    def test_predict_raises(self):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=1000, conversation_turn_count=5)
        with pytest.raises(ValueError, match="Unknown tool"):
            model.predict("nonexistent", ctx)

    def test_predict_tool_duration_raises(self):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=1000, conversation_turn_count=5)
        with pytest.raises(ValueError, match="Unknown tool"):
            model.predict_tool_duration("nonexistent", ctx)

    def test_predict_cost_raises(self):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=1000, conversation_turn_count=5)
        with pytest.raises(ValueError, match="Unknown tool"):
            model.predict_cost("nonexistent", ctx)


class TestThreadSafety:
    def test_concurrent_predict_and_observe(self):
        model = ToolLatencyModel()
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)
        errors = []

        def observer():
            try:
                for i in range(50):
                    model.observe(1000, 500, 2000.0 + i)
                    model.observe_tool("assumption_auditor", ctx, 400 + i)
            except Exception as e:
                errors.append(e)

        def predictor():
            try:
                for _ in range(50):
                    pred = model.predict("research_gap_scanner", ctx)
                    assert pred.latency.mean_ms > 0
                    assert pred.cost.total > 0
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=observer),
            threading.Thread(target=observer),
            threading.Thread(target=predictor),
            threading.Thread(target=predictor),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread safety errors: {errors}"
