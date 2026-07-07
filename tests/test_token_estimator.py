"""Tests for the per-tool token estimator (Issue #8)."""

import random
import threading
import time

import pytest

from bicameral_agent.token_estimator import (
    ContextFeatures,
    TokenEstimate,
    TokenEstimator,
    _TOOL_PROFILES,
)

# 5 conversation sizes for parametrized tests
_CONV_SIZES = [
    ContextFeatures(conversation_length_tokens=0, conversation_turn_count=0),
    ContextFeatures(conversation_length_tokens=500, conversation_turn_count=3),
    ContextFeatures(conversation_length_tokens=2000, conversation_turn_count=10),
    ContextFeatures(conversation_length_tokens=6000, conversation_turn_count=20),
    ContextFeatures(conversation_length_tokens=15000, conversation_turn_count=50),
]

_TOOL_IDS = ["research_gap_scanner", "assumption_auditor", "context_refresher"]


def _expected_scanner_input(conv: int) -> int:
    gaps = min(max(1, conv // 2000), 5)
    return (500 + conv) + (gaps * 2000) + (500 + conv + gaps * 2000)


def _expected_scanner_calls(conv: int) -> int:
    return 2 + min(max(1, conv // 2000), 5)


def _expected_auditor_input(conv: int) -> int:
    return 400 + conv


def _expected_refresher_input(conv: int, turns: int) -> int:
    avg_msg = conv / max(turns, 1)
    bounded = min(4 * avg_msg, conv)
    return 300 + int(bounded)


class TestContextFeatures:
    def test_fields_accessible(self):
        cf = ContextFeatures(conversation_length_tokens=1000, conversation_turn_count=5)
        assert cf.conversation_length_tokens == 1000
        assert cf.conversation_turn_count == 5

    def test_frozen(self):
        cf = ContextFeatures(conversation_length_tokens=1000, conversation_turn_count=5)
        with pytest.raises(AttributeError):
            cf.conversation_length_tokens = 2000


class TestTokenEstimate:
    def test_fields_accessible(self):
        te = TokenEstimate(input_tokens=100, output_tokens=200, num_calls=3)
        assert te.input_tokens == 100
        assert te.output_tokens == 200
        assert te.num_calls == 3

    def test_frozen(self):
        te = TokenEstimate(input_tokens=100, output_tokens=200, num_calls=3)
        with pytest.raises(AttributeError):
            te.input_tokens = 500


class TestInputTokenAccuracy:
    """AC1: Input token estimates match deterministic formulas."""

    @pytest.mark.parametrize(
        "ctx", _CONV_SIZES, ids=lambda c: f"conv={c.conversation_length_tokens}"
    )
    def test_scanner_input_tokens(self, token_estimator, ctx):
        est = token_estimator.estimate("research_gap_scanner", ctx)
        expected = _expected_scanner_input(ctx.conversation_length_tokens)
        assert est.input_tokens == expected

    @pytest.mark.parametrize(
        "ctx", _CONV_SIZES, ids=lambda c: f"conv={c.conversation_length_tokens}"
    )
    def test_auditor_input_tokens(self, token_estimator, ctx):
        est = token_estimator.estimate("assumption_auditor", ctx)
        expected = _expected_auditor_input(ctx.conversation_length_tokens)
        assert est.input_tokens == expected

    @pytest.mark.parametrize(
        "ctx", _CONV_SIZES, ids=lambda c: f"conv={c.conversation_length_tokens}"
    )
    def test_refresher_input_tokens(self, token_estimator, ctx):
        est = token_estimator.estimate("context_refresher", ctx)
        expected = _expected_refresher_input(
            ctx.conversation_length_tokens, ctx.conversation_turn_count
        )
        assert est.input_tokens == expected


class TestOutputTokenConvergence:
    """AC2: After 15 noisy observations, 70%+ estimates within 30% of actual."""

    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    def test_convergence(self, tool_id):
        rng = random.Random(42)
        estimator = TokenEstimator()

        true_output_per_call = 300
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)

        # Compute num_calls for this tool/context
        profile = _TOOL_PROFILES[tool_id]
        num_calls = estimator._compute_num_calls(
            profile, ctx.conversation_length_tokens
        )

        # Train with 15 noisy observations
        for _ in range(15):
            noise = rng.gauss(0, true_output_per_call * 0.2)
            actual_per_call = max(true_output_per_call + noise, 1)
            estimator.observe_tool(tool_id, ctx, int(actual_per_call * num_calls))

        # Check convergence: estimate should be within 30% of noisy test samples
        within_30_pct = 0
        n_test = 20
        for _ in range(n_test):
            noise = rng.gauss(0, true_output_per_call * 0.2)
            actual_per_call = max(true_output_per_call + noise, 1)
            est = estimator.estimate(tool_id, ctx)
            est_per_call = est.output_tokens / max(est.num_calls, 1)

            if abs(est_per_call - actual_per_call) / actual_per_call <= 0.30:
                within_30_pct += 1

        assert within_30_pct / n_test >= 0.70


class TestNumCalls:
    """AC3: num_calls is 1 for auditor/refresher, variable for scanner."""

    @pytest.mark.parametrize(
        "ctx", _CONV_SIZES, ids=lambda c: f"conv={c.conversation_length_tokens}"
    )
    def test_auditor_always_one(self, token_estimator, ctx):
        est = token_estimator.estimate("assumption_auditor", ctx)
        assert est.num_calls == 1

    @pytest.mark.parametrize(
        "ctx", _CONV_SIZES, ids=lambda c: f"conv={c.conversation_length_tokens}"
    )
    def test_refresher_always_one(self, token_estimator, ctx):
        est = token_estimator.estimate("context_refresher", ctx)
        assert est.num_calls == 1

    @pytest.mark.parametrize(
        "ctx", _CONV_SIZES, ids=lambda c: f"conv={c.conversation_length_tokens}"
    )
    def test_scanner_variable(self, token_estimator, ctx):
        est = token_estimator.estimate("research_gap_scanner", ctx)
        expected = _expected_scanner_calls(ctx.conversation_length_tokens)
        assert est.num_calls == expected

    def test_scanner_increases_with_conversation(self, token_estimator):
        small = ContextFeatures(conversation_length_tokens=500, conversation_turn_count=3)
        large = ContextFeatures(
            conversation_length_tokens=12000, conversation_turn_count=40
        )
        est_small = token_estimator.estimate("research_gap_scanner", small)
        est_large = token_estimator.estimate("research_gap_scanner", large)
        assert est_large.num_calls >= est_small.num_calls


class TestObserveUpdates:
    """AC4: Predictions shift toward observed values."""

    def test_shifts_up(self):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)
        before = estimator.estimate("assumption_auditor", ctx)

        # Observe much higher output
        for _ in range(10):
            estimator.observe_tool("assumption_auditor", ctx, 2000)

        after = estimator.estimate("assumption_auditor", ctx)
        assert after.output_tokens > before.output_tokens

    def test_shifts_down(self):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)
        before = estimator.estimate("assumption_auditor", ctx)

        # Observe much lower output
        for _ in range(10):
            estimator.observe_tool("assumption_auditor", ctx, 10)

        after = estimator.estimate("assumption_auditor", ctx)
        assert after.output_tokens < before.output_tokens

    def test_first_observation_replaces_default(self):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)

        # Default for auditor is 450
        estimator.observe_tool("assumption_auditor", ctx, 1000)
        est = estimator.estimate("assumption_auditor", ctx)

        # First observation should fully replace default
        assert est.output_tokens == 1000


class TestPerformance:
    """AC5: 10K estimate() calls in < 2ms average."""

    def test_estimate_speed(self, token_estimator):
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)

        # Warm up
        for _ in range(100):
            token_estimator.estimate("assumption_auditor", ctx)

        n_calls = 10_000
        start = time.perf_counter()
        for _ in range(n_calls):
            token_estimator.estimate("assumption_auditor", ctx)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_calls) * 1000
        assert avg_ms < 2.0, f"Average estimate() took {avg_ms:.3f}ms, expected < 2ms"


class TestColdStartDefaults:
    """AC6: All tools return sensible positive defaults with 0 observations."""

    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    def test_positive_defaults(self, tool_id):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=2000, conversation_turn_count=10)
        est = estimator.estimate(tool_id, ctx)

        assert est.input_tokens > 0
        assert est.output_tokens > 0
        assert est.num_calls >= 1

    @pytest.mark.parametrize("tool_id", _TOOL_IDS)
    def test_defaults_with_empty_conversation(self, tool_id):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=0, conversation_turn_count=0)
        est = estimator.estimate(tool_id, ctx)

        assert est.input_tokens > 0
        assert est.output_tokens > 0
        assert est.num_calls >= 1


class TestUnknownTool:
    def test_estimate_raises(self):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=1000, conversation_turn_count=5)
        with pytest.raises(ValueError, match="Unknown tool"):
            estimator.estimate("nonexistent_tool", ctx)

    def test_observe_raises(self):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=1000, conversation_turn_count=5)
        with pytest.raises(ValueError, match="Unknown tool"):
            estimator.observe_tool("nonexistent_tool", ctx, 100)


class TestThreadSafety:
    def test_concurrent_estimate_and_observe(self):
        estimator = TokenEstimator()
        ctx = ContextFeatures(conversation_length_tokens=3000, conversation_turn_count=10)
        errors = []

        def observer():
            try:
                for i in range(100):
                    estimator.observe_tool("assumption_auditor", ctx, 400 + i)
            except Exception as e:
                errors.append(e)

        def predictor():
            try:
                for _ in range(100):
                    est = estimator.estimate("assumption_auditor", ctx)
                    assert est.output_tokens > 0
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
