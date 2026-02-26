"""Tests for the API latency model (Issue #7)."""

import math
import threading
import time

import numpy as np
import pytest

from bicameral_agent.latency import APILatencyModel, LatencyEstimate

# Known synthetic model: duration = 200 + 0.01*inp + 15*out + noise
_TRUE_ALPHA = 200.0
_TRUE_BETA = 0.01
_TRUE_GAMMA = 15.0
_TRUE_NOISE_STD = 100.0


def _true_duration(rng, input_tokens: int, output_tokens: int) -> float:
    """Generate a duration from the known synthetic model."""
    return max(
        _TRUE_ALPHA
        + _TRUE_BETA * input_tokens
        + _TRUE_GAMMA * output_tokens
        + rng.normal(0, _TRUE_NOISE_STD),
        1.0,
    )


def _assert_finite_positive(est: LatencyEstimate) -> None:
    """Assert that all fields of a LatencyEstimate are finite and the mean is positive."""
    assert est.mean_ms > 0
    assert not math.isnan(est.mean_ms)
    assert not math.isinf(est.mean_ms)
    assert not math.isnan(est.p25_ms)
    assert not math.isnan(est.p75_ms)


class TestLatencyEstimate:
    def test_fields_accessible(self):
        est = LatencyEstimate(mean_ms=100.0, p25_ms=80.0, p75_ms=120.0)
        assert est.mean_ms == 100.0
        assert est.p25_ms == 80.0
        assert est.p75_ms == 120.0

    def test_frozen(self):
        est = LatencyEstimate(mean_ms=100.0, p25_ms=80.0, p75_ms=120.0)
        with pytest.raises(AttributeError):
            est.mean_ms = 200.0


class TestColdStart:
    """AC1: With 0 observations, predict() returns conservative estimates (>= 2x baseline)."""

    def test_zero_observations_returns_conservative(self):
        model = APILatencyModel()
        est = model.predict(1000, 500)
        reasonable_baseline = _TRUE_ALPHA + _TRUE_BETA * 1000 + _TRUE_GAMMA * 500
        assert est.mean_ms >= 2 * reasonable_baseline

    def test_zero_observations_has_wide_spread(self):
        model = APILatencyModel()
        est = model.predict(1000, 500)
        spread = est.p75_ms - est.p25_ms
        assert spread > 0.3 * est.mean_ms

    def test_few_observations_still_biased_toward_priors(self):
        model = APILatencyModel()
        rng = np.random.default_rng(42)
        for _ in range(5):
            inp, out = 1000, 500
            model.observe(inp, out, _true_duration(rng, inp, out))

        est = model.predict(1000, 500)
        reasonable_baseline = _TRUE_ALPHA + _TRUE_BETA * 1000 + _TRUE_GAMMA * 500
        assert est.mean_ms > 1.5 * reasonable_baseline


class TestConvergence:
    """AC2: After 50 synthetic observations, predictions within 25% of actual >= 70%."""

    def test_convergence_with_synthetic_data(self, trained_latency_model):
        rng = np.random.default_rng(99)
        within_25_pct = 0
        n_test = 200

        for _ in range(n_test):
            inp = int(rng.integers(100, 5000))
            out = int(rng.integers(50, 2000))
            true = _true_duration(rng, inp, out)
            est = trained_latency_model.predict(inp, out)

            if abs(est.mean_ms - true) / true <= 0.25:
                within_25_pct += 1

        assert within_25_pct / n_test >= 0.70


class TestPercentileCalibration:
    """AC3: Actual durations fall between p25 and p75 ~50% of the time."""

    def test_calibration(self):
        model = APILatencyModel()
        rng = np.random.default_rng(42)

        # Train with 100 observations
        for _ in range(100):
            inp = int(rng.integers(100, 5000))
            out = int(rng.integers(50, 2000))
            model.observe(inp, out, _true_duration(rng, inp, out))

        # Test calibration on 200 new observations
        in_range = 0
        n_test = 200
        for _ in range(n_test):
            inp = int(rng.integers(100, 5000))
            out = int(rng.integers(50, 2000))
            actual = _true_duration(rng, inp, out)
            est = model.predict(inp, out)

            if est.p25_ms <= actual <= est.p75_ms:
                in_range += 1

        # Should be roughly 50%, allow 30-70% range for statistical variation
        ratio = in_range / n_test
        assert 0.30 <= ratio <= 0.70, f"Calibration ratio {ratio:.2%} outside [30%, 70%]"


class TestOnlineUpdate:
    """AC4: observe() integrates new data without full refit."""

    def test_predictions_shift_after_observe(self):
        model = APILatencyModel()
        before = model.predict(1000, 500)

        # Observe a very fast duration — should pull predictions down
        for _ in range(25):
            model.observe(1000, 500, 100.0)

        after = model.predict(1000, 500)
        assert after.mean_ms < before.mean_ms

    def test_fixed_size_state(self):
        """Model does not grow in memory with observations."""
        model = APILatencyModel()
        rng = np.random.default_rng(42)

        for _ in range(1000):
            inp = int(rng.integers(100, 5000))
            out = int(rng.integers(50, 2000))
            model.observe(inp, out, _true_duration(rng, inp, out))

        # Internal state is fixed-size: 3x3 matrix + 3-vector + 3 scalars
        assert model._xtx.shape == (3, 3)
        assert model._xty.shape == (3,)


class TestPredictionPerformance:
    """AC5: predict() completes in < 1ms."""

    def test_predict_speed(self, trained_latency_model):
        # Warm up
        for _ in range(100):
            trained_latency_model.predict(1000, 500)

        n_calls = 10_000
        start = time.perf_counter()
        for _ in range(n_calls):
            trained_latency_model.predict(1000, 500)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_calls) * 1000
        assert avg_ms < 1.0, f"Average predict() took {avg_ms:.3f}ms, expected < 1ms"


class TestObservationCount:
    """AC6: observation_count correctly tracks number of data points."""

    def test_starts_at_zero(self):
        model = APILatencyModel()
        assert model.observation_count == 0

    def test_increments_on_observe(self):
        model = APILatencyModel()
        model.observe(100, 50, 500.0)
        assert model.observation_count == 1
        model.observe(200, 100, 1000.0)
        assert model.observation_count == 2

    def test_after_many(self):
        model = APILatencyModel()
        for i in range(100):
            model.observe(100, 50, 500.0)
        assert model.observation_count == 100


class TestEdgeCases:
    """AC7: Handles 0 tokens, very large token counts (1M+)."""

    def test_zero_input_tokens(self):
        model = APILatencyModel()
        _assert_finite_positive(model.predict(0, 500))

    def test_zero_output_tokens(self):
        model = APILatencyModel()
        _assert_finite_positive(model.predict(1000, 0))

    def test_zero_both(self):
        model = APILatencyModel()
        _assert_finite_positive(model.predict(0, 0))

    def test_very_large_input_tokens(self):
        model = APILatencyModel()
        _assert_finite_positive(model.predict(1_000_000, 500))

    def test_very_large_output_tokens(self):
        model = APILatencyModel()
        _assert_finite_positive(model.predict(1000, 1_000_000))

    def test_both_very_large(self):
        model = APILatencyModel()
        _assert_finite_positive(model.predict(1_000_000, 1_000_000))

    def test_large_tokens_after_training(self):
        """Large token counts don't cause overflow after training."""
        model = APILatencyModel()
        rng = np.random.default_rng(42)
        for _ in range(50):
            inp = int(rng.integers(100, 5000))
            out = int(rng.integers(50, 2000))
            model.observe(inp, out, _true_duration(rng, inp, out))

        _assert_finite_positive(model.predict(1_000_000, 1_000_000))


class TestThreadSafety:
    def test_concurrent_observe_and_predict(self):
        model = APILatencyModel()
        errors = []

        def observer():
            rng = np.random.default_rng(42)
            try:
                for _ in range(100):
                    inp = int(rng.integers(100, 5000))
                    out = int(rng.integers(50, 2000))
                    model.observe(inp, out, _true_duration(rng, inp, out))
            except Exception as e:
                errors.append(e)

        def predictor():
            try:
                for _ in range(100):
                    est = model.predict(1000, 500)
                    assert est.mean_ms > 0
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
