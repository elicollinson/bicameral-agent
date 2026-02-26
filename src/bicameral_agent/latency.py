"""API latency model with online learning for token-based prediction.

Models total API call latency as a linear function of input and output
token counts, using incremental OLS with Welford's variance tracking
for percentile estimation. Uses conservative priors during cold start
(< 20 observations) and smoothly transitions to learned parameters.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

# Cold-start threshold: below this, blend priors with learned parameters
_COLD_START_THRESHOLD = 20

# Conservative priors (deliberately ~2x overestimate)
_PRIOR_ALPHA = 1000.0  # TTFT intercept (ms)
_PRIOR_BETA = 0.02  # ms per input token
_PRIOR_OUTPUT_RATE = 25.0  # tokens per second (conservative; reasonable ~50)
_PRIOR_OVERHEAD = 200.0  # fixed overhead (ms)
_PRIOR_VARIANCE_FRAC = 0.5  # residual std as fraction of predicted mean

# Normal distribution z-score for 25th percentile
_Z_25 = 0.6745


def _feature_vector(input_tokens: int, output_tokens: int) -> np.ndarray:
    """Build the [1, input_tokens, output_tokens] feature vector."""
    return np.array([1.0, float(input_tokens), float(output_tokens)], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LatencyEstimate:
    """Predicted latency distribution for an API call."""

    mean_ms: float
    p25_ms: float
    p75_ms: float


class APILatencyModel:
    """Online latency model for API calls.

    Models total latency as a linear function of input and output tokens:
        total_ms = alpha + beta * input_tokens + gamma * output_tokens

    Uses incremental OLS (sufficient statistics) for parameter estimation
    and Welford's algorithm for residual variance tracking.

    With fewer than 20 observations, predictions are blended with
    conservative priors that overestimate by ~2x.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n: int = 0

        # Sufficient statistics for incremental OLS: X^T X and X^T y
        # Ridge initialization (1e-6) prevents singular matrix early on
        self._xtx = np.eye(3, dtype=np.float64) * 1e-6
        self._xty = np.zeros(3, dtype=np.float64)

        # Welford's algorithm for residual variance
        self._resid_mean: float = 0.0
        self._resid_m2: float = 0.0

    @property
    def observation_count(self) -> int:
        """Number of data points observed."""
        with self._lock:
            return self._n

    def predict(self, input_tokens: int, output_tokens: int) -> LatencyEstimate:
        """Predict latency distribution for given token counts.

        Args:
            input_tokens: Number of input tokens for the API call.
            output_tokens: Number of output tokens expected.

        Returns:
            LatencyEstimate with mean, p25, and p75 in milliseconds.
        """
        with self._lock:
            return self._predict_locked(input_tokens, output_tokens)

    def observe(
        self, input_tokens: int, output_tokens: int, actual_duration_ms: float
    ) -> None:
        """Record an observed latency for online parameter update.

        Args:
            input_tokens: Number of input tokens in the API call.
            output_tokens: Number of output tokens produced.
            actual_duration_ms: Observed wall-clock duration in milliseconds.
        """
        with self._lock:
            x = _feature_vector(input_tokens, output_tokens)
            self._xtx += np.outer(x, x)
            self._xty += x * actual_duration_ms
            self._n += 1

            # Residual against learned OLS (not blended) to avoid inflating
            # variance with cold-start prior mismatch
            params = np.linalg.solve(self._xtx, self._xty)
            resid = actual_duration_ms - float(x @ params)

            # Welford's online variance update
            delta = resid - self._resid_mean
            self._resid_mean += delta / self._n
            delta2 = resid - self._resid_mean
            self._resid_m2 += delta * delta2

    def _predict_locked(
        self, input_tokens: int, output_tokens: int
    ) -> LatencyEstimate:
        """Predict without acquiring lock (caller must hold lock)."""
        w = min(self._n / _COLD_START_THRESHOLD, 1.0)

        prior_mean = (
            _PRIOR_ALPHA
            + _PRIOR_BETA * input_tokens
            + (output_tokens / _PRIOR_OUTPUT_RATE) * 1000
            + _PRIOR_OVERHEAD
        )

        if self._n == 0:
            mean_ms = prior_mean
            spread = prior_mean * _PRIOR_VARIANCE_FRAC
        else:
            params = np.linalg.solve(self._xtx, self._xty)
            x = _feature_vector(input_tokens, output_tokens)
            learned_mean = max(float(x @ params), 1.0)

            mean_ms = (1 - w) * prior_mean + w * learned_mean

            learned_std = (
                math.sqrt(self._resid_m2 / self._n) if self._resid_m2 > 0 else 0.0
            )
            prior_std = prior_mean * _PRIOR_VARIANCE_FRAC
            spread = (1 - w) * prior_std + w * learned_std

        p25 = max(mean_ms - _Z_25 * spread, 0.0)
        p75 = mean_ms + _Z_25 * spread

        return LatencyEstimate(mean_ms=mean_ms, p25_ms=p25, p75_ms=p75)
