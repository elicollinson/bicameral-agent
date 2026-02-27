"""Composite predictor combining token estimation, latency prediction, and cost model.

Composes TokenEstimator (Layer 1) and APILatencyModel (Layer 2) into a unified
interface for predicting tool invocation duration and cost. Decomposes multi-call
tools into per-sub-call predictions with sequential latency summation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bicameral_agent.latency import APILatencyModel, LatencyEstimate
from bicameral_agent.token_estimator import ContextFeatures, TokenEstimate, TokenEstimator

# Normal distribution z-score for 25th percentile
_Z_25 = 0.6745

# Gemini pricing
_INPUT_COST_PER_TOKEN = 0.50 / 1_000_000  # $0.50 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 3.00 / 1_000_000  # $3.00 per 1M output tokens


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Predicted cost for a tool invocation in USD."""

    input_cost: float
    output_cost: float
    total: float


@dataclass(frozen=True, slots=True)
class SubCallPrediction:
    """Per-sub-call prediction detail for debugging/logging."""

    label: str
    input_tokens: int
    output_tokens: int
    latency: LatencyEstimate


@dataclass(frozen=True, slots=True)
class ToolPrediction:
    """Full prediction for a tool invocation with sub-call details."""

    tool_id: str
    latency: LatencyEstimate
    cost: CostEstimate
    token_estimate: TokenEstimate
    sub_calls: tuple[SubCallPrediction, ...]


class ToolLatencyModel:
    """Composite predictor combining token estimation, latency, and cost.

    Composes TokenEstimator and APILatencyModel to predict end-to-end
    tool invocation duration and cost. For multi-call tools, decomposes
    into per-sub-call predictions and sums latencies sequentially.

    Thread-safe: delegates locking to sub-models.
    """

    def __init__(
        self,
        token_estimator: TokenEstimator | None = None,
        latency_model: APILatencyModel | None = None,
    ) -> None:
        self._token_estimator = token_estimator or TokenEstimator()
        self._latency_model = latency_model or APILatencyModel()

    @property
    def token_estimator(self) -> TokenEstimator:
        """Access the underlying token estimator."""
        return self._token_estimator

    @property
    def latency_model(self) -> APILatencyModel:
        """Access the underlying API latency model."""
        return self._latency_model

    def predict(self, tool_id: str, context_features: ContextFeatures) -> ToolPrediction:
        """Full prediction with sub-call decomposition.

        Args:
            tool_id: Identifier of the tool to predict for.
            context_features: Current conversation context.

        Returns:
            ToolPrediction with latency, cost, and per-sub-call details.

        Raises:
            ValueError: If tool_id is not in the registry.
        """
        token_est = self._token_estimator.estimate(tool_id, context_features)
        sub_calls = self._decompose_calls(tool_id, context_features, token_est)
        latency = self._aggregate_latencies(sub_calls)
        cost = self._compute_cost(token_est)

        return ToolPrediction(
            tool_id=tool_id,
            latency=latency,
            cost=cost,
            token_estimate=token_est,
            sub_calls=sub_calls,
        )

    def predict_tool_duration(
        self, tool_id: str, context_features: ContextFeatures
    ) -> LatencyEstimate:
        """Predict aggregated latency estimate for a tool invocation.

        Args:
            tool_id: Identifier of the tool to predict for.
            context_features: Current conversation context.

        Returns:
            Aggregated LatencyEstimate across all sub-calls.
        """
        return self.predict(tool_id, context_features).latency

    def predict_cost(
        self, tool_id: str, context_features: ContextFeatures
    ) -> CostEstimate:
        """Predict cost estimate for a tool invocation.

        Args:
            tool_id: Identifier of the tool to predict for.
            context_features: Current conversation context.

        Returns:
            CostEstimate with input, output, and total cost in USD.
        """
        token_est = self._token_estimator.estimate(tool_id, context_features)
        return self._compute_cost(token_est)

    def observe(
        self, input_tokens: int, output_tokens: int, actual_duration_ms: float
    ) -> None:
        """Record observed API call latency. Delegates to APILatencyModel.

        Args:
            input_tokens: Number of input tokens in the API call.
            output_tokens: Number of output tokens produced.
            actual_duration_ms: Observed wall-clock duration in milliseconds.
        """
        self._latency_model.observe(input_tokens, output_tokens, actual_duration_ms)

    def observe_tool(
        self,
        tool_id: str,
        context_features: ContextFeatures,
        actual_output_tokens: int,
    ) -> None:
        """Record observed tool output tokens. Delegates to TokenEstimator.

        Args:
            tool_id: Identifier of the tool observed.
            context_features: Conversation context at time of observation.
            actual_output_tokens: Total output tokens across all calls.
        """
        self._token_estimator.observe_tool(
            tool_id, context_features, actual_output_tokens
        )

    def _decompose_calls(
        self,
        tool_id: str,
        context_features: ContextFeatures,
        token_est: TokenEstimate,
    ) -> tuple[SubCallPrediction, ...]:
        """Break a tool invocation into per-sub-call predictions."""
        output_per_call = token_est.output_tokens // max(token_est.num_calls, 1)
        conv = context_features.conversation_length_tokens

        if tool_id == "research_gap_scanner":
            gaps = token_est.num_calls - 2
            calls = [
                ("gap_identification", 500 + conv),
                *((f"search_{i + 1}", 2000) for i in range(gaps)),
                ("synthesis", 500 + conv + gaps * 2000),
            ]
            return tuple(
                SubCallPrediction(
                    label=label,
                    input_tokens=inp,
                    output_tokens=output_per_call,
                    latency=self._latency_model.predict(inp, output_per_call),
                )
                for label, inp in calls
            )

        return (
            SubCallPrediction(
                label=tool_id,
                input_tokens=token_est.input_tokens,
                output_tokens=token_est.output_tokens,
                latency=self._latency_model.predict(
                    token_est.input_tokens, token_est.output_tokens
                ),
            ),
        )

    @staticmethod
    def _aggregate_latencies(
        sub_calls: tuple[SubCallPrediction, ...],
    ) -> LatencyEstimate:
        """Sum sub-call latencies assuming sequential independent calls."""
        total_mean = sum(sc.latency.mean_ms for sc in sub_calls)

        total_var = 0.0
        for sc in sub_calls:
            spread = sc.latency.p75_ms - sc.latency.p25_ms
            sigma = spread / (2 * _Z_25) if spread > 0 else 0.0
            total_var += sigma * sigma

        total_sigma = math.sqrt(total_var)
        return LatencyEstimate(
            mean_ms=total_mean,
            p25_ms=max(total_mean - _Z_25 * total_sigma, 0.0),
            p75_ms=total_mean + _Z_25 * total_sigma,
        )

    @staticmethod
    def _compute_cost(token_est: TokenEstimate) -> CostEstimate:
        """Compute cost from token estimates using Gemini pricing."""
        input_cost = token_est.input_tokens * _INPUT_COST_PER_TOKEN
        output_cost = token_est.output_tokens * _OUTPUT_COST_PER_TOKEN
        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total=input_cost + output_cost,
        )
