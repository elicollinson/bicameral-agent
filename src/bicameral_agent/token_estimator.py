"""Per-tool token estimator with online learning (Layer 1 of latency model).

Predicts token counts and API call counts for each tool given a conversation
context. Uses per-tool profiles with EMA-based output token tracking.
Composes with APILatencyModel (Layer 2) which converts token estimates
to wall-clock time predictions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from bicameral_agent.heuristic_controller import TOOL_IDS, Action


@dataclass(frozen=True, slots=True)
class ContextFeatures:
    """Features extracted from the current conversation state."""

    conversation_length_tokens: int
    conversation_turn_count: int


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Predicted token usage and call count for a tool invocation."""

    input_tokens: int
    output_tokens: int
    num_calls: int


@dataclass(frozen=True, slots=True)
class _ToolProfile:
    """Per-tool constants for token estimation."""

    system_prompt_tokens: int
    default_output_per_call: int
    num_calls: int


# Call counts mirror the actual tool implementations: the scanner makes
# exactly two LLM calls (gap identification + result ranking; its searches
# run against a local provider), the auditor makes two (assumption
# extraction + evidence assessment), and the refresher makes one.
# Default output-per-call values are anchored to the median recorded
# output token counts from the #46 baseline re-run on the Ollama cloud
# backend (#44): scanner 1566/2, auditor 1856/2, refresher 1110/1.
_TOOL_PROFILES: dict[str, _ToolProfile] = {
    TOOL_IDS[Action.SCANNER]: _ToolProfile(
        system_prompt_tokens=500,
        default_output_per_call=780,
        num_calls=2,
    ),
    TOOL_IDS[Action.AUDITOR]: _ToolProfile(
        system_prompt_tokens=400,
        default_output_per_call=930,
        num_calls=2,
    ),
    TOOL_IDS[Action.REFRESHER]: _ToolProfile(
        system_prompt_tokens=300,
        default_output_per_call=1100,
        num_calls=1,
    ),
}

# Estimated input size of the scanner's second call (identified gaps plus
# local search results) and the auditor's second call (evidence block).
SCANNER_RANKING_INPUT_TOKENS = 1500
AUDITOR_ASSESSMENT_INPUT_TOKENS = 1200

# EMA smoothing factor
_EMA_ALPHA = 0.3


def sub_call_inputs(
    tool_id: str, context_features: ContextFeatures
) -> list[tuple[str, int]]:
    """Per-sub-call ``(label, input_tokens)`` breakdown for a tool invocation.

    Single source of truth for the sub-call input structure: TokenEstimator
    sums it for the aggregate input estimate, and ToolLatencyModel iterates
    it for per-sub-call latency predictions, so the two cannot drift apart.
    """
    conv = context_features.conversation_length_tokens
    if tool_id == TOOL_IDS[Action.SCANNER]:
        return [
            ("gap_identification", 500 + conv),
            ("result_ranking", SCANNER_RANKING_INPUT_TOKENS),
        ]
    if tool_id == TOOL_IDS[Action.AUDITOR]:
        return [
            ("assumption_extraction", 400 + conv),
            ("evidence_assessment", AUDITOR_ASSESSMENT_INPUT_TOKENS),
        ]
    if tool_id == TOOL_IDS[Action.REFRESHER]:
        avg_msg = conv / max(context_features.conversation_turn_count, 1)
        bounded = min(4 * avg_msg, conv)
        return [(tool_id, 300 + int(bounded))]
    return [(tool_id, conv)]


class TokenEstimator:
    """Per-tool token estimator with online learning.

    Predicts input tokens, output tokens, and number of API calls
    for each tool given conversation context features. Output token
    predictions are updated via exponential moving average from
    observed values.

    Thread-safe: all mutable state is protected by a single lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # tool_id -> (ema_mean_output_per_call, observation_count)
        self._observations: dict[str, tuple[float, int]] = {}

    def estimate(self, tool_id: str, context_features: ContextFeatures) -> TokenEstimate:
        """Estimate token usage for a tool invocation.

        Args:
            tool_id: Identifier of the tool to estimate for.
            context_features: Current conversation context.

        Returns:
            TokenEstimate with predicted input/output tokens and call count.

        Raises:
            ValueError: If tool_id is not in the registry.
        """
        profile = _TOOL_PROFILES.get(tool_id)
        if profile is None:
            raise ValueError(f"Unknown tool: {tool_id!r}")

        num_calls = profile.num_calls
        input_tokens = sum(
            inp for _, inp in sub_call_inputs(tool_id, context_features)
        )

        with self._lock:
            obs = self._observations.get(tool_id)

        output_per_call = obs[0] if obs is not None else float(profile.default_output_per_call)
        output_tokens = int(output_per_call * num_calls)

        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_calls=num_calls,
        )

    def observe_tool(
        self,
        tool_id: str,
        context_features: ContextFeatures,
        actual_output_tokens: int,
    ) -> None:
        """Record observed output tokens for EMA update.

        Args:
            tool_id: Identifier of the tool observed.
            context_features: Conversation context at time of observation.
            actual_output_tokens: Total output tokens across all calls.

        Raises:
            ValueError: If tool_id is not in the registry.
        """
        profile = _TOOL_PROFILES.get(tool_id)
        if profile is None:
            raise ValueError(f"Unknown tool: {tool_id!r}")

        per_call = actual_output_tokens / max(profile.num_calls, 1)

        with self._lock:
            obs = self._observations.get(tool_id)
            if obs is None:
                # First observation replaces default entirely
                self._observations[tool_id] = (per_call, 1)
            else:
                old_mean, count = obs
                new_mean = (1 - _EMA_ALPHA) * old_mean + _EMA_ALPHA * per_call
                self._observations[tool_id] = (new_mean, count + 1)
