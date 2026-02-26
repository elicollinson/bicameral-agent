"""Per-tool token estimator with online learning (Layer 1 of latency model).

Predicts token counts and API call counts for each tool given a conversation
context. Uses per-tool profiles with EMA-based output token tracking.
Composes with APILatencyModel (Layer 2) which converts token estimates
to wall-clock time predictions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


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
    fixed_calls: int | None  # None = variable (computed from context)


_TOOL_PROFILES: dict[str, _ToolProfile] = {
    "research_gap_scanner": _ToolProfile(
        system_prompt_tokens=500,
        default_output_per_call=400,
        fixed_calls=None,
    ),
    "assumption_auditor": _ToolProfile(
        system_prompt_tokens=400,
        default_output_per_call=450,
        fixed_calls=1,
    ),
    "context_refresher": _ToolProfile(
        system_prompt_tokens=300,
        default_output_per_call=100,
        fixed_calls=1,
    ),
}

# EMA smoothing factor
_EMA_ALPHA = 0.3


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

        conv = context_features.conversation_length_tokens
        turns = context_features.conversation_turn_count

        num_calls = self._compute_num_calls(profile, conv)
        input_tokens = self._compute_input_tokens(tool_id, conv, turns, num_calls)

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

        conv = context_features.conversation_length_tokens
        num_calls = self._compute_num_calls(profile, conv)
        per_call = actual_output_tokens / max(num_calls, 1)

        with self._lock:
            obs = self._observations.get(tool_id)
            if obs is None:
                # First observation replaces default entirely
                self._observations[tool_id] = (per_call, 1)
            else:
                old_mean, count = obs
                new_mean = (1 - _EMA_ALPHA) * old_mean + _EMA_ALPHA * per_call
                self._observations[tool_id] = (new_mean, count + 1)

    @staticmethod
    def _compute_num_calls(profile: _ToolProfile, conv: int) -> int:
        if profile.fixed_calls is not None:
            return profile.fixed_calls
        gaps = min(max(1, conv // 2000), 5)
        return 2 + gaps

    @staticmethod
    def _compute_input_tokens(
        tool_id: str, conv: int, turns: int, num_calls: int
    ) -> int:
        if tool_id == "research_gap_scanner":
            gaps = num_calls - 2
            return (500 + conv) + (gaps * 2000) + (500 + conv + gaps * 2000)
        if tool_id == "assumption_auditor":
            return 400 + conv
        if tool_id == "context_refresher":
            avg_msg = conv / max(turns, 1)
            bounded = min(4 * avg_msg, conv)
            return 300 + int(bounded)
        return conv
