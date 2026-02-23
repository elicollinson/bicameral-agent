"""Episode-level semantic validation.

Pydantic handles structural validation (types, required fields, non-negative
constraints). This module handles cross-list semantic rules such as temporal
ordering of events within an episode.

The validator returns a structured ``ValidationResult`` with errors and
warnings rather than raising immediately, which is better for batch processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bicameral_agent.schema import Episode


@dataclass
class ValidationResult:
    """Result of validating an Episode.

    Attributes:
        is_valid: True if no errors were found.
        errors: List of error messages describing constraint violations.
        warnings: List of warning messages for suspicious but not invalid data.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Record a validation error and mark the result as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Record a validation warning (does not affect is_valid)."""
        self.warnings.append(msg)


class EpisodeValidator:
    """Validates semantic constraints on a fully-constructed Episode.

    Checks performed:
    - Messages are in non-decreasing timestamp order
    - User events are in non-decreasing timestamp order
    - Context injections are in non-decreasing timestamp order
    - Tool invocations are in non-decreasing invoked_at_ms order
    - Outcome consistency (warns if total_turns=0 but messages exist)
    """

    def validate(self, episode: Episode) -> ValidationResult:
        """Validate an episode and return all errors/warnings found.

        Args:
            episode: The episode to validate.

        Returns:
            A ValidationResult with all errors and warnings collected.
        """
        result = ValidationResult()
        self._check_message_ordering(episode, result)
        self._check_user_event_ordering(episode, result)
        self._check_context_injection_ordering(episode, result)
        self._check_tool_invocation_ordering(episode, result)
        self._check_outcome_consistency(episode, result)
        return result

    def _check_message_ordering(self, episode: Episode, result: ValidationResult) -> None:
        timestamps = [m.timestamp_ms for m in episode.messages]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                result.add_error(
                    f"Messages out of order: messages[{i}].timestamp_ms={timestamps[i]} "
                    f"< messages[{i - 1}].timestamp_ms={timestamps[i - 1]}"
                )

    def _check_user_event_ordering(self, episode: Episode, result: ValidationResult) -> None:
        timestamps = [e.timestamp_ms for e in episode.user_events]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                result.add_error(
                    f"User events out of order: user_events[{i}].timestamp_ms={timestamps[i]} "
                    f"< user_events[{i - 1}].timestamp_ms={timestamps[i - 1]}"
                )

    def _check_context_injection_ordering(
        self, episode: Episode, result: ValidationResult
    ) -> None:
        timestamps = [c.timestamp_ms for c in episode.context_injections]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                result.add_error(
                    f"Context injections out of order: "
                    f"context_injections[{i}].timestamp_ms={timestamps[i]} "
                    f"< context_injections[{i - 1}].timestamp_ms={timestamps[i - 1]}"
                )

    def _check_tool_invocation_ordering(self, episode: Episode, result: ValidationResult) -> None:
        timestamps = [t.invoked_at_ms for t in episode.tool_invocations]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                result.add_error(
                    f"Tool invocations out of order: "
                    f"tool_invocations[{i}].invoked_at_ms={timestamps[i]} "
                    f"< tool_invocations[{i - 1}].invoked_at_ms={timestamps[i - 1]}"
                )

    def _check_outcome_consistency(self, episode: Episode, result: ValidationResult) -> None:
        msg_count = len(episode.messages)
        if episode.outcome.total_turns == 0 and msg_count > 0:
            result.add_warning(
                f"outcome.total_turns is 0 but episode has {msg_count} messages"
            )
