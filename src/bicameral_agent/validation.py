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

        self._check_ordering(
            result,
            timestamps=[m.timestamp_ms for m in episode.messages],
            label="Messages",
            field_path="messages",
            field_name="timestamp_ms",
        )
        self._check_ordering(
            result,
            timestamps=[e.timestamp_ms for e in episode.user_events],
            label="User events",
            field_path="user_events",
            field_name="timestamp_ms",
        )
        self._check_ordering(
            result,
            timestamps=[c.timestamp_ms for c in episode.context_injections],
            label="Context injections",
            field_path="context_injections",
            field_name="timestamp_ms",
        )
        self._check_ordering(
            result,
            timestamps=[t.invoked_at_ms for t in episode.tool_invocations],
            label="Tool invocations",
            field_path="tool_invocations",
            field_name="invoked_at_ms",
        )

        msg_count = len(episode.messages)
        if episode.outcome.total_turns == 0 and msg_count > 0:
            result.add_warning(
                f"outcome.total_turns is 0 but episode has {msg_count} messages"
            )

        return result

    @staticmethod
    def _check_ordering(
        result: ValidationResult,
        timestamps: list[int],
        label: str,
        field_path: str,
        field_name: str,
    ) -> None:
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                result.add_error(
                    f"{label} out of order: {field_path}[{i}].{field_name}={timestamps[i]} "
                    f"< {field_path}[{i - 1}].{field_name}={timestamps[i - 1]}"
                )
