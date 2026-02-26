"""Canonical episode data schema for the bicameral-agent system.

Defines the data contracts used across logging, replay, and training.
All schema classes use Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

import enum
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator


class UserEventType(str, enum.Enum):
    """Types of user-initiated events during an episode.

    - STOP: User explicitly stopped the agent.
    - EDIT: User edited the agent's output before it was finalized.
    - FOLLOW_UP: User sent a follow-up message continuing the conversation.
    """

    STOP = "stop"
    EDIT = "edit"
    FOLLOW_UP = "follow_up"


class Message(BaseModel):
    """A single message in the episode conversation.

    Produced by the conversation logger. Each message records who sent it,
    the content, when it was sent, and how many tokens it consumed.
    """

    role: str
    """The sender role, e.g. 'user', 'assistant', 'system'."""

    content: str
    """The text content of the message."""

    timestamp_ms: int = Field(ge=0)
    """Milliseconds since epoch when the message was created."""

    token_count: int = Field(ge=0)
    """Number of tokens in this message as counted by the tokenizer."""


class UserEvent(BaseModel):
    """A user-initiated event during the episode.

    Produced by the UI/interaction layer. Captures discrete user actions
    such as stopping the agent, editing output, or sending follow-ups.
    """

    event_type: UserEventType
    """The type of user event."""

    timestamp_ms: int = Field(ge=0)
    """Milliseconds since epoch when the event occurred."""

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata associated with this event (must be JSON-serializable)."""


class ContextInjection(BaseModel):
    """Context injected into the episode from an external tool or source.

    Produced by the context injection queue. Tracks what context was available,
    its priority, and whether/when the agent consumed it.
    """

    content: str
    """The injected context text."""

    source_tool_id: str
    """Identifier of the tool that produced this context."""

    priority: int = Field(ge=0)
    """Priority level for consumption ordering (higher = more important)."""

    timestamp_ms: int = Field(ge=0)
    """Milliseconds since epoch when the injection was created."""

    token_count: int = Field(ge=0)
    """Number of tokens in the injected content."""

    consumed: bool = False
    """Whether the agent consumed this injection."""

    consumed_at_turn: int | None = Field(default=None, ge=0)
    """The turn number at which this injection was consumed, if applicable."""


class ToolInvocation(BaseModel):
    """Record of a tool being invoked during the episode.

    Produced by the tool execution layer. Tracks timing, token usage,
    and whether the result was deposited back into the conversation.
    """

    tool_id: str
    """Identifier of the tool that was invoked."""

    invoked_at_ms: int = Field(ge=0)
    """Milliseconds since epoch when the tool was invoked."""

    completed_at_ms: int = Field(ge=0)
    """Milliseconds since epoch when the tool completed."""

    input_tokens: int = Field(ge=0)
    """Number of tokens in the tool's input."""

    output_tokens: int = Field(ge=0)
    """Number of tokens in the tool's output."""

    result_deposited: bool = False
    """Whether the tool's result was deposited back into the conversation."""

    @model_validator(mode="after")
    def check_temporal_order(self) -> ToolInvocation:
        """Ensure completed_at_ms >= invoked_at_ms."""
        if self.completed_at_ms < self.invoked_at_ms:
            raise ValueError(
                f"completed_at_ms ({self.completed_at_ms}) must be >= "
                f"invoked_at_ms ({self.invoked_at_ms})"
            )
        return self


class EpisodeOutcome(BaseModel):
    """Outcome metrics for a completed episode.

    Produced by the episode finalizer. Captures aggregate statistics
    about the episode's resource usage and optional quality assessment.
    """

    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    """Quality score in [0.0, 1.0], or None if not yet scored."""

    total_tokens: int = Field(ge=0)
    """Total tokens consumed across all messages and tool calls."""

    total_turns: int = Field(ge=0)
    """Total number of conversational turns in the episode."""

    wall_clock_ms: int = Field(ge=0)
    """Wall-clock duration of the episode in milliseconds."""


class Episode(BaseModel):
    """The canonical episode data record.

    An episode represents a complete interaction session including messages,
    user events, context injections, tool invocations, and outcome metrics.
    This schema is the single source of truth used by the conversation logger,
    replay engine, and training pipeline.
    """

    model_config = ConfigDict(
        json_schema_extra={"description": "Canonical episode schema for bicameral-agent"}
    )

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this episode (UUID4 string)."""

    messages: list[Message] = Field(default_factory=list)
    """Ordered list of messages in the conversation."""

    user_events: list[UserEvent] = Field(default_factory=list)
    """Ordered list of user-initiated events."""

    context_injections: list[ContextInjection] = Field(default_factory=list)
    """Ordered list of context injections from external tools."""

    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    """Ordered list of tool invocations."""

    outcome: EpisodeOutcome
    """Aggregate outcome metrics for the episode."""

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata: controller type, model version, hyperparameters, etc.
    Values must be JSON-serializable."""

    def to_json(self) -> str:
        """Serialize this episode to a JSON string.

        Returns:
            A JSON string representation of the episode.
        """
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Episode:
        """Deserialize an episode from a JSON string.

        Args:
            json_str: A JSON string produced by to_json().

        Returns:
            A reconstructed Episode instance.
        """
        return cls.model_validate_json(json_str)

    def to_parquet(self, path: str) -> None:
        """Serialize this episode to a Parquet file.

        Args:
            path: File path to write the Parquet file to.
        """
        from bicameral_agent.serialization import episode_to_parquet

        episode_to_parquet(self, path)

    @classmethod
    def from_parquet(cls, path: str) -> Episode:
        """Deserialize an episode from a Parquet file.

        Args:
            path: File path to read the Parquet file from.

        Returns:
            A reconstructed Episode instance.
        """
        from bicameral_agent.serialization import episode_from_parquet

        return episode_from_parquet(path)
