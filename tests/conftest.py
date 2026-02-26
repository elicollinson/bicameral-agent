"""Shared test fixtures for episode schema tests."""

import uuid

import numpy as np
import pytest

from bicameral_agent.embeddings import HashEmbedder
from bicameral_agent.encoder import StateEncoder
from bicameral_agent.latency import APILatencyModel
from bicameral_agent.queue import ContextQueue, Priority, QueueItem
from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    EpisodeOutcome,
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)


@pytest.fixture
def make_message():
    """Factory fixture: create a Message with defaults."""

    def _make(role="user", content="hello", timestamp_ms=1000, token_count=5):
        return Message(
            role=role,
            content=content,
            timestamp_ms=timestamp_ms,
            token_count=token_count,
        )

    return _make


@pytest.fixture
def make_episode(make_message):
    """Factory fixture: create a well-formed Episode."""

    def _make(num_messages=3, **overrides):
        messages = [
            make_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
                timestamp_ms=1000 + i * 100,
                token_count=10 + i,
            )
            for i in range(num_messages)
        ]
        defaults = dict(
            episode_id=str(uuid.uuid4()),
            messages=messages,
            user_events=[
                UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=1500),
            ],
            context_injections=[
                ContextInjection(
                    content="ctx",
                    source_tool_id="tool-1",
                    priority=1,
                    timestamp_ms=1050,
                    token_count=20,
                ),
            ],
            tool_invocations=[
                ToolInvocation(
                    tool_id="tool-1",
                    invoked_at_ms=1100,
                    completed_at_ms=1200,
                    input_tokens=15,
                    output_tokens=25,
                ),
            ],
            outcome=EpisodeOutcome(
                total_tokens=100,
                total_turns=num_messages,
                wall_clock_ms=5000,
            ),
            metadata={"source": "test"},
        )
        defaults.update(overrides)
        return Episode(**defaults)

    return _make


@pytest.fixture
def five_episodes(make_episode):
    """Five distinct well-formed episodes for round-trip tests."""
    return [make_episode(num_messages=i + 2) for i in range(5)]


@pytest.fixture
def encoder():
    """StateEncoder with a deterministic HashEmbedder (seed=42)."""
    return StateEncoder(HashEmbedder(seed=42))


@pytest.fixture
def simple_conversation():
    """3 messages, 1 follow-up event, 1 tool invocation."""
    messages = [
        Message(role="user", content="Tell me about Python", timestamp_ms=1000, token_count=5),
        Message(
            role="assistant",
            content="Python is a programming language",
            timestamp_ms=2500,
            token_count=6,
        ),
        Message(role="user", content="Can you explain more?", timestamp_ms=5000, token_count=5),
    ]
    events = [UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=5000)]
    tools = [
        ToolInvocation(
            tool_id="research_gap_scanner",
            invoked_at_ms=1100,
            completed_at_ms=1200,
            input_tokens=10,
            output_tokens=20,
        )
    ]
    return messages, events, tools


@pytest.fixture
def make_queue_item():
    """Factory fixture: create a QueueItem with defaults."""

    def _make(
        content="test context",
        priority=Priority.MEDIUM,
        source_tool_id="tool-1",
        token_count=10,
        expiry_turns=10,
        dedup_key=None,
        enqueued_at_turn=0,
    ):
        return QueueItem(
            content=content,
            priority=priority,
            source_tool_id=source_tool_id,
            token_count=token_count,
            expiry_turns=expiry_turns,
            dedup_key=dedup_key,
            enqueued_at_turn=enqueued_at_turn,
        )

    return _make


@pytest.fixture
def empty_queue():
    """An empty ContextQueue."""
    return ContextQueue()


@pytest.fixture
def trained_latency_model():
    """An APILatencyModel with 50 synthetic observations.

    Synthetic model: duration = 200 + 0.01*inp + 15*out + N(0, 100).
    """
    model = APILatencyModel()
    rng = np.random.default_rng(42)
    for _ in range(50):
        inp = int(rng.integers(100, 5000))
        out = int(rng.integers(50, 2000))
        true_duration = max(200.0 + 0.01 * inp + 15.0 * out + rng.normal(0, 100), 1.0)
        model.observe(inp, out, true_duration)
    return model
