"""Tests for the episode-to-training-data pipeline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from bicameral_agent.heuristic_controller import TOOL_IDS, Action
from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    EpisodeOutcome,
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)
from bicameral_agent.training_pipeline import (
    DEFAULT_MAX_TURNS,
    DISCOUNT_GAMMA,
    STATE_DIM,
    TrainingDataPipeline,
    _ACTION_INDEX,
    _ACTION_ORDER,
    R_DRAIN_CONSUMED,
    R_REDIRECT_CORRECTION,
    R_STOP,
    R_TOKEN_PER_100,
)


# ---------------------------------------------------------------------------
# Episode builders
# ---------------------------------------------------------------------------


def _build_episode(
    *,
    num_turns: int,
    quality: float | None = 0.5,
    user_msg_tokens: int = 10,
    assistant_msg_tokens: int = 20,
    user_events: list[tuple[UserEventType, int]] | None = None,
    tool_invocations_per_turn: dict[int, str] | None = None,
    tool_invoked_offset_ms: int = 100,
    context_injections: list[ContextInjection] | None = None,
    user_messages: list[str] | None = None,
    metadata: dict | None = None,
) -> Episode:
    """Build a synthetic Episode with ``num_turns`` user/assistant pairs.

    Parameters
    ----------
    user_events:
        List of ``(event_type, turn_number)`` pairs. The event timestamp
        is placed just after the assistant message of the given turn.
    tool_invocations_per_turn:
        Map from turn_number to tool_id. The tool is invoked at
        ``user_ts + tool_invoked_offset_ms`` of that turn.
    tool_invoked_offset_ms:
        Offset of the invocation from the turn's user message. The default
        (100) places it before the assistant message (pre-#50 logging
        order); values above 500 place it after (current runner order).
    context_injections:
        Already-built ContextInjection records to attach.
    user_messages:
        Optional override for user message text per turn.
    """
    messages: list[Message] = []
    tools: list[ToolInvocation] = []
    base_ts = 1_000_000

    for turn in range(1, num_turns + 1):
        user_ts = base_ts + (turn - 1) * 1000
        assistant_ts = user_ts + 500
        text = (
            user_messages[turn - 1]
            if user_messages and turn - 1 < len(user_messages)
            else f"user msg {turn}"
        )
        messages.append(
            Message(
                role="user",
                content=text,
                timestamp_ms=user_ts,
                token_count=user_msg_tokens,
            )
        )

        if tool_invocations_per_turn and turn in tool_invocations_per_turn:
            tool_id = tool_invocations_per_turn[turn]
            tools.append(
                ToolInvocation(
                    tool_id=tool_id,
                    invoked_at_ms=user_ts + tool_invoked_offset_ms,
                    completed_at_ms=user_ts + tool_invoked_offset_ms + 200,
                    input_tokens=50,
                    output_tokens=80,
                    result_deposited=False,
                )
            )

        messages.append(
            Message(
                role="assistant",
                content=f"assistant msg {turn}",
                timestamp_ms=assistant_ts,
                token_count=assistant_msg_tokens,
            )
        )

    schema_user_events: list[UserEvent] = []
    if user_events:
        for evt_type, turn in user_events:
            ts = base_ts + (turn - 1) * 1000 + 600  # after that turn's assistant msg
            schema_user_events.append(
                UserEvent(event_type=evt_type, timestamp_ms=ts, metadata={})
            )

    return Episode(
        messages=messages,
        user_events=schema_user_events,
        context_injections=context_injections or [],
        tool_invocations=tools,
        outcome=EpisodeOutcome(
            quality_score=quality,
            total_tokens=sum(m.token_count for m in messages),
            total_turns=num_turns,
            wall_clock_ms=num_turns * 1000,
        ),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline() -> TrainingDataPipeline:
    return TrainingDataPipeline()


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: decision-point count
# ---------------------------------------------------------------------------


def test_20_turn_episode_produces_20_examples(pipeline: TrainingDataPipeline) -> None:
    episode = _build_episode(num_turns=20)
    examples = pipeline.process_episode(episode)
    assert len(examples) == 20
    # Last example is terminal
    assert examples[-1].done is True
    assert all(not e.done for e in examples[:-1])
    # decision_index is monotone
    assert [e.decision_index for e in examples] == list(range(20))


def test_empty_episode_yields_no_examples(pipeline: TrainingDataPipeline) -> None:
    episode = Episode(
        messages=[],
        outcome=EpisodeOutcome(
            quality_score=None, total_tokens=0, total_turns=0, wall_clock_ms=0
        ),
    )
    assert pipeline.process_episode(episode) == []


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: state vector dimensionality
# ---------------------------------------------------------------------------


def test_state_dim_is_documented_and_under_256() -> None:
    assert STATE_DIM == 108
    assert STATE_DIM < 256


def test_state_vectors_are_correct_shape_and_dtype(
    pipeline: TrainingDataPipeline,
) -> None:
    episode = _build_episode(num_turns=5)
    examples = pipeline.process_episode(episode)
    for ex in examples:
        assert ex.state.shape == (STATE_DIM,)
        assert ex.state.dtype == np.float32
        assert ex.next_state.shape == (STATE_DIM,)
        assert ex.next_state.dtype == np.float32


def test_state_values_are_finite(pipeline: TrainingDataPipeline) -> None:
    episode = _build_episode(num_turns=10)
    examples = pipeline.process_episode(episode)
    for ex in examples:
        assert np.all(np.isfinite(ex.state))
        assert np.all(np.isfinite(ex.next_state))


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: reward construction by hand
# ---------------------------------------------------------------------------


def test_reward_components_by_hand(pipeline: TrainingDataPipeline) -> None:
    """Verify per-step rewards and discounted return on a small episode.

    A FOLLOW_UP event after assistant N is logged when the user has chosen
    to send the next user message (turn N+1) as a follow-up. So the
    follow-up content being classified is the *next* user message.

    Episode layout (3 turns):
      Turn 1: assistant gets STOP feedback at end of turn 1
      Turn 2: assistant gets FOLLOW_UP at end of turn 2 — next user
              message is "wrong, that's not right" (CORRECTION)
      Turn 3: terminal turn (no follow-up)

    Quality score = 0.8 (added to terminal step's reward).
    """
    episode = _build_episode(
        num_turns=3,
        quality=0.8,
        assistant_msg_tokens=100,  # exactly 100 tok → -0.01 token cost
        user_msg_tokens=0,
        user_messages=[
            "initial question",
            "elaborate please",
            "wrong, that's not right",  # next msg after turn 2's FOLLOW_UP
        ],
        user_events=[
            (UserEventType.STOP, 1),
            (UserEventType.FOLLOW_UP, 2),
        ],
    )
    examples = pipeline.process_episode(episode)
    assert len(examples) == 3

    # Step 1: STOP (-0.3) + token cost (-0.01)
    assert examples[0].reward == pytest.approx(R_STOP + R_TOKEN_PER_100)
    # Step 2: FOLLOW_UP classified as CORRECTION (-0.2) + token cost (-0.01)
    assert examples[1].reward == pytest.approx(
        R_REDIRECT_CORRECTION + R_TOKEN_PER_100
    )
    # Step 3 (terminal): no follow-up + token cost + quality (0.8)
    expected_step3 = R_TOKEN_PER_100 + 0.8
    assert examples[2].reward == pytest.approx(expected_step3)

    # Discounted returns (γ = 0.95)
    g = DISCOUNT_GAMMA
    r0, r1, r2 = examples[0].reward, examples[1].reward, examples[2].reward
    expected_g2 = r2
    expected_g1 = r1 + g * expected_g2
    expected_g0 = r0 + g * expected_g1
    assert examples[2].discounted_return == pytest.approx(expected_g2)
    assert examples[1].discounted_return == pytest.approx(expected_g1)
    assert examples[0].discounted_return == pytest.approx(expected_g0)


def test_drain_consumed_reward(pipeline: TrainingDataPipeline) -> None:
    """A consumed injection at turn N adds +0.05 to that step's reward."""
    base_ts = 1_000_000
    inj = ContextInjection(
        content="ctx",
        source_tool_id="research_gap_scanner",
        priority=1,
        timestamp_ms=base_ts + 200,
        token_count=50,
        consumed=True,
        consumed_at_turn=2,
    )
    episode = _build_episode(
        num_turns=3,
        quality=0.0,
        assistant_msg_tokens=0,
        user_msg_tokens=0,
        context_injections=[inj],
    )
    examples = pipeline.process_episode(episode)
    # Step 1 has the injection (timestamp falls in turn 1's window) but
    # consumed_at_turn=2, so credit goes to step 2.
    assert examples[0].reward == pytest.approx(0.0)
    assert examples[1].reward == pytest.approx(R_DRAIN_CONSUMED)


def test_terminal_quality_only_on_last_step(pipeline: TrainingDataPipeline) -> None:
    """Terminal quality reward applies only to the final decision point."""
    episode = _build_episode(
        num_turns=2,
        quality=1.0,
        assistant_msg_tokens=0,
        user_msg_tokens=0,
    )
    examples = pipeline.process_episode(episode)
    assert examples[0].reward == pytest.approx(0.0)
    assert examples[1].reward == pytest.approx(1.0)


def test_quality_none_treated_as_zero(pipeline: TrainingDataPipeline) -> None:
    episode = _build_episode(
        num_turns=2, quality=None, assistant_msg_tokens=0, user_msg_tokens=0
    )
    examples = pipeline.process_episode(episode)
    assert examples[-1].reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Acceptance Criterion 4: shaped reward correlates with quality (r > 0.5)
# ---------------------------------------------------------------------------


def test_shaped_reward_correlates_with_quality(
    pipeline: TrainingDataPipeline,
) -> None:
    """Across 50 episodes, mean discounted return should correlate with
    final quality score.

    Construction: high-quality episodes have more positive intermediate
    signals (encouragement follow-ups), low-quality ones have more
    negative signals (stops, corrections).
    """
    rng = np.random.default_rng(123)
    qualities: list[float] = []
    returns: list[float] = []

    for _ in range(50):
        # Random quality
        q = float(rng.uniform(0.0, 1.0))
        # More follow-ups (encouragement) for high quality, more stops for low
        events: list[tuple[UserEventType, int]] = []
        msgs: list[str] = ["initial question"]
        for turn in range(1, 6):
            if rng.uniform() < q:
                events.append((UserEventType.FOLLOW_UP, turn))
                msgs.append("thanks great perfect" if turn < 5 else "")
            else:
                events.append((UserEventType.STOP, turn))
                msgs.append("wrong, that's not right" if turn < 5 else "")
        episode = _build_episode(
            num_turns=5,
            quality=q,
            user_messages=msgs,
            user_events=events,
            assistant_msg_tokens=10,
            user_msg_tokens=5,
        )
        examples = pipeline.process_episode(episode)
        qualities.append(q)
        returns.append(float(np.mean([e.discounted_return for e in examples])))

    # Pearson correlation
    qs = np.asarray(qualities)
    rs = np.asarray(returns)
    r = float(np.corrcoef(qs, rs)[0, 1])
    assert r > 0.5, f"Expected correlation > 0.5, got {r:.3f}"


# ---------------------------------------------------------------------------
# Acceptance Criterion 5: no label leakage
# ---------------------------------------------------------------------------


def test_no_label_leakage_states_are_monotone(
    pipeline: TrainingDataPipeline,
) -> None:
    """The state at decision t reflects only data up to that point.

    We verify by mutating events after a target turn and checking the
    state vector at that target turn is unchanged.
    """
    base_episode = _build_episode(
        num_turns=5,
        user_events=[(UserEventType.FOLLOW_UP, 2)],
    )
    base_examples = pipeline.process_episode(base_episode)

    # Add a STOP event AFTER turn 2 — should not affect state at turn 1.
    perturbed = _build_episode(
        num_turns=5,
        user_events=[
            (UserEventType.FOLLOW_UP, 2),
            (UserEventType.STOP, 4),
            (UserEventType.STOP, 5),
        ],
    )
    perturbed_examples = pipeline.process_episode(perturbed)

    # State at decision points 0 and 1 must be identical (events at turns
    # 4-5 shouldn't leak into earlier states).
    np.testing.assert_array_equal(
        base_examples[0].state, perturbed_examples[0].state
    )
    np.testing.assert_array_equal(
        base_examples[1].state, perturbed_examples[1].state
    )


def test_state_vector_reflects_only_past_messages(
    pipeline: TrainingDataPipeline,
) -> None:
    """Construct two episodes that differ only in messages AFTER turn 2.
    State at decision point 1 (turn 2) must match between them.
    """
    ep_a = _build_episode(num_turns=5, user_messages=[f"q{i}" for i in range(5)])
    # Same first 2 turns, different later turns
    ep_b = _build_episode(
        num_turns=5,
        user_messages=["q0", "q1", "DIFFERENT_2", "DIFFERENT_3", "DIFFERENT_4"],
    )
    ex_a = pipeline.process_episode(ep_a)
    ex_b = pipeline.process_episode(ep_b)
    # First decision point: both have only user msg 0 in state. Identical.
    np.testing.assert_array_equal(ex_a[0].state, ex_b[0].state)
    # Second decision point: state has user msgs 0,1 and assistant msg 1.
    # Identical despite differing later messages.
    np.testing.assert_array_equal(ex_a[1].state, ex_b[1].state)


def test_completion_fraction_uses_configured_max_turns(
    pipeline: TrainingDataPipeline,
) -> None:
    """Dims 63/107 are turn / configured max_turns, not turn / actual length."""
    episode = _build_episode(num_turns=4)
    examples = pipeline.process_episode(episode)
    for i, ex in enumerate(examples):
        turn = i + 1
        expected = turn / DEFAULT_MAX_TURNS
        assert ex.state[63] == pytest.approx(expected)
        assert ex.state[107] == pytest.approx(expected)

    # A custom configured limit changes the fraction accordingly.
    custom = TrainingDataPipeline(max_turns=10)
    examples = custom.process_episode(episode)
    assert examples[1].state[63] == pytest.approx(2 / 10)
    assert examples[1].state[107] == pytest.approx(2 / 10)


def test_no_leakage_from_episode_length(pipeline: TrainingDataPipeline) -> None:
    """Turn-k states of a short and a long episode with identical prefixes
    must be identical: the episode's final turn count is future information
    and must not appear in any feature (e.g. completion fraction).
    """
    short = _build_episode(num_turns=4)
    long = _build_episode(num_turns=8)
    ex_short = pipeline.process_episode(short)
    ex_long = pipeline.process_episode(long)

    for k in range(len(ex_short)):
        np.testing.assert_array_equal(ex_short[k].state, ex_long[k].state)


def test_pipeline_rejects_invalid_max_turns() -> None:
    with pytest.raises(ValueError):
        TrainingDataPipeline(max_turns=0)


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


def test_action_order_matches_policy_value_net() -> None:
    """_ACTION_ORDER must stay identical to policy_value_net.ACTION_ORDER,
    otherwise action indices in training data silently mismatch the policy
    head."""
    pytest.importorskip("torch")
    from bicameral_agent.policy_value_net import ACTION_ORDER

    assert _ACTION_ORDER == ACTION_ORDER


def test_default_max_turns_matches_episode_config() -> None:
    """DEFAULT_MAX_TURNS must stay identical to EpisodeConfig.max_turns,
    otherwise a config-default change silently mis-scales the
    completion-fraction features (dims 63/107)."""
    from bicameral_agent.episode_runner import EpisodeConfig

    assert DEFAULT_MAX_TURNS == EpisodeConfig().max_turns


def test_queue_slices_identical_between_encoder_and_pipeline(
    pipeline: TrainingDataPipeline,
) -> None:
    """Both queue slices of the state vector use the shared encoding."""
    base_ts = 1_000_000
    inj = ContextInjection(
        content="ctx",
        source_tool_id="research_gap_scanner",
        priority=0,  # Priority.LOW — must be distinguishable from empty
        timestamp_ms=base_ts + 100,
        token_count=50,
        consumed=False,
    )
    episode = _build_episode(num_turns=2, context_injections=[inj])
    examples = pipeline.process_episode(episode)
    for ex in examples:
        np.testing.assert_array_equal(ex.state[53:59], ex.state[97:103])
    # The pending LOW-priority injection is visible from turn 1 onward.
    assert examples[0].state[55] == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Tool-recency features (seconds cap)
# ---------------------------------------------------------------------------


def test_tool_recency_uses_seconds_cap() -> None:
    """Elapsed time since a tool invocation must not saturate at 20s."""
    inv = ToolInvocation(
        tool_id=TOOL_IDS[Action.SCANNER],
        invoked_at_ms=0,
        completed_at_ms=0,
        input_tokens=1,
        output_tokens=1,
    )
    # 30s ago: with the old turns-based cap (20) this saturated to 1.0.
    out = TrainingDataPipeline._encode_tool_history_times([inv], cutoff_ms=30_000)
    assert out[0] == pytest.approx(30.0 / 300.0)
    assert out[0] < 1.0
    # Empty slots stay at 1.0 ("long ago / never").
    assert out[1] == 1.0
    assert out[2] == 1.0


# ---------------------------------------------------------------------------
# Acceptance Criterion 6: throughput (100 episodes in < 60s)
# ---------------------------------------------------------------------------


def test_processing_throughput(pipeline: TrainingDataPipeline) -> None:
    episodes = [_build_episode(num_turns=10) for _ in range(100)]
    t0 = time.perf_counter()
    examples = pipeline.process_episodes(episodes)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"Took {elapsed:.1f}s; budget is 60s"
    # 10 turns each
    assert len(examples) == 100 * 10


# ---------------------------------------------------------------------------
# Acceptance Criterion 7: PyTorch DataLoader compatibility
# ---------------------------------------------------------------------------


def test_to_torch_dataset_compatibility(pipeline: TrainingDataPipeline) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    episode = _build_episode(num_turns=8)
    examples = pipeline.process_episode(episode)
    dataset = TrainingDataPipeline.to_torch_dataset(examples)

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2  # 8 / 4

    states, actions, rewards, next_states, dones, returns = batches[0]
    assert states.shape == (4, STATE_DIM)
    assert next_states.shape == (4, STATE_DIM)
    assert states.dtype == torch.float32
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.bool
    assert returns.dtype == torch.float32


def test_to_torch_dataset_empty(pipeline: TrainingDataPipeline) -> None:
    pytest.importorskip("torch")
    dataset = TrainingDataPipeline.to_torch_dataset([])
    assert len(dataset) == 0


# ---------------------------------------------------------------------------
# Action inference
# ---------------------------------------------------------------------------


def test_action_inference_from_tool_invocations(
    pipeline: TrainingDataPipeline,
) -> None:
    """Per-turn action is inferred from tool invocations within the turn."""
    episode = _build_episode(
        num_turns=4,
        tool_invocations_per_turn={
            1: TOOL_IDS[Action.SCANNER],
            3: TOOL_IDS[Action.AUDITOR],
        },
    )
    examples = pipeline.process_episode(episode)
    assert examples[0].action == _ACTION_INDEX[Action.SCANNER]
    assert examples[1].action == _ACTION_INDEX[Action.DO_NOTHING]
    assert examples[2].action == _ACTION_INDEX[Action.AUDITOR]
    assert examples[3].action == _ACTION_INDEX[Action.DO_NOTHING]


def test_all_action_indices_in_valid_range(
    pipeline: TrainingDataPipeline,
) -> None:
    episode = _build_episode(
        num_turns=5,
        tool_invocations_per_turn={
            1: TOOL_IDS[Action.SCANNER],
            2: TOOL_IDS[Action.AUDITOR],
            3: TOOL_IDS[Action.REFRESHER],
        },
    )
    examples = pipeline.process_episode(episode)
    for ex in examples:
        assert 0 <= ex.action <= 3


def test_action_inference_with_live_runner_ordering(
    pipeline: TrainingDataPipeline,
) -> None:
    """Regression (#80): tools logged after the assistant message are attributed.

    Since #50 the runner logs the assistant message at generation time,
    before tool events, so invocations fall after assistant_ts. The old
    (user_ts, assistant_ts] window mislabeled these turns DO_NOTHING.
    """
    episode = _build_episode(
        num_turns=3,
        tool_invocations_per_turn={
            1: TOOL_IDS[Action.SCANNER],
            3: TOOL_IDS[Action.REFRESHER],
        },
        tool_invoked_offset_ms=600,  # after the assistant message (+500)
    )
    examples = pipeline.process_episode(episode)
    assert examples[0].action == _ACTION_INDEX[Action.SCANNER]
    assert examples[1].action == _ACTION_INDEX[Action.DO_NOTHING]
    assert examples[2].action == _ACTION_INDEX[Action.REFRESHER]


def test_action_inference_prefers_turn_linkage_over_timestamps(
    pipeline: TrainingDataPipeline,
) -> None:
    """An invocation carrying ``turn`` is attributed to it regardless of timestamps."""
    episode = _build_episode(num_turns=3)
    inv = ToolInvocation(
        tool_id=TOOL_IDS[Action.AUDITOR],
        # Timestamps fall inside turn 3's window; the turn field must win.
        invoked_at_ms=episode.messages[-1].timestamp_ms + 100,
        completed_at_ms=episode.messages[-1].timestamp_ms + 200,
        input_tokens=50,
        output_tokens=80,
        turn=2,
    )
    episode = episode.model_copy(update={"tool_invocations": [inv]})
    examples = pipeline.process_episode(episode)
    assert examples[0].action == _ACTION_INDEX[Action.DO_NOTHING]
    assert examples[1].action == _ACTION_INDEX[Action.AUDITOR]
    assert examples[2].action == _ACTION_INDEX[Action.DO_NOTHING]


# ---------------------------------------------------------------------------
# next_state correctness
# ---------------------------------------------------------------------------


def test_next_state_matches_subsequent_state(
    pipeline: TrainingDataPipeline,
) -> None:
    episode = _build_episode(num_turns=4)
    examples = pipeline.process_episode(episode)
    for i in range(len(examples) - 1):
        np.testing.assert_array_equal(examples[i].next_state, examples[i + 1].state)
    # Final next_state is zeros (terminal)
    np.testing.assert_array_equal(
        examples[-1].next_state, np.zeros(STATE_DIM, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Episode provenance
# ---------------------------------------------------------------------------


def test_examples_carry_episode_id(pipeline: TrainingDataPipeline) -> None:
    episode = _build_episode(num_turns=3)
    examples = pipeline.process_episode(episode)
    assert all(e.episode_id == episode.episode_id for e in examples)
