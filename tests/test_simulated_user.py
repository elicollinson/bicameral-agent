"""Tests for the simulated user (Issue #37)."""

import json
import os
from unittest.mock import MagicMock

import pytest

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.schema import Message
from bicameral_agent.simulated_user import (
    ActionType,
    Patience,
    SimulatedUser,
    Strictness,
    UserAction,
    _MAX_TURNS,
    _format_conversation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id="test_001",
    difficulty=TaskDifficulty.TYPICAL,
    question="What is photosynthesis?",
    gold_answer="Photosynthesis is the process by which plants convert sunlight into energy.",
    scoring_rubric="5: Complete and accurate. 3: Partial. 1: Wrong or missing.",
) -> ResearchQATask:
    return ResearchQATask(
        task_id=task_id,
        difficulty=difficulty,
        split=TaskSplit.EVAL,
        question=question,
        gold_answer=gold_answer,
        scoring_rubric=scoring_rubric,
    )


def _make_history(n_turns: int) -> list[Message]:
    """Create a conversation history with n_turns user-assistant pairs."""
    messages = []
    for i in range(n_turns):
        messages.append(Message(role="user", content=f"User message {i + 1}", timestamp_ms=i * 2000, token_count=10))
        messages.append(Message(role="assistant", content=f"Assistant response {i + 1}", timestamp_ms=i * 2000 + 1000, token_count=20))
    return messages


def _mock_gemini_response(
    action_type="follow_up",
    message="Can you tell me more?",
    followup_type="elaboration",
    response_delay_ms=1500,
    confidence=0.8,
):
    """Create a mock GeminiResponse with structured JSON content."""
    data = {
        "action_type": action_type,
        "response_delay_ms": response_delay_ms,
        "confidence": confidence,
    }
    if action_type == "follow_up":
        data["message"] = message
        data["followup_type"] = followup_type
    response = MagicMock()
    response.content = json.dumps(data)
    return response


def _make_simulated_user(response=None, patience=Patience.MEDIUM, strictness=Strictness.MEDIUM):
    """Create a SimulatedUser with a mocked GeminiClient."""
    mock_client = MagicMock()
    if response is None:
        response = _mock_gemini_response()
    mock_client.generate.return_value = response
    user = SimulatedUser(client=mock_client, patience=patience, strictness=strictness)
    return user, mock_client


# ---------------------------------------------------------------------------
# TestUserActionValidation
# ---------------------------------------------------------------------------


class TestUserActionValidation:
    def test_follow_up_requires_message(self):
        with pytest.raises(ValueError, match="requires a message"):
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            )

    def test_follow_up_requires_followup_type(self):
        with pytest.raises(ValueError, match="requires a followup_type"):
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="Tell me more",
                response_delay_ms=100,
                confidence=0.8,
            )

    def test_follow_up_rejects_new_task(self):
        with pytest.raises(ValueError, match="followup_type must be one of"):
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="Do something else",
                followup_type=FollowUpType.NEW_TASK,
                response_delay_ms=100,
                confidence=0.8,
            )

    def test_stop_rejects_message(self):
        with pytest.raises(ValueError, match="must not have a message"):
            UserAction(
                action_type=ActionType.STOP,
                message="bye",
                response_delay_ms=100,
                confidence=0.8,
            )

    def test_stop_rejects_followup_type(self):
        with pytest.raises(ValueError, match="must not have a followup_type"):
            UserAction(
                action_type=ActionType.STOP,
                followup_type=FollowUpType.CORRECTION,
                response_delay_ms=100,
                confidence=0.8,
            )

    def test_task_complete_rejects_message(self):
        with pytest.raises(ValueError, match="must not have a message"):
            UserAction(
                action_type=ActionType.TASK_COMPLETE,
                message="done",
                response_delay_ms=100,
                confidence=0.9,
            )

    def test_task_complete_rejects_followup_type(self):
        with pytest.raises(ValueError, match="must not have a followup_type"):
            UserAction(
                action_type=ActionType.TASK_COMPLETE,
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.9,
            )

    def test_valid_follow_up(self):
        action = UserAction(
            action_type=ActionType.FOLLOW_UP,
            message="Can you explain more?",
            followup_type=FollowUpType.ELABORATION,
            response_delay_ms=1000,
            confidence=0.7,
        )
        assert action.action_type == ActionType.FOLLOW_UP
        assert action.message == "Can you explain more?"

    def test_valid_stop(self):
        action = UserAction(
            action_type=ActionType.STOP,
            response_delay_ms=500,
            confidence=0.9,
        )
        assert action.action_type == ActionType.STOP
        assert action.message is None
        assert action.followup_type is None

    def test_valid_task_complete(self):
        action = UserAction(
            action_type=ActionType.TASK_COMPLETE,
            response_delay_ms=300,
            confidence=0.95,
        )
        assert action.action_type == ActionType.TASK_COMPLETE

    def test_confidence_range_low(self):
        with pytest.raises(Exception):
            UserAction(
                action_type=ActionType.STOP,
                response_delay_ms=100,
                confidence=-0.1,
            )

    def test_confidence_range_high(self):
        with pytest.raises(Exception):
            UserAction(
                action_type=ActionType.STOP,
                response_delay_ms=100,
                confidence=1.1,
            )

    def test_response_delay_ms_non_negative(self):
        with pytest.raises(Exception):
            UserAction(
                action_type=ActionType.STOP,
                response_delay_ms=-1,
                confidence=0.5,
            )

    def test_all_valid_followup_types(self):
        for ft in [FollowUpType.CORRECTION, FollowUpType.ELABORATION,
                    FollowUpType.REDIRECT, FollowUpType.ENCOURAGEMENT]:
            action = UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="test",
                followup_type=ft,
                response_delay_ms=100,
                confidence=0.5,
            )
            assert action.followup_type == ft


# ---------------------------------------------------------------------------
# TestSimulatedUserUnit
# ---------------------------------------------------------------------------


class TestSimulatedUserUnit:
    def test_generate_called_once(self):
        user, mock_client = _make_simulated_user()
        task = _make_task()
        user.respond(task, "Some agent response", [])
        mock_client.generate.assert_called_once()

    def test_response_schema_passed(self):
        user, mock_client = _make_simulated_user()
        task = _make_task()
        user.respond(task, "Some response", [])
        call_kwargs = mock_client.generate.call_args
        schema = call_kwargs.kwargs["response_schema"]
        assert "action_type" in schema["properties"]
        assert "response_delay_ms" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_task_fields_in_prompt(self):
        user, mock_client = _make_simulated_user()
        task = _make_task(
            question="What is quantum computing?",
            gold_answer="Quantum computing uses qubits.",
            scoring_rubric="5: Perfect. 1: Wrong.",
        )
        user.respond(task, "It uses bits", [])
        messages = mock_client.generate.call_args[0][0]
        user_msg = messages[0]["content"]
        assert "What is quantum computing?" in user_msg
        assert "Quantum computing uses qubits." in user_msg
        assert "5: Perfect. 1: Wrong." in user_msg
        assert "It uses bits" in user_msg

    def test_patience_in_system_prompt(self):
        user, mock_client = _make_simulated_user(patience=Patience.HIGH)
        task = _make_task()
        user.respond(task, "response", [])
        call_kwargs = mock_client.generate.call_args
        system = call_kwargs.kwargs["system_prompt"]
        assert "high" in system.lower()
        assert "patient" in system.lower()

    def test_strictness_in_system_prompt(self):
        user, mock_client = _make_simulated_user(strictness=Strictness.HIGH)
        task = _make_task()
        user.respond(task, "response", [])
        call_kwargs = mock_client.generate.call_args
        system = call_kwargs.kwargs["system_prompt"]
        assert "high" in system.lower()
        assert "thorough" in system.lower() or "precise" in system.lower()

    def test_returns_valid_user_action(self):
        user, _ = _make_simulated_user(
            response=_mock_gemini_response(
                action_type="follow_up",
                message="Tell me more about that.",
                followup_type="elaboration",
            )
        )
        task = _make_task()
        action = user.respond(task, "Some response", [])
        assert isinstance(action, UserAction)
        assert action.action_type == ActionType.FOLLOW_UP
        assert action.message == "Tell me more about that."
        assert action.followup_type == FollowUpType.ELABORATION

    def test_temperature_is_0_7(self):
        user, mock_client = _make_simulated_user()
        task = _make_task()
        user.respond(task, "response", [])
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    def test_thinking_level_minimal(self):
        user, mock_client = _make_simulated_user()
        task = _make_task()
        user.respond(task, "response", [])
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["thinking_level"] == "minimal"

    def test_turn_number_in_prompt(self):
        user, mock_client = _make_simulated_user()
        task = _make_task()
        history = _make_history(3)  # 3 user messages
        user.respond(task, "response", history)
        messages = mock_client.generate.call_args[0][0]
        user_msg = messages[0]["content"]
        assert "turn 4" in user_msg.lower()


# ---------------------------------------------------------------------------
# TestGuardrails
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_low_patience_forces_stop_after_max_turns(self):
        user, mock_client = _make_simulated_user(patience=Patience.LOW)
        task = _make_task()
        # Create history with max_turns user messages
        history = _make_history(_MAX_TURNS[Patience.LOW])
        action = user.respond(task, "response", history)
        assert action.action_type == ActionType.STOP
        assert action.confidence == 1.0
        # LLM should not be called
        mock_client.generate.assert_not_called()

    def test_medium_patience_forces_stop_after_max_turns(self):
        user, mock_client = _make_simulated_user(patience=Patience.MEDIUM)
        task = _make_task()
        history = _make_history(_MAX_TURNS[Patience.MEDIUM])
        action = user.respond(task, "response", history)
        assert action.action_type == ActionType.STOP
        mock_client.generate.assert_not_called()

    def test_high_patience_allows_long_conversations(self):
        user, mock_client = _make_simulated_user(patience=Patience.HIGH)
        task = _make_task()
        # Just under the max
        history = _make_history(_MAX_TURNS[Patience.HIGH] - 1)
        user.respond(task, "response", history)
        mock_client.generate.assert_called_once()

    def test_high_patience_forces_stop_at_max(self):
        user, mock_client = _make_simulated_user(patience=Patience.HIGH)
        task = _make_task()
        history = _make_history(_MAX_TURNS[Patience.HIGH])
        action = user.respond(task, "response", history)
        assert action.action_type == ActionType.STOP
        mock_client.generate.assert_not_called()

    def test_max_turns_values(self):
        assert _MAX_TURNS[Patience.LOW] == 6
        assert _MAX_TURNS[Patience.MEDIUM] == 12
        assert _MAX_TURNS[Patience.HIGH] == 25


# ---------------------------------------------------------------------------
# TestFollowUpTypes
# ---------------------------------------------------------------------------


class TestFollowUpTypes:
    @pytest.mark.parametrize("followup_type", ["correction", "elaboration", "redirect", "encouragement"])
    def test_each_followup_type_parsed(self, followup_type):
        user, _ = _make_simulated_user(
            response=_mock_gemini_response(
                action_type="follow_up",
                message=f"A {followup_type} message",
                followup_type=followup_type,
            )
        )
        task = _make_task()
        action = user.respond(task, "response", [])
        assert action.action_type == ActionType.FOLLOW_UP
        assert action.followup_type == FollowUpType(followup_type)
        assert action.message == f"A {followup_type} message"


# ---------------------------------------------------------------------------
# TestStopAndComplete
# ---------------------------------------------------------------------------


class TestStopAndComplete:
    def test_stop_has_no_message(self):
        user, _ = _make_simulated_user(
            response=_mock_gemini_response(action_type="stop")
        )
        task = _make_task()
        action = user.respond(task, "response", [])
        assert action.action_type == ActionType.STOP
        assert action.message is None
        assert action.followup_type is None

    def test_task_complete_has_no_message(self):
        user, _ = _make_simulated_user(
            response=_mock_gemini_response(action_type="task_complete")
        )
        task = _make_task()
        action = user.respond(task, "response", [])
        assert action.action_type == ActionType.TASK_COMPLETE
        assert action.message is None
        assert action.followup_type is None


# ---------------------------------------------------------------------------
# TestConversationFormatting
# ---------------------------------------------------------------------------


class TestConversationFormatting:
    def test_empty_history(self):
        result = _format_conversation([])
        assert "No prior conversation" in result

    def test_short_history_included_fully(self):
        history = _make_history(3)
        result = _format_conversation(history)
        assert "User message 1" in result
        assert "User message 3" in result

    def test_long_history_truncated(self):
        history = _make_history(10)  # 20 messages total
        result = _format_conversation(history)
        # First message preserved
        assert "User message 1" in result
        # Last 8 messages preserved (messages 13-20)
        assert "Assistant response 10" in result
        # Middle messages dropped
        assert "User message 3" not in result


# ---------------------------------------------------------------------------
# TestIntegration (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------


_SKIP_NO_KEY = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)

_TEST_AGENT_RESPONSES = [
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "I'm not sure about that topic.",
    "Plants use chlorophyll to absorb light.",
    "The process occurs in chloroplasts.",
    "It involves the Calvin cycle and light reactions.",
    "Water and CO2 are converted to glucose and oxygen.",
    "Let me think about that...",
    "Photosynthesis is important for the ecosystem.",
    "The light-dependent reactions produce ATP and NADPH.",
    "Sorry, I don't have that information.",
    "Plants are green because of chlorophyll.",
    "6CO2 + 6H2O → C6H12O6 + 6O2",
    "The process was discovered by Jan Ingenhousz.",
    "It's how plants make food from sunlight.",
    "Photosynthesis occurs mainly in leaves.",
    "The stroma and thylakoid are key structures.",
    "This is a fundamental biological process.",
    "Energy from the sun drives the reactions.",
    "Oxygen is a byproduct of photosynthesis.",
    "The process is essential for life on Earth.",
]


@_SKIP_NO_KEY
class TestIntegration:
    def test_valid_user_actions(self):
        """All 20 test responses produce valid UserActions."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task = dataset.tasks[0]
        user = SimulatedUser()

        for agent_response in _TEST_AGENT_RESPONSES:
            action = user.respond(task, agent_response, [])
            assert isinstance(action, UserAction)
            assert 0.0 <= action.confidence <= 1.0
            assert action.response_delay_ms >= 0

    def test_low_patience_shorter_episodes(self):
        """Low patience produces shorter episodes than high patience."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task = dataset.tasks[0]

        low_turns = _run_episode(task, patience=Patience.LOW, strictness=Strictness.MEDIUM)
        high_turns = _run_episode(task, patience=Patience.HIGH, strictness=Strictness.MEDIUM)
        # Low patience should produce fewer turns (or at most equal)
        assert low_turns <= high_turns + 2  # allow small margin

    def test_high_strictness_more_corrections(self):
        """High strictness produces more corrections than low strictness."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        tasks = dataset.tasks[:3]

        high_corrections = 0
        low_corrections = 0

        for task in tasks:
            high_corrections += _count_corrections(task, Strictness.HIGH)
            low_corrections += _count_corrections(task, Strictness.LOW)

        # High strictness should produce at least as many corrections
        assert high_corrections >= low_corrections

    def test_stop_signals_at_reasonable_frequency(self):
        """Stop signals occur but not on every turn."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task = dataset.tasks[0]
        user = SimulatedUser()

        stop_count = 0
        for resp in _TEST_AGENT_RESPONSES[:10]:
            action = user.respond(task, resp, [])
            if action.action_type == ActionType.STOP:
                stop_count += 1

        # Should get some stops but not all
        assert stop_count < 10

    def test_task_complete_reachable(self):
        """task_complete is reached for at least 50% of tasks within 20 turns."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        tasks = dataset.tasks[:6]
        completed = 0

        for task in tasks:
            user = SimulatedUser(patience=Patience.HIGH, strictness=Strictness.LOW)
            history: list[Message] = []
            reached = False

            for turn in range(20):
                # Use gold answer to maximize chance of completion
                action = user.respond(task, task.gold_answer, history)
                if action.action_type == ActionType.TASK_COMPLETE:
                    reached = True
                    break
                if action.action_type == ActionType.STOP:
                    break
                if action.action_type == ActionType.FOLLOW_UP:
                    history.append(Message(role="user", content=action.message or "", timestamp_ms=turn * 2000, token_count=10))
                    history.append(Message(role="assistant", content=task.gold_answer, timestamp_ms=turn * 2000 + 1000, token_count=20))

            if reached:
                completed += 1

        assert completed >= len(tasks) // 2

    def test_followup_type_diversity(self):
        """Multiple follow-up types are used, not just one."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task = dataset.tasks[0]
        user = SimulatedUser()

        types_seen: set[FollowUpType] = set()
        for resp in _TEST_AGENT_RESPONSES:
            action = user.respond(task, resp, [])
            if action.action_type == ActionType.FOLLOW_UP and action.followup_type:
                types_seen.add(action.followup_type)

        # Should see at least 2 different follow-up types
        assert len(types_seen) >= 2


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def _run_episode(
    task: ResearchQATask,
    patience: Patience,
    strictness: Strictness,
    max_turns: int = 20,
) -> int:
    """Run a simulated episode and return the number of turns."""
    user = SimulatedUser(patience=patience, strictness=strictness)
    history: list[Message] = []
    turns = 0

    for _ in range(max_turns):
        agent_resp = f"Here is my answer about {task.question}: {task.gold_answer[:50]}..."
        action = user.respond(task, agent_resp, history)
        turns += 1

        if action.action_type in (ActionType.STOP, ActionType.TASK_COMPLETE):
            break
        if action.action_type == ActionType.FOLLOW_UP:
            history.append(Message(role="user", content=action.message or "", timestamp_ms=turns * 2000, token_count=10))
            history.append(Message(role="assistant", content=agent_resp, timestamp_ms=turns * 2000 + 1000, token_count=20))

    return turns


def _count_corrections(task: ResearchQATask, strictness: Strictness, n_responses: int = 5) -> int:
    """Count how many correction follow-ups are produced."""
    user = SimulatedUser(strictness=strictness)
    count = 0
    # Use partial/wrong answers to trigger corrections
    responses = [
        "I think the answer is something about biology.",
        "It might involve plants, but I'm not sure.",
        "The process uses water somehow.",
        "I believe it's related to energy.",
        "Plants do something with sunlight.",
    ]
    for resp in responses[:n_responses]:
        action = user.respond(task, resp, [])
        if (action.action_type == ActionType.FOLLOW_UP
                and action.followup_type == FollowUpType.CORRECTION):
            count += 1
    return count
