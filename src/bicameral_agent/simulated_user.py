"""Simulated user for generating realistic training episodes.

Uses Gemini Flash to decide whether to follow up, stop, or mark a task
complete based on configurable patience and strictness. Produces UserAction
objects with simulated timing metadata for episode construction.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field, model_validator

from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.llm_output import clamp, coerce_int, safe_parse_json
from bicameral_agent.schema import Message


class ActionType(str, enum.Enum):
    """Types of actions the simulated user can take."""

    FOLLOW_UP = "follow_up"
    STOP = "stop"
    TASK_COMPLETE = "task_complete"


class Patience(str, enum.Enum):
    """How many turns the simulated user will tolerate."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Strictness(str, enum.Enum):
    """How demanding the simulated user is about answer quality."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Valid follow-up types for the simulated user (excludes NEW_TASK)
_VALID_FOLLOWUP_TYPES = {
    FollowUpType.CORRECTION,
    FollowUpType.ELABORATION,
    FollowUpType.REDIRECT,
    FollowUpType.ENCOURAGEMENT,
}

# Maximum turns before forcing a stop, keyed by patience level
_MAX_TURNS: dict[Patience, int] = {
    Patience.LOW: 6,
    Patience.MEDIUM: 12,
    Patience.HIGH: 25,
}


class UserAction(BaseModel):
    """An action taken by the simulated user."""

    action_type: ActionType
    message: str | None = None
    followup_type: FollowUpType | None = None
    response_delay_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_fields_consistent(self) -> UserAction:
        """FOLLOW_UP requires message and followup_type; others must not have them."""
        if self.action_type == ActionType.FOLLOW_UP:
            if self.message is None:
                raise ValueError("FOLLOW_UP action requires a message")
            if self.followup_type is None:
                raise ValueError("FOLLOW_UP action requires a followup_type")
            if self.followup_type not in _VALID_FOLLOWUP_TYPES:
                raise ValueError(
                    f"followup_type must be one of {sorted(ft.value for ft in _VALID_FOLLOWUP_TYPES)}, "
                    f"got {self.followup_type.value!r}"
                )
        else:
            if self.message is not None:
                raise ValueError(f"{self.action_type.value} action must not have a message")
            if self.followup_type is not None:
                raise ValueError(f"{self.action_type.value} action must not have a followup_type")
        return self


_SYSTEM_PROMPT_TEMPLATE = """\
You are simulating a user interacting with an AI research assistant. You are \
evaluating whether the assistant's response adequately answers your research \
question. You have access to the gold-standard answer and scoring rubric.

Your persona:
- Patience: {patience}. {patience_desc}
- Strictness: {strictness}. {strictness_desc}

Guidelines:
- If the assistant's answer is substantially correct and complete per the \
rubric, choose "task_complete".
- If the answer is clearly wrong, off-topic, or the assistant is going in \
circles, choose "stop".
- Otherwise, choose "follow_up" with an appropriate follow-up message.

For follow-ups, select a followup_type:
- "correction": Point out a specific error in the response.
- "elaboration": Ask for more detail on a point.
- "redirect": Steer the conversation back on track.
- "encouragement": Acknowledge progress and ask to continue.

Keep follow-up messages natural, concise (1-2 sentences), and in character \
with your patience/strictness persona."""

_PATIENCE_DESCRIPTIONS: dict[Patience, str] = {
    Patience.LOW: "You get frustrated quickly and give up after a few poor responses.",
    Patience.MEDIUM: "You give the assistant a reasonable chance but won't wait forever.",
    Patience.HIGH: "You are very patient and willing to guide the assistant through many turns.",
}

_STRICTNESS_DESCRIPTIONS: dict[Strictness, str] = {
    Strictness.LOW: "You are lenient and accept approximate or partial answers.",
    Strictness.MEDIUM: "You expect reasonably complete and accurate answers.",
    Strictness.HIGH: "You demand thorough, precise answers that fully address every aspect of the rubric.",
}

_USER_PROMPT_TEMPLATE = """\
## Research Question
{question}

## Gold-Standard Answer
{gold_answer}

## Scoring Rubric
{scoring_rubric}

## Conversation History
{conversation_history}

## Latest Agent Response
{agent_response}

## Turn Number
This is turn {turn_number}.

Decide your next action. Consider whether the agent has adequately addressed \
the research question according to the rubric and gold answer."""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [a.value for a in ActionType],
        },
        "message": {"type": "string"},
        "followup_type": {
            "type": "string",
            "enum": [ft.value for ft in _VALID_FOLLOWUP_TYPES],
        },
        "response_delay_ms": {"type": "integer"},
        "confidence": {"type": "number"},
    },
    "required": ["action_type", "response_delay_ms", "confidence"],
}


def _format_conversation(
    conversation_history: list[Message],
) -> str:
    """Format conversation history, keeping first message + last 8."""
    if not conversation_history:
        return "(No prior conversation)"

    if len(conversation_history) <= 9:
        selected = conversation_history
    else:
        selected = [conversation_history[0]] + conversation_history[-8:]

    lines = []
    for msg in selected:
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role_label}: {msg.content}")
    return "\n\n".join(lines)


class SimulatedUser:
    """Simulated user that generates realistic follow-up behavior.

    Uses Gemini Flash to decide next actions based on configurable patience
    and strictness personas. Each call to respond() makes a single LLM call.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        patience: Patience = Patience.MEDIUM,
        strictness: Strictness = Strictness.MEDIUM,
    ) -> None:
        self._client = client or GeminiClient()
        self._patience = patience
        self._strictness = strictness
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            patience=patience.value,
            patience_desc=_PATIENCE_DESCRIPTIONS[patience],
            strictness=strictness.value,
            strictness_desc=_STRICTNESS_DESCRIPTIONS[strictness],
        )

    def respond(
        self,
        task: ResearchQATask,
        agent_response: str,
        conversation_history: list[Message],
        *,
        turn_number: int,
    ) -> UserAction:
        """Decide the next user action given the agent's latest response.

        Parameters
        ----------
        task:
            The research QA task being worked on.
        agent_response:
            The agent's latest response text.
        conversation_history:
            Full conversation history (truncated internally for prompt).
            May or may not include the current exchange, which is why the
            turn number must be passed explicitly rather than derived here.
        turn_number:
            The 1-based turn the caller is currently on (the turn that
            produced ``agent_response``).

        Returns
        -------
        UserAction
            The simulated user's next action.
        """
        # Guardrail: force stop after max turns for this patience level
        if turn_number > _MAX_TURNS[self._patience]:
            return UserAction(
                action_type=ActionType.STOP,
                response_delay_ms=0,
                confidence=1.0,
            )

        user_msg = _USER_PROMPT_TEMPLATE.format(
            question=task.question,
            gold_answer=task.gold_answer,
            scoring_rubric=task.scoring_rubric,
            conversation_history=_format_conversation(conversation_history),
            agent_response=agent_response,
            turn_number=turn_number,
        )

        response = self._client.generate(
            [{"role": "user", "content": user_msg}],
            system_prompt=self._system_prompt,
            thinking_level="minimal",
            temperature=0.7,
            max_output_tokens=300,
            response_schema=_RESPONSE_SCHEMA,
        )

        raw = safe_parse_json(
            response, context="SimulatedUser.respond", default=None
        )
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: dict | None) -> UserAction:
        """Parse the LLM's structured response into a UserAction.

        Malformed/truncated output degrades to a neutral follow-up rather than
        crashing; the patience guardrail still bounds episode length.
        """
        if raw is None:
            return _fallback_action()
        try:
            action_type = ActionType(raw["action_type"])
        except (KeyError, ValueError):
            return _fallback_action()

        # The LLM occasionally treats confidence as a 1-5 scale; clamp to [0, 1].
        confidence = clamp(raw.get("confidence"), 0.0, 1.0, 0.8)
        # response_delay_ms feeds Field(ge=0); clamp negatives/non-numerics.
        response_delay_ms = max(0, coerce_int(raw.get("response_delay_ms"), 500))
        kwargs: dict = {
            "action_type": action_type,
            "response_delay_ms": response_delay_ms,
            "confidence": confidence,
        }

        if action_type == ActionType.FOLLOW_UP:
            kwargs["message"] = raw.get("message") or "Can you elaborate?"
            try:
                followup_type = FollowUpType(raw.get("followup_type", "elaboration"))
            except ValueError:
                followup_type = FollowUpType.ELABORATION
            if followup_type not in _VALID_FOLLOWUP_TYPES:
                followup_type = FollowUpType.ELABORATION
            kwargs["followup_type"] = followup_type

        return UserAction(**kwargs)


def _fallback_action() -> UserAction:
    """Neutral action used when the LLM's structured output is unusable."""
    return UserAction(
        action_type=ActionType.FOLLOW_UP,
        message="Can you elaborate?",
        followup_type=FollowUpType.ELABORATION,
        response_delay_ms=500,
        confidence=0.5,
    )
