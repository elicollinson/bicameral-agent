"""Context Refresher tool primitive.

Detects when the reasoning loop has drifted from the original task
requirements. Reads the first user message and last 3 messages, makes a
single LLM call, and returns a concise reminder (≤100 words) or None if
no drift is detected.
"""

from __future__ import annotations

import enum
import hashlib
import json

from bicameral_agent.queue import Priority, QueueItem
from bicameral_agent.schema import Message
from bicameral_agent.tool_primitive import (
    BudgetExceededError,
    ToolMetadata,
    ToolPrimitive,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DriftCategory(str, enum.Enum):
    """Categories of detected reasoning drift."""

    CONSTRAINT_VIOLATION = "constraint_violation"
    SCOPE_CREEP = "scope_creep"
    REQUIREMENT_IGNORED = "requirement_ignored"


def _drift_priority(category: DriftCategory) -> Priority:
    """Map a drift category to a queue priority."""
    if category in (DriftCategory.CONSTRAINT_VIOLATION, DriftCategory.REQUIREMENT_IGNORED):
        return Priority.HIGH
    return Priority.MEDIUM


# ---------------------------------------------------------------------------
# LLM prompt and schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a task-drift detector. Compare the original user requirements against \
the most recent reasoning messages.

Identify any drift:
- constraint_violation: a stated constraint is being broken
- scope_creep: reasoning has expanded beyond the original scope
- requirement_ignored: a specific requirement is not being addressed

If drift is found, produce a concise reminder (≤100 words) that refocuses \
the reasoning on the original task. If no drift is detected, set \
drift_detected to false and leave drifts empty and reminder null."""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "drift_detected": {"type": "boolean"},
        "drifts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["constraint_violation", "scope_creep", "requirement_ignored"],
                    },
                    "description": {"type": "string"},
                },
                "required": ["category", "description"],
            },
        },
        "reminder": {"type": ["string", "null"]},
    },
    "required": ["drift_detected", "drifts", "reminder"],
}


# ---------------------------------------------------------------------------
# ContextRefresher
# ---------------------------------------------------------------------------


class ContextRefresher(ToolPrimitive):
    """Detects reasoning drift and produces a concise refocusing reminder.

    Uses exactly 1 LLM call comparing the original task against recent
    messages to detect constraint violations, scope creep, or ignored
    requirements.
    """

    def __init__(self) -> None:
        super().__init__("context_refresher")

    def _execute(self, conversation_history, reasoning_state, client):
        # Need at least 2 messages and a user message to be meaningful
        first_user = _extract_first_user_message(conversation_history)
        if first_user is None or len(conversation_history) < 2:
            return ToolResult(
                queue_deposit=None,
                metadata=ToolMetadata(
                    tool_id=self.tool_id,
                    action_taken="skipped: insufficient conversation history",
                    confidence=0.0,
                    items_found=0,
                    estimated_relevance=0.0,
                ),
            )

        # Format the comparison prompt
        recent = conversation_history[-3:]
        recent_text = _format_recent_messages(recent)
        user_content = (
            f"## Original Task\n{first_user.content}\n\n"
            f"## Recent Reasoning\n{recent_text}"
        )

        try:
            response = client.generate(
                [{"role": "user", "content": user_content}],
                system_prompt=_SYSTEM_PROMPT,
                thinking_level="low",
                temperature=0,
                max_output_tokens=200,
                response_schema=_RESPONSE_SCHEMA,
            )
        except BudgetExceededError:
            return ToolResult(
                queue_deposit=None,
                metadata=ToolMetadata(
                    tool_id=self.tool_id,
                    action_taken="skipped: budget exceeded",
                    confidence=0.0,
                    items_found=0,
                    estimated_relevance=0.0,
                ),
            )

        parsed = _parse_json(response.content)

        if not parsed.get("drift_detected", False):
            return ToolResult(
                queue_deposit=None,
                metadata=ToolMetadata(
                    tool_id=self.tool_id,
                    action_taken="checked for drift, none detected",
                    confidence=0.8,
                    items_found=0,
                    estimated_relevance=0.0,
                ),
            )

        # Parse drifts and compute priority
        drifts = []
        for item in parsed.get("drifts", []):
            try:
                category = DriftCategory(item["category"])
            except (ValueError, KeyError):
                category = DriftCategory.SCOPE_CREEP
            drifts.append((category, item.get("description", "")))

        reminder = parsed.get("reminder") or ""
        # Truncate to 100 words if LLM exceeded limit
        words = reminder.split()[:100]
        reminder = " ".join(words)

        max_priority = max((_drift_priority(cat) for cat, _ in drifts), default=Priority.MEDIUM)
        dedup_key = _make_dedup_key(drifts)

        return ToolResult(
            queue_deposit=QueueItem(
                content=reminder,
                priority=max_priority,
                source_tool_id=self.tool_id,
                token_count=len(words),
                expiry_turns=3,
                dedup_key=dedup_key,
            ),
            metadata=ToolMetadata(
                tool_id=self.tool_id,
                action_taken="detected reasoning drift",
                confidence=0.7,
                items_found=len(drifts),
                estimated_relevance=0.6,
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM response, handling preamble text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        # Find the first top-level JSON object by matching balanced braces
        start = text.find("{")
        if start != -1:
            decoder = json.JSONDecoder()
            try:
                obj, _ = decoder.raw_decode(text, start)
                return obj
            except json.JSONDecodeError:
                pass
        return {"drift_detected": False, "drifts": [], "reminder": None}


def _extract_first_user_message(history: list[Message]) -> Message | None:
    """Return the first message with role == 'user', or None."""
    for msg in history:
        if msg.role == "user":
            return msg
    return None


def _format_recent_messages(messages: list[Message]) -> str:
    """Format a list of messages as [role]: content lines."""
    return "\n".join(f"[{msg.role}]: {msg.content}" for msg in messages)


def _make_dedup_key(drifts: list[tuple[DriftCategory, str]]) -> str:
    """SHA-256 hash of sorted drift descriptions, prefixed context_refresher:."""
    descriptions = sorted(desc for _, desc in drifts)
    h = hashlib.sha256("|".join(descriptions).encode()).hexdigest()
    return f"context_refresher:{h}"
