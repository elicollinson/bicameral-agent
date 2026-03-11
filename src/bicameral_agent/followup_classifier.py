"""Follow-up type classifier for user messages in a conversation.

Classifies user follow-up messages into five categories using weighted
keyword/pattern scoring with priority-based tiebreaking.

Design tradeoffs
-----------------
This classifier uses a rule-based approach (keyword patterns with weighted
scoring) rather than ML/LLM approaches. This gives:

- **Speed**: <1ms per classification, no model loading or inference cost.
- **Interpretability**: Every classification can be traced to specific pattern
  matches and their weights.
- **No dependencies**: Works with zero external packages beyond the stdlib.

An ML classifier (e.g. fine-tuned transformer) would improve accuracy on
ambiguous inputs and generalize to unseen phrasings, but at the cost of
latency (~10-100x slower), model size, and training data requirements. An
LLM-based approach would be even more accurate but adds API latency and cost.
For the encoder's feature vector, the rule-based approach is sufficient and
the speed advantage is material since encoding runs on every turn.
"""

from __future__ import annotations

import enum
import re

from bicameral_agent.schema import Message


class FollowUpType(str, enum.Enum):
    """Categories of user follow-up messages."""

    CORRECTION = "correction"
    ELABORATION = "elaboration"
    REDIRECT = "redirect"
    ENCOURAGEMENT = "encouragement"
    NEW_TASK = "new_task"


# One-hot index and tiebreak priority for each type (matches encoder's slot
# order). Lower index = higher priority when scores are tied.
_TYPE_INDEX: dict[FollowUpType, int] = {
    FollowUpType.CORRECTION: 0,
    FollowUpType.REDIRECT: 1,
    FollowUpType.ELABORATION: 2,
    FollowUpType.NEW_TASK: 3,
    FollowUpType.ENCOURAGEMENT: 4,
}

# ---------------------------------------------------------------------------
# Pattern definitions: (pattern, weight) pairs per type
# Multi-word patterns are checked first for specificity.
# ---------------------------------------------------------------------------

_CORRECTION_PATTERNS: list[tuple[str, float]] = [
    # Multi-word (high weight)
    (r"\bnot right\b", 3.0),
    (r"\bthat's wrong\b", 3.0),
    (r"\bthat is wrong\b", 3.0),
    (r"\bnot correct\b", 3.0),
    (r"\bnot what i\b", 2.5),
    (r"\bi said\b", 2.0),
    (r"\bi meant\b", 2.5),
    (r"\bi didn'?t mean\b", 2.5),
    (r"\bi didn'?t say\b", 2.5),
    (r"\bi didn'?t ask\b", 2.0),
    (r"\byou misunderstood\b", 3.0),
    (r"\bthat'?s not\b", 2.0),
    (r"\bthat is not\b", 2.0),
    (r"\byou'?re wrong\b", 3.0),
    (r"\byou are wrong\b", 3.0),
    # Single-word
    (r"\bwrong\b", 2.0),
    (r"\bincorrect\b", 2.0),
    (r"\bmistake\b", 2.0),
    (r"\berror\b", 1.5),
    (r"\bfix\b", 1.5),
    (r"\bcorrection\b", 2.0),
    (r"\bactually\b", 1.5),
    (r"\bno\b", 1.0),
]

_REDIRECT_PATTERNS: list[tuple[str, float]] = [
    (r"\bbut what about\b", 3.0),
    (r"\blet'?s talk about\b", 3.0),
    (r"\blet us talk about\b", 3.0),
    (r"\bchange (?:the )?(?:topic|subject)\b", 3.0),
    (r"\bwhat about\b", 2.0),
    (r"\bhow about\b", 2.0),
    (r"\binstead\b", 2.0),
    (r"\brather\b", 1.5),
    (r"\bdifferent\b", 1.5),
    (r"\bswitch\b", 1.5),
    (r"\btopic\b", 1.0),
    (r"\bfocus on\b", 2.0),
    (r"\bmoving on\b", 2.5),
    (r"\bmove on\b", 2.5),
    (r"\bforget (?:about )?that\b", 2.5),
    (r"\bnever\s?mind\b", 2.5),
]

_ELABORATION_PATTERNS: list[tuple[str, float]] = [
    (r"\btell me more\b", 3.0),
    (r"\bgo (?:into )?more detail\b", 3.0),
    (r"\bcan you explain\b", 2.5),
    (r"\bcould you explain\b", 2.5),
    (r"\bwhat do you mean\b", 2.5),
    (r"\bwhat does that mean\b", 2.5),
    (r"\bin more detail\b", 3.0),
    (r"\belaborate\b", 2.0),
    (r"\bexpand\b", 1.5),
    (r"\bexplain\b", 1.5),
    (r"\bclarify\b", 2.0),
    (r"\bmore\b", 1.0),
    (r"\bdetail\b", 1.5),
    (r"\bdeeper\b", 1.5),
    (r"\bfurther\b", 1.0),
    (r"\bhow\b", 1.5),
    (r"\bwhy\b", 0.5),
    (r"\bspecifically\b", 1.5),
    (r"\bspecifics\b", 1.5),
    (r"\bexample\b", 1.5),
    (r"\bunderstand\b", 1.5),
    (r"\bdon'?t understand\b", 2.5),
    (r"\bwalk me through\b", 3.0),
    (r"\bstep by step\b", 2.5),
]

_NEW_TASK_PATTERNS: list[tuple[str, float]] = [
    (r"\banother thing\b", 2.5),
    (r"\bone more thing\b", 2.5),
    (r"\bcan you also\b", 2.5),
    (r"\bcould you also\b", 2.5),
    (r"\bcan you\b", 1.0),
    (r"\bcould you\b", 1.0),
    (r"\bplease\b", 0.5),
    (r"\bnow\b", 1.0),
    (r"\bnext\b", 1.5),
    (r"\balso\b", 1.0),
    (r"\badditionally\b", 2.0),
    (r"\bnew\b", 1.0),
    (r"\bwrite\b", 1.0),
    (r"\bcreate\b", 1.0),
    (r"\bgenerate\b", 1.0),
    (r"\bbuild\b", 1.0),
    (r"\bimplement\b", 1.5),
    (r"\badd\b", 1.0),
]

_ENCOURAGEMENT_PATTERNS: list[tuple[str, float]] = [
    (r"\bthat'?s (?:exactly|just) what\b", 3.0),
    (r"\bthat is (?:exactly|just) what\b", 3.0),
    (r"\bwell done\b", 3.0),
    (r"\bgood job\b", 3.0),
    (r"\bkeep going\b", 2.5),
    (r"\bkeep it up\b", 2.5),
    (r"\blooks good\b", 2.5),
    (r"\bthat'?s right\b", 2.5),
    (r"\bthat is right\b", 2.5),
    (r"\bthat'?s correct\b", 2.5),
    (r"\bthat is correct\b", 2.5),
    (r"\bthanks\b", 1.5),
    (r"\bthank you\b", 2.0),
    (r"\bgreat\b", 1.5),
    (r"\bgood\b", 1.0),
    (r"\bperfect\b", 2.0),
    (r"\bnice\b", 1.0),
    (r"\bexactly\b", 2.0),
    (r"\byes\b", 1.0),
    (r"\bcorrect\b", 1.5),
    (r"\bright\b", 1.0),
    (r"\bok\b", 0.5),
    (r"\bawesome\b", 2.0),
    (r"\bexcellent\b", 2.0),
    (r"\bamazing\b", 2.0),
    (r"\bwonderful\b", 2.0),
    (r"\bhelpful\b", 1.5),
    (r"\blove it\b", 2.5),
]

# Compile all patterns once at import time
_COMPILED_PATTERNS: dict[FollowUpType, list[tuple[re.Pattern[str], float]]] = {
    FollowUpType.CORRECTION: [(re.compile(p, re.IGNORECASE), w) for p, w in _CORRECTION_PATTERNS],
    FollowUpType.REDIRECT: [(re.compile(p, re.IGNORECASE), w) for p, w in _REDIRECT_PATTERNS],
    FollowUpType.ELABORATION: [(re.compile(p, re.IGNORECASE), w) for p, w in _ELABORATION_PATTERNS],
    FollowUpType.NEW_TASK: [(re.compile(p, re.IGNORECASE), w) for p, w in _NEW_TASK_PATTERNS],
    FollowUpType.ENCOURAGEMENT: [(re.compile(p, re.IGNORECASE), w) for p, w in _ENCOURAGEMENT_PATTERNS],
}

# Maximum message length to consider (truncate beyond this)
_MAX_MESSAGE_LENGTH = 2000


class FollowUpClassifier:
    """Classifies user follow-up messages into five categories.

    Uses weighted keyword/pattern scoring with priority tiebreaking.
    Scoring works by summing weights of all matching patterns per type,
    then selecting the highest-scoring type. On ties, priority order
    determines the winner: correction > redirect > elaboration >
    new_task > encouragement.

    Context-aware rules adjust scores when conversation history provides
    disambiguation cues (e.g., "right" after an assistant question is
    likely encouragement, not a standalone keyword).
    """

    @staticmethod
    def classify(
        user_message: str,
        conversation_history: list[Message] | None = None,
    ) -> FollowUpType:
        """Classify a user follow-up message.

        Parameters
        ----------
        user_message:
            The user's follow-up text.
        conversation_history:
            Prior messages for context-aware disambiguation. Optional.

        Returns
        -------
        FollowUpType
            The predicted follow-up category.
        """
        if not user_message or not user_message.strip():
            return FollowUpType.NEW_TASK

        # Truncate very long messages
        text = user_message[:_MAX_MESSAGE_LENGTH]

        scores = _compute_scores(text)
        _apply_context_rules(scores, text, conversation_history or [])

        return _select_type(scores)

    @staticmethod
    def type_index(follow_up_type: FollowUpType) -> int:
        """Return the one-hot index for the given follow-up type."""
        return _TYPE_INDEX[follow_up_type]


# ---------------------------------------------------------------------------
# Internal scoring functions
# ---------------------------------------------------------------------------


def _compute_scores(text: str) -> dict[FollowUpType, float]:
    """Sum pattern weights for each follow-up type."""
    scores: dict[FollowUpType, float] = {}
    for ftype, patterns in _COMPILED_PATTERNS.items():
        total = 0.0
        for pattern, weight in patterns:
            if pattern.search(text):
                total += weight
        scores[ftype] = total
    return scores


def _apply_context_rules(
    scores: dict[FollowUpType, float],
    text: str,
    history: list[Message],
) -> None:
    """Adjust scores in-place based on conversation context."""
    # If the previous assistant message ended with a question,
    # short affirmative responses are likely encouragement
    if history:
        last_assistant = None
        for msg in reversed(history):
            if msg.role == "assistant":
                last_assistant = msg
                break

        if last_assistant and last_assistant.content.rstrip().endswith("?"):
            # Boost encouragement for short affirmative replies
            if text.count(" ") < 5:
                scores[FollowUpType.ENCOURAGEMENT] = (
                    scores.get(FollowUpType.ENCOURAGEMENT, 0) + 2.0
                )

    # If the message ends with '?', it's likely a question -> boost elaboration
    if text.rstrip().endswith("?"):
        scores[FollowUpType.ELABORATION] = (
            scores.get(FollowUpType.ELABORATION, 0) + 2.0
        )

    # Negation handling: if message starts with "no" or "not",
    # suppress encouragement score
    text_stripped = text.strip().lower()
    if text_stripped.startswith(("no,", "no ", "not ", "nope", "nah")):
        scores[FollowUpType.ENCOURAGEMENT] = 0.0
        # Boost correction if it has any score
        if scores.get(FollowUpType.CORRECTION, 0) > 0:
            scores[FollowUpType.CORRECTION] += 1.0


def _select_type(scores: dict[FollowUpType, float]) -> FollowUpType:
    """Select the type with highest score, using priority for ties."""
    max_score = max(scores.values()) if scores else 0.0

    if max_score == 0.0:
        return FollowUpType.NEW_TASK

    # Among types with the max score, pick by priority (lowest index wins)
    candidates = [ft for ft, s in scores.items() if s == max_score]
    return min(candidates, key=lambda ft: _TYPE_INDEX[ft])
