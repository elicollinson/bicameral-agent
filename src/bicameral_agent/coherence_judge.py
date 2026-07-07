"""LLM-as-judge for reasoning coherence evaluation.

Scores conversation transcripts on logical flow, consistency, and overall
coherence using Gemini Flash. Thread-safe with caching and batch support.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from bicameral_agent.gemini import GeminiClient
from bicameral_agent.llm_output import coerce_int, safe_parse_json
from bicameral_agent.schema import Message
from bicameral_agent.scorer import CachedConcurrentScorer, _normalize_score


class CoherenceScore(BaseModel):
    """Coherence scores for a conversation, all in [0.0, 1.0]."""

    logical_flow: float = Field(ge=0.0, le=1.0)
    """How well reasoning progresses logically from point to point."""

    consistency: float = Field(ge=0.0, le=1.0)
    """Internal consistency — no contradictions or reversals."""

    overall: float = Field(ge=0.0, le=1.0)
    """Overall coherence of the conversation."""

    @classmethod
    def from_raw(cls, logical_flow: int, consistency: int, overall: int) -> CoherenceScore:
        """Create from raw 1-5 integer scores, normalizing to [0, 1]."""
        lf = _normalize_score(logical_flow)
        c = _normalize_score(consistency)
        o = _normalize_score(overall)
        return cls(logical_flow=lf, consistency=c, overall=o)


_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the coherence of a multi-turn "
    "conversation between a user and an AI assistant. Score the assistant's "
    "reasoning on three dimensions using an integer from 1 to 5."
)

_JUDGE_USER_TEMPLATE = """\
## Conversation Transcript
{transcript}

Rate the assistant's coherence on each dimension (1-5):
- logical_flow: How well does the reasoning progress logically? \
(5 = clear logical progression, 1 = disjointed/random)
- consistency: Is the reasoning internally consistent? \
(5 = no contradictions, 1 = frequent contradictions or reversals)
- overall: Overall coherence of the conversation. \
(5 = highly coherent, 1 = incoherent)"""

_JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "logical_flow": {"type": "integer"},
        "consistency": {"type": "integer"},
        "overall": {"type": "integer"},
    },
    "required": ["logical_flow", "consistency", "overall"],
}


class CoherenceJudge(CachedConcurrentScorer):
    """LLM-as-judge for conversation coherence.

    Thread-safe with caching. Uses Gemini Flash for scoring.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        max_workers: int = 10,
    ) -> None:
        super().__init__(max_workers=max_workers)
        self._client = client or GeminiClient()

    def score(self, messages: list[Message]) -> CoherenceScore:
        """Score a conversation's coherence.

        Returns cached result if this conversation was scored before.
        """
        return self._score_cached(self._cache_key(messages), messages)

    def score_batch(
        self,
        conversations: list[list[Message]],
    ) -> list[CoherenceScore]:
        """Score multiple conversations concurrently."""
        keys = [self._cache_key(msgs) for msgs in conversations]
        return self._score_batch_cached(keys, conversations)

    @staticmethod
    def _cache_key(messages: list[Message]) -> str:
        content = "|".join(f"{m.role}:{m.content}" for m in messages)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def _score_uncached(self, messages: list[Message]) -> CoherenceScore:
        transcript = _format_transcript(messages)
        user_msg = _JUDGE_USER_TEMPLATE.format(transcript=transcript)
        response = self._client.generate(
            [{"role": "user", "content": user_msg}],
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            thinking_level="minimal",
            temperature=0,
            max_output_tokens=100,
            response_schema=_JUDGE_RESPONSE_SCHEMA,
        )
        # Malformed/truncated judge output degrades to a neutral mid-scale
        # score (3 normalizes to 0.5); extra keys are ignored, not splatted.
        parsed = safe_parse_json(response, context="CoherenceJudge", default={})
        return CoherenceScore.from_raw(
            logical_flow=coerce_int(parsed.get("logical_flow"), 3),
            consistency=coerce_int(parsed.get("consistency"), 3),
            overall=coerce_int(parsed.get("overall"), 3),
        )


def _format_transcript(messages: list[Message]) -> str:
    """Format messages as a readable transcript."""
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"[{role}]: {msg.content}")
    return "\n\n".join(lines)
