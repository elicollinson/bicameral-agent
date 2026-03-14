"""LLM-as-judge for reasoning coherence evaluation.

Scores conversation transcripts on logical flow, consistency, and overall
coherence using Gemini Flash. Thread-safe with caching and batch support.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from bicameral_agent.gemini import GeminiClient
from bicameral_agent.schema import Message
from bicameral_agent.scorer import _normalize_score


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


class CoherenceJudge:
    """LLM-as-judge for conversation coherence.

    Thread-safe with caching. Uses Gemini Flash for scoring.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        max_workers: int = 10,
    ) -> None:
        self._client = client or GeminiClient()
        self._max_workers = max_workers
        self._cache: dict[str, CoherenceScore] = {}
        self._lock = threading.Lock()

    def score(self, messages: list[Message]) -> CoherenceScore:
        """Score a conversation's coherence.

        Returns cached result if this conversation was scored before.
        """
        key = self._cache_key(messages)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._score_uncached(messages)
        with self._lock:
            self._cache[key] = result
        return result

    def score_batch(
        self,
        conversations: list[list[Message]],
    ) -> list[CoherenceScore]:
        """Score multiple conversations concurrently."""
        results: dict[int, CoherenceScore] = {}
        uncached_indices: list[int] = []

        for i, msgs in enumerate(conversations):
            key = self._cache_key(msgs)
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                future_to_idx = {
                    pool.submit(self._score_uncached, conversations[i]): i
                    for i in uncached_indices
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    score = future.result()
                    results[idx] = score
                    key = self._cache_key(conversations[idx])
                    with self._lock:
                        self._cache[key] = score

        return [results[i] for i in range(len(conversations))]

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

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
        parsed = json.loads(response.content)
        return CoherenceScore.from_raw(**parsed)


def _format_transcript(messages: list[Message]) -> str:
    """Format messages as a readable transcript."""
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"[{role}]: {msg.content}")
    return "\n\n".join(lines)
