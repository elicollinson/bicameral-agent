"""Automated task quality scorer using LLM-as-judge and lexical baselines.

Provides TaskScorer (Gemini-based LLM judge) and LexicalScorer (ROUGE-L/F1)
for evaluating agent answers against research QA tasks with scoring rubrics.
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.llm_output import coerce_int, safe_parse_json, tokenize


def _normalize_score(v: int) -> float:
    """Normalize a raw 1-5 integer score to [0.0, 1.0], clamping to [1, 5]."""
    return (max(1, min(5, v)) - 1) / 4.0


class TaskScore(BaseModel):
    """Normalized quality scores for a task answer, all in [0.0, 1.0]."""

    quality: float = Field(ge=0.0, le=1.0)
    """Overall quality per the task's scoring rubric."""

    completeness: float = Field(ge=0.0, le=1.0)
    """Coverage of the reference answer's key points."""

    accuracy: float = Field(ge=0.0, le=1.0)
    """Factual correctness relative to the reference answer."""

    overall: float = Field(ge=0.0, le=1.0)
    """Weighted aggregate score. Compatible with EpisodeOutcome.quality_score."""

    @classmethod
    def from_raw(cls, quality: int, completeness: int, accuracy: int) -> TaskScore:
        """Create from raw 1-5 integer scores, normalizing to [0, 1].

        Values are clamped to [1, 5] before normalization.
        The overall score is the mean of the three normalized dimensions.
        """
        q, c, a = _normalize_score(quality), _normalize_score(completeness), _normalize_score(accuracy)
        return cls(quality=q, completeness=c, accuracy=a, overall=(q + c + a) / 3.0)


_JUDGE_SYSTEM_PROMPT = (
    "You are a demanding expert evaluator scoring research question answers "
    "against a reference answer and rubric. Use the full 1-5 range: a merely "
    "reasonable answer earns 3, and 5 is reserved for flawless answers that "
    "cover every key point of the reference with no errors. Score only what "
    "is explicitly present in the answer — never give the benefit of the "
    "doubt, and when torn between two adjacent scores, choose the lower one. "
    "Justify your scores against specific reference key points before giving "
    "them."
)

_JUDGE_USER_TEMPLATE = """\
## Task
Question: {question}

## Reference Answer
{gold_answer}

## Scoring Rubric
{scoring_rubric}

## Agent Answer to Score
{agent_answer}

First, in "justification", state concretely which key points of the \
reference answer the agent answer covers, which it misses, and any factual \
errors (1-3 sentences).

Then rate the agent answer on each dimension using an integer from 1 to 5, \
anchored as follows:
- 5 = flawless: every key point of the reference covered accurately; top \
rubric level; nothing missing, nothing wrong. This should be rare.
- 4 = strong: at most one minor omission or imprecision; no factual errors.
- 3 = adequate: the central idea is right but key points are missing, \
vague, or shallow. This is the expected score for a merely reasonable answer.
- 2 = weak: touches the topic but misses most key points, or contains a \
significant factual error.
- 1 = failing: wrong, off-topic, empty, or fabricated.

Dimensions:
- quality: overall quality according to the scoring rubric above.
- completeness: coverage of the reference answer's key points.
- accuracy: factual correctness relative to the reference answer.

Hard caps: if any key point is missing, completeness is at most 3. If there \
is any factual error, accuracy is at most 3; if the error is major or \
fabricated, at most 2."""

# "justification" is deliberately first: the judge must commit to a concrete
# coverage/error analysis before emitting scores (rationale-first scoring),
# which is what breaks the reflexive 5/5/5 ceiling.
_JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "justification": {"type": "string"},
        "quality": {"type": "integer"},
        "completeness": {"type": "integer"},
        "accuracy": {"type": "integer"},
    },
    "required": ["justification", "quality", "completeness", "accuracy"],
}

# The old cap of 100 tokens left no room for any reasoning before the
# scores; 400 accommodates the rationale-first justification.
_JUDGE_MAX_OUTPUT_TOKENS = 400


class CachedConcurrentScorer:
    """Shared lock/cache/ThreadPool machinery for LLM-as-judge scorers.

    Subclasses implement ``_score_uncached(item)`` and call
    ``_score_cached`` / ``_score_batch_cached`` with a hashable cache key
    per item. Thread-safe; batch scoring dispatches uncached items to a
    ThreadPoolExecutor. Shared by TaskScorer and CoherenceJudge (issue #54).
    """

    def __init__(self, max_workers: int = 10) -> None:
        self._max_workers = max_workers
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _score_uncached(self, item):
        """Compute the score for one item. Implement in subclasses."""
        raise NotImplementedError

    def _score_cached(self, key, item):
        """Return the cached score for ``key``, computing and caching on miss."""
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._score_uncached(item)
        with self._lock:
            self._cache[key] = result
        return result

    def _score_batch_cached(self, keys: list, items: list) -> list:
        """Score items concurrently, checking the cache per item first."""
        results: dict[int, object] = {}
        uncached_indices: list[int] = []

        for i, key in enumerate(keys):
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                future_to_idx = {
                    pool.submit(self._score_uncached, items[i]): i
                    for i in uncached_indices
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    score = future.result()
                    results[idx] = score
                    with self._lock:
                        self._cache[keys[idx]] = score

        return [results[i] for i in range(len(keys))]

    @property
    def cache_size(self) -> int:
        """Number of cached scores."""
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the score cache."""
        with self._lock:
            self._cache.clear()


class TaskScorer(CachedConcurrentScorer):
    """LLM-as-judge scorer using Gemini Flash.

    Scores agent answers against ResearchQATask rubrics and gold answers.
    Caches results to avoid re-scoring identical (task, answer) pairs.
    Thread-safe.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        max_workers: int = 10,
    ) -> None:
        super().__init__(max_workers=max_workers)
        self._client = client or GeminiClient()

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore:
        """Score a single agent answer against a task.

        Returns cached result if this (task, answer) pair was scored before.
        """
        key = self._cache_key(task.task_id, agent_answer)
        return self._score_cached(key, (task, agent_answer))

    def score_batch(
        self,
        tasks: list[ResearchQATask],
        answers: list[str],
    ) -> list[TaskScore]:
        """Score multiple (task, answer) pairs concurrently.

        Uses ThreadPoolExecutor for parallelism. Cache is checked per-item
        before dispatching to the thread pool.
        """
        if len(tasks) != len(answers):
            raise ValueError(
                f"Length mismatch: {len(tasks)} tasks vs {len(answers)} answers"
            )
        keys = [self._cache_key(t.task_id, a) for t, a in zip(tasks, answers)]
        return self._score_batch_cached(keys, list(zip(tasks, answers)))

    @staticmethod
    def _cache_key(task_id: str, answer: str) -> tuple[str, str]:
        """Compute cache key from task_id and answer hash."""
        h = hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]
        return (task_id, h)

    def _score_uncached(self, item: tuple[ResearchQATask, str]) -> TaskScore:
        """Call the LLM judge and return a TaskScore."""
        task, answer = item
        user_msg = _JUDGE_USER_TEMPLATE.format(
            question=task.question,
            gold_answer=task.gold_answer,
            scoring_rubric=task.scoring_rubric,
            agent_answer=answer,
        )
        response = self._client.generate(
            [{"role": "user", "content": user_msg}],
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            thinking_level="minimal",
            temperature=0,
            max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
            response_schema=_JUDGE_RESPONSE_SCHEMA,
        )
        # Malformed/truncated judge output degrades to a neutral mid-scale
        # score; missing or non-integer dimensions default to 3 (from_raw
        # normalizes 3 -> 0.5). Extra keys are ignored rather than splatted.
        parsed = safe_parse_json(response, context="TaskScorer", default={})
        return TaskScore.from_raw(
            quality=coerce_int(parsed.get("quality"), 3),
            completeness=coerce_int(parsed.get("completeness"), 3),
            accuracy=coerce_int(parsed.get("accuracy"), 3),
        )


def _f_measure(precision: float, recall: float) -> float:
    """Compute F1-measure from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _token_f1(reference: str, hypothesis: str) -> tuple[float, float, float]:
    """Compute unigram precision, recall, F1 between reference and hypothesis.

    Returns (precision, recall, f1) each in [0.0, 1.0].
    """
    ref_tokens = set(tokenize(reference))
    hyp_tokens = set(tokenize(hypothesis))
    if not ref_tokens or not hyp_tokens:
        return (0.0, 0.0, 0.0)
    common = ref_tokens & hyp_tokens
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    return (precision, recall, _f_measure(precision, recall))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Space-optimized: O(min(m, n)) memory using two rows.
    """
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def _rouge_l(reference: str, hypothesis: str) -> tuple[float, float, float]:
    """Compute ROUGE-L precision, recall, F-measure.

    Returns (precision, recall, f_measure) each in [0.0, 1.0].
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return (0.0, 0.0, 0.0)
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return (precision, recall, _f_measure(precision, recall))


class LexicalScorer:
    """Token-overlap baseline scorer using F1 and ROUGE-L.

    No LLM calls; purely lexical comparison against the gold answer.
    Deterministic and near-instant.
    """

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore:
        """Score using token F1 and ROUGE-L.

        Mapping: quality=F1, completeness=recall, accuracy=precision,
        overall=ROUGE-L F-measure.
        """
        precision, recall, f1 = _token_f1(task.gold_answer, agent_answer)
        _, _, rouge_l_f = _rouge_l(task.gold_answer, agent_answer)
        return TaskScore(
            quality=f1,
            completeness=recall,
            accuracy=precision,
            overall=rouge_l_f,
        )

    def score_batch(
        self,
        tasks: list[ResearchQATask],
        answers: list[str],
    ) -> list[TaskScore]:
        """Score multiple pairs sequentially (already fast)."""
        if len(tasks) != len(answers):
            raise ValueError(
                f"Length mismatch: {len(tasks)} tasks vs {len(answers)} answers"
            )
        return [self.score(t, a) for t, a in zip(tasks, answers)]
