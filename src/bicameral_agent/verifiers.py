"""Verifier registry: pick a scoring backend by metric name (Issue #56).

Formalizes the contract that ``TaskScorer`` / ``LexicalScorer`` already
duck-type -- ``score(task, answer) -> TaskScore`` -- as a plain dict registry
mirroring ``model_client.build_client``. Datasets declare a ``default_metric``
(overridable within their ``supported_metrics``); callers turn the resolved
metric name into a concrete verifier here.

Deterministic verifiers (``exact_match``, ``multiple_choice``) make no LLM
calls and emit binary 1.0/0.0 scores; LLM-backed verifiers
(``rubric_coverage``, ``abstention``) grade against per-task rubric items or
abstention labels. All verifiers return the existing ``TaskScore`` shape,
using ``TaskScore.detail`` for the mode-specific report.
"""

from __future__ import annotations

import re
from typing import Callable, Protocol, runtime_checkable

from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.llm_output import safe_parse_json
from bicameral_agent.scorer import LexicalScorer, TaskScore, TaskScorer


@runtime_checkable
class Verifier(Protocol):
    """Anything that scores an agent answer against a task."""

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore: ...


def _binary_score(correct: bool, detail: str) -> TaskScore:
    """A 1.0/0.0 TaskScore for deterministic verifiers."""
    v = 1.0 if correct else 0.0
    return TaskScore(quality=v, completeness=v, accuracy=v, overall=v, detail=detail)


def _clip(text: str, limit: int = 80) -> str:
    """Truncate long answers so TaskScore.detail stays readable."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


_ARTICLES = frozenset({"a", "an", "the"})

# Trailing "answer is X" / "Answer: X" extraction for verbose responses.
_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer)\s*(?:is|:)\s*(.+?)\s*(?:\.\s*)?$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_answer(text: str) -> str:
    """Normalize for exact-match comparison.

    Lowercases, strips punctuation, drops articles, collapses whitespace --
    the standard short-form-QA normalization (SQuAD-style).
    """
    lowered = "".join(
        ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower()
    )
    return " ".join(t for t in lowered.split() if t not in _ARTICLES)


def _as_number(text: str) -> float | None:
    """Parse a numeric answer, tolerating $ , % decorations."""
    cleaned = text.strip().rstrip("%").replace(",", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


class ExactMatchVerifier:
    """Deterministic normalized/numeric equality against the gold answer.

    The full response is compared, plus any trailing "answer is X" /
    "Answer: X" extraction, so a verbose response that states the right
    final answer still matches. No LLM calls.
    """

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore:
        gold = task.gold_answer
        candidates = [agent_answer]
        extracted = _FINAL_ANSWER_RE.findall(agent_answer)
        if extracted:
            candidates.append(extracted[-1])
        correct = any(self._matches(gold, c) for c in candidates)
        shown = _clip(candidates[-1])
        return _binary_score(
            correct,
            f"exact_match: expected {_clip(gold)!r}, got {shown!r} -> "
            f"{'match' if correct else 'no match'}",
        )

    @staticmethod
    def _matches(gold: str, candidate: str) -> bool:
        norm_gold = normalize_answer(gold)
        if norm_gold and norm_gold == normalize_answer(candidate):
            return True
        gold_num, cand_num = _as_number(gold), _as_number(candidate)
        return gold_num is not None and cand_num is not None and gold_num == cand_num


_MC_ANSWER_RE = re.compile(
    r"(?:final answer|answer)\s*(?:is|:)?\s*\(?([A-Ja-j])\)?(?![A-Za-z])",
    re.IGNORECASE,
)
_LONE_LETTER_RE = re.compile(r"^\W*([A-Ja-j])\W*$")


class MultipleChoiceVerifier:
    """Deterministic letter-choice grading (gold_answer is a letter A-J).

    Extraction order: explicit "answer is X" statements (last wins), a bare
    letter response, then a unique match of one option's text within the
    response (using ``task.choices``). No LLM calls.
    """

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore:
        expected = task.gold_answer.strip().upper()
        extracted = self._extract_letter(agent_answer, task.choices)
        correct = extracted is not None and extracted == expected
        return _binary_score(
            correct,
            f"multiple_choice: extracted {extracted!r}, expected {expected!r}",
        )

    @staticmethod
    def _extract_letter(answer: str, choices: list[str] | None) -> str | None:
        stated = _MC_ANSWER_RE.findall(answer)
        if stated:
            return stated[-1].upper()
        lone = _LONE_LETTER_RE.match(answer.strip())
        if lone:
            return lone.group(1).upper()
        if choices:
            norm_answer = normalize_answer(answer)
            hits = [
                i
                for i, choice in enumerate(choices)
                if normalize_answer(choice)
                and normalize_answer(choice) in norm_answer
            ]
            if len(hits) == 1:
                return chr(ord("A") + hits[0])
        return None


_RUBRIC_SYSTEM_PROMPT = (
    "You are a strict grader. Given a response and a numbered list of rubric "
    "criteria, decide which criteria the response explicitly satisfies. Only "
    "count a criterion as met if the response clearly and correctly addresses "
    "it -- never give the benefit of the doubt."
)

_RUBRIC_USER_TEMPLATE = """\
## Question
{question}

## Response to Grade
{agent_answer}

## Rubric Criteria
{criteria}

Return the 1-based indices of the criteria that the response satisfies."""

_RUBRIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "met_indices": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["met_indices"],
}


class RubricCoverageVerifier:
    """LLM-graded weighted rubric coverage (requires ``task.rubric_items``).

    One judge call decides which criteria the response satisfies; the score
    is earned points over the sum of positive points, clamped to [0, 1]
    (negative-point penalty items subtract, HealthBench-style).
    """

    def __init__(self, client: GeminiClient | None = None) -> None:
        self._client = client or GeminiClient()

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore:
        items = task.rubric_items
        if not items:
            raise ValueError(
                f"Task {task.task_id} has no rubric_items; the rubric_coverage "
                "metric requires a rubric-annotated dataset"
            )
        criteria = "\n".join(
            f"{i}. [{item.points:g} points] {item.criterion}"
            for i, item in enumerate(items, start=1)
        )
        response = self._client.generate(
            [
                {
                    "role": "user",
                    "content": _RUBRIC_USER_TEMPLATE.format(
                        question=task.question,
                        agent_answer=agent_answer,
                        criteria=criteria,
                    ),
                }
            ],
            system_prompt=_RUBRIC_SYSTEM_PROMPT,
            thinking_level="minimal",
            temperature=0,
            max_output_tokens=400,
            response_schema=_RUBRIC_RESPONSE_SCHEMA,
        )
        parsed = safe_parse_json(response, context="RubricCoverageVerifier", default={})
        raw_indices = parsed.get("met_indices") or []
        met = {
            i for i in raw_indices if isinstance(i, int) and 1 <= i <= len(items)
        }
        earned = sum(items[i - 1].points for i in met)
        total = sum(item.points for item in items if item.points > 0)
        overall = min(1.0, max(0.0, earned / total)) if total > 0 else 0.0
        detail = (
            f"rubric_coverage: met {len(met)}/{len(items)} criteria "
            f"({earned:g}/{total:g} points)"
        )
        return TaskScore(
            quality=overall,
            completeness=overall,
            accuracy=overall,
            overall=overall,
            detail=detail,
        )


_ABSTENTION_SYSTEM_PROMPT = (
    "You judge whether a response abstains. A response abstains if it "
    "declines to answer, says the question is unanswerable or underspecified, "
    "or states that it does not know -- rather than committing to a direct "
    "answer."
)

_ABSTENTION_USER_TEMPLATE = """\
## Question
{question}

## Response
{agent_answer}

Did the response abstain rather than give a direct answer?"""

_ABSTENTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"abstained": {"type": "boolean"}},
    "required": ["abstained"],
}


class AbstentionVerifier:
    """LLM-judged abstention correctness (uses ``task.abstention_expected``).

    Scores 1.0 when the response's abstention behavior matches the task's
    label (abstained when it should, answered when it should).
    """

    def __init__(self, client: GeminiClient | None = None) -> None:
        self._client = client or GeminiClient()

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore:
        response = self._client.generate(
            [
                {
                    "role": "user",
                    "content": _ABSTENTION_USER_TEMPLATE.format(
                        question=task.question, agent_answer=agent_answer
                    ),
                }
            ],
            system_prompt=_ABSTENTION_SYSTEM_PROMPT,
            thinking_level="minimal",
            temperature=0,
            max_output_tokens=100,
            response_schema=_ABSTENTION_RESPONSE_SCHEMA,
        )
        parsed = safe_parse_json(response, context="AbstentionVerifier", default={})
        abstained = bool(parsed.get("abstained", False))
        expected = bool(task.abstention_expected)
        correct = abstained == expected
        return _binary_score(
            correct,
            f"abstention: abstained={abstained}, expected_abstention={expected}",
        )


# client -> verifier factories; ``client`` is any model client satisfying the
# build_client contract (only LLM-backed verifiers use it).
_VERIFIERS: dict[str, Callable[[object | None], Verifier]] = {
    "llm_judge": lambda client: TaskScorer(client=client),
    "lexical": lambda client: LexicalScorer(),
    "exact_match": lambda client: ExactMatchVerifier(),
    "multiple_choice": lambda client: MultipleChoiceVerifier(),
    "rubric_coverage": lambda client: RubricCoverageVerifier(client=client),
    "abstention": lambda client: AbstentionVerifier(client=client),
}


def verifier_names() -> list[str]:
    """Names of all registered verification metrics."""
    return sorted(_VERIFIERS)


def build_verifier(metric: str = "llm_judge", client: object | None = None) -> Verifier:
    """Construct the verifier registered under *metric*.

    Args:
        metric: Registry key (see :func:`verifier_names`).
        client: Model client for LLM-backed verifiers; None uses the
            verifier's own default. Ignored by deterministic verifiers.

    Raises:
        ValueError: If *metric* is not registered.
    """
    try:
        factory = _VERIFIERS[metric]
    except KeyError:
        raise ValueError(
            f"Unknown metric {metric!r}; known metrics: {sorted(_VERIFIERS)}"
        ) from None
    return factory(client)
