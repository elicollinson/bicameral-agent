"""Tests for the verifier registry (Issue #56). No LLM calls."""

import pytest

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.scorer import LexicalScorer, TaskScorer
from bicameral_agent.verifiers import Verifier, build_verifier, verifier_names


def _task() -> ResearchQATask:
    return ResearchQATask(
        task_id="t1",
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question="What color is the sky?",
        gold_answer="The sky is blue.",
        scoring_rubric="5: correct. 1: incorrect.",
    )


class TestBuildVerifier:
    def test_known_metrics(self):
        assert verifier_names() == ["lexical", "llm_judge"]

    def test_lexical_is_deterministic_scorer(self):
        verifier = build_verifier("lexical")
        assert isinstance(verifier, LexicalScorer)
        score = verifier.score(_task(), "The sky is blue.")
        assert score.overall == 1.0  # exact match, no LLM involved

    def test_llm_judge_receives_client(self):
        sentinel = object()
        verifier = build_verifier("llm_judge", client=sentinel)
        assert isinstance(verifier, TaskScorer)
        assert verifier._client is sentinel

    def test_unknown_metric_lists_known(self):
        with pytest.raises(ValueError, match="lexical"):
            build_verifier("exact_match")

    def test_registry_entries_satisfy_protocol(self):
        for metric in verifier_names():
            assert isinstance(build_verifier(metric, client=object()), Verifier)
