"""Tests for the verifier registry (Issue #56). No LLM calls."""

import json
from unittest.mock import MagicMock

import pytest

from bicameral_agent.dataset import (
    ResearchQATask,
    RubricItem,
    TaskDifficulty,
    TaskSplit,
)
from bicameral_agent.scorer import LexicalScorer, TaskScorer
from bicameral_agent.verifiers import (
    AbstentionVerifier,
    ExactMatchVerifier,
    MultipleChoiceVerifier,
    RubricCoverageVerifier,
    Verifier,
    build_verifier,
    normalize_answer,
    verifier_names,
)


def _task(**overrides) -> ResearchQATask:
    defaults = dict(
        task_id="t1",
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question="What color is the sky?",
        gold_answer="The sky is blue.",
        scoring_rubric="5: correct. 1: incorrect.",
    )
    defaults.update(overrides)
    return ResearchQATask(**defaults)


def _mock_client(payload: dict) -> MagicMock:
    """A model client whose generate() returns the given JSON payload."""
    response = MagicMock()
    response.content = json.dumps(payload)
    client = MagicMock()
    client.generate.return_value = response
    return client


class TestBuildVerifier:
    def test_known_metrics(self):
        assert verifier_names() == [
            "abstention",
            "exact_match",
            "lexical",
            "llm_judge",
            "multiple_choice",
            "rubric_coverage",
        ]

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

    def test_deterministic_verifiers_dispatch(self):
        assert isinstance(build_verifier("exact_match"), ExactMatchVerifier)
        assert isinstance(build_verifier("multiple_choice"), MultipleChoiceVerifier)

    def test_llm_backed_verifiers_receive_client(self):
        sentinel = object()
        assert build_verifier("rubric_coverage", client=sentinel)._client is sentinel
        assert build_verifier("abstention", client=sentinel)._client is sentinel

    def test_unknown_metric_lists_known(self):
        with pytest.raises(ValueError, match="lexical"):
            build_verifier("nope")

    def test_registry_entries_satisfy_protocol(self):
        for metric in verifier_names():
            assert isinstance(build_verifier(metric, client=object()), Verifier)


class TestNormalizeAnswer:
    def test_case_punctuation_articles(self):
        assert normalize_answer("The Sky, is BLUE!") == "sky is blue"

    def test_whitespace_collapsed(self):
        assert normalize_answer("  a   b  \n c ") == "b c"


class TestExactMatchVerifier:
    def test_exact_match_scores_one(self):
        score = ExactMatchVerifier().score(_task(gold_answer="Paris"), "Paris")
        assert score.overall == 1.0
        assert "match" in score.detail

    def test_normalized_match(self):
        score = ExactMatchVerifier().score(_task(gold_answer="Paris"), "the PARIS.")
        assert score.overall == 1.0

    def test_mismatch_scores_zero_with_detail(self):
        score = ExactMatchVerifier().score(_task(gold_answer="Paris"), "London")
        assert score.overall == 0.0
        assert "'Paris'" in score.detail
        assert "'London'" in score.detail

    def test_final_answer_extraction(self):
        answer = "Let me reason about this.\nThe answer is Paris."
        score = ExactMatchVerifier().score(_task(gold_answer="Paris"), answer)
        assert score.overall == 1.0

    def test_numeric_equivalence(self):
        score = ExactMatchVerifier().score(_task(gold_answer="1,000"), "1000.0")
        assert score.overall == 1.0

    def test_verbose_non_answer_scores_zero(self):
        score = ExactMatchVerifier().score(
            _task(gold_answer="Paris"), "It could be Paris or London."
        )
        assert score.overall == 0.0


class TestMultipleChoiceVerifier:
    def _mc_task(self, gold="B") -> ResearchQATask:
        return _task(
            gold_answer=gold,
            choices=["red herring", "correct choice", "another option"],
        )

    def test_stated_answer_correct(self):
        score = MultipleChoiceVerifier().score(
            self._mc_task(), "Reasoning... The answer is B."
        )
        assert score.overall == 1.0
        assert "'B'" in score.detail

    def test_stated_answer_wrong(self):
        score = MultipleChoiceVerifier().score(self._mc_task(), "Answer: (C)")
        assert score.overall == 0.0
        assert "extracted 'C'" in score.detail
        assert "expected 'B'" in score.detail

    def test_bare_letter(self):
        assert MultipleChoiceVerifier().score(self._mc_task(), "B.").overall == 1.0

    def test_last_stated_answer_wins(self):
        answer = "Initially the answer is A. On reflection, the answer is B."
        assert MultipleChoiceVerifier().score(self._mc_task(), answer).overall == 1.0

    def test_choice_text_match(self):
        score = MultipleChoiceVerifier().score(
            self._mc_task(), "I would go with the correct choice here."
        )
        assert score.overall == 1.0

    def test_no_extractable_choice_scores_zero(self):
        score = MultipleChoiceVerifier().score(
            self._mc_task(), "This is hard to say."
        )
        assert score.overall == 0.0
        assert "None" in score.detail


class TestRubricCoverageVerifier:
    def _rubric_task(self) -> ResearchQATask:
        return _task(
            gold_answer="",
            rubric_items=[
                RubricItem(criterion="mentions blue", points=3.0),
                RubricItem(criterion="mentions scattering", points=1.0),
                RubricItem(criterion="claims the sky is green", points=-2.0),
            ],
        )

    def test_weighted_coverage(self):
        client = _mock_client({"met_indices": [1]})
        score = RubricCoverageVerifier(client=client).score(self._rubric_task(), "blue")
        assert score.overall == pytest.approx(3.0 / 4.0)
        assert "1/3 criteria" in score.detail

    def test_negative_points_subtract_and_clamp(self):
        client = _mock_client({"met_indices": [3]})
        score = RubricCoverageVerifier(client=client).score(self._rubric_task(), "green")
        assert score.overall == 0.0

    def test_full_coverage_clamped_to_one(self):
        client = _mock_client({"met_indices": [1, 2]})
        score = RubricCoverageVerifier(client=client).score(
            self._rubric_task(), "blue scattering"
        )
        assert score.overall == 1.0

    def test_out_of_range_indices_ignored(self):
        client = _mock_client({"met_indices": [0, 2, 99]})
        score = RubricCoverageVerifier(client=client).score(self._rubric_task(), "x")
        assert score.overall == pytest.approx(1.0 / 4.0)

    def test_malformed_judge_output_scores_zero(self):
        response = MagicMock()
        response.content = "not json"
        client = MagicMock()
        client.generate.return_value = response
        score = RubricCoverageVerifier(client=client).score(self._rubric_task(), "x")
        assert score.overall == 0.0

    def test_task_without_rubric_items_rejected(self):
        with pytest.raises(ValueError, match="rubric_items"):
            RubricCoverageVerifier(client=MagicMock()).score(_task(), "x")

    def test_criteria_reach_the_judge(self):
        client = _mock_client({"met_indices": []})
        RubricCoverageVerifier(client=client).score(self._rubric_task(), "x")
        prompt = client.generate.call_args[0][0][0]["content"]
        assert "mentions blue" in prompt
        assert "mentions scattering" in prompt


class TestAbstentionVerifier:
    def test_correct_abstention(self):
        client = _mock_client({"abstained": True})
        score = AbstentionVerifier(client=client).score(
            _task(abstention_expected=True), "I cannot answer that."
        )
        assert score.overall == 1.0
        assert "abstained=True" in score.detail

    def test_missed_abstention(self):
        client = _mock_client({"abstained": False})
        score = AbstentionVerifier(client=client).score(
            _task(abstention_expected=True), "It is definitely 42."
        )
        assert score.overall == 0.0

    def test_unwarranted_abstention(self):
        client = _mock_client({"abstained": True})
        score = AbstentionVerifier(client=client).score(
            _task(abstention_expected=False), "I do not know."
        )
        assert score.overall == 0.0

    def test_answering_when_expected(self):
        client = _mock_client({"abstained": False})
        score = AbstentionVerifier(client=client).score(
            _task(abstention_expected=False), "The sky is blue."
        )
        assert score.overall == 1.0
