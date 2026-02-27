"""Tests for the automated task quality scorer (Issue #12)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.scorer import (
    LexicalScorer,
    TaskScore,
    TaskScorer,
    _lcs_length,
    _rouge_l,
    _token_f1,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HUMAN_SCORES_PATH = Path(__file__).parent / "data" / "human_scores.json"


def _make_task(
    task_id="test_001",
    difficulty=TaskDifficulty.TYPICAL,
    question="What is X?",
    gold_answer="X is the answer.",
    scoring_rubric="5: Perfect. 4: Good. 3: OK. 2: Weak. 1: Wrong.",
) -> ResearchQATask:
    return ResearchQATask(
        task_id=task_id,
        difficulty=difficulty,
        split=TaskSplit.EVAL,
        question=question,
        gold_answer=gold_answer,
        scoring_rubric=scoring_rubric,
    )


def _mock_gemini_response(quality=4, completeness=3, accuracy=5):
    """Create a mock GeminiResponse with structured JSON content."""
    response = MagicMock()
    response.content = json.dumps({
        "quality": quality,
        "completeness": completeness,
        "accuracy": accuracy,
    })
    return response


def _make_scorer_with_mock(response=None, **kwargs):
    """Create a TaskScorer with a mocked GeminiClient."""
    mock_client = MagicMock()
    if response is None:
        response = _mock_gemini_response()
    mock_client.generate.return_value = response
    scorer = TaskScorer(client=mock_client, **kwargs)
    return scorer, mock_client


# ---------------------------------------------------------------------------
# TestTaskScore
# ---------------------------------------------------------------------------


class TestTaskScore:
    def test_from_raw_max(self):
        s = TaskScore.from_raw(5, 5, 5)
        assert s.quality == 1.0
        assert s.completeness == 1.0
        assert s.accuracy == 1.0
        assert s.overall == 1.0

    def test_from_raw_min(self):
        s = TaskScore.from_raw(1, 1, 1)
        assert s.quality == 0.0
        assert s.completeness == 0.0
        assert s.accuracy == 0.0
        assert s.overall == 0.0

    def test_from_raw_mid(self):
        s = TaskScore.from_raw(3, 3, 3)
        assert s.quality == pytest.approx(0.5)
        assert s.overall == pytest.approx(0.5)

    def test_from_raw_clamps_above(self):
        s = TaskScore.from_raw(6, 6, 6)
        assert s.quality == 1.0

    def test_from_raw_clamps_below(self):
        s = TaskScore.from_raw(0, 0, 0)
        assert s.quality == 0.0

    def test_from_raw_overall_is_mean(self):
        s = TaskScore.from_raw(5, 1, 3)
        expected = (1.0 + 0.0 + 0.5) / 3.0
        assert s.overall == pytest.approx(expected)

    def test_from_raw_asymmetric(self):
        s = TaskScore.from_raw(5, 3, 1)
        assert s.quality == 1.0
        assert s.completeness == 0.5
        assert s.accuracy == 0.0

    def test_validation_rejects_above_one(self):
        with pytest.raises(Exception):
            TaskScore(quality=1.1, completeness=0.5, accuracy=0.5, overall=0.5)

    def test_validation_rejects_below_zero(self):
        with pytest.raises(Exception):
            TaskScore(quality=-0.1, completeness=0.5, accuracy=0.5, overall=0.5)

    def test_compatible_with_episode_outcome(self):
        s = TaskScore.from_raw(4, 4, 4)
        assert 0.0 <= s.overall <= 1.0


# ---------------------------------------------------------------------------
# TestTokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert _tokenize("Hello, world! How's it?") == ["hello", "world", "how", "s", "it"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_preserved(self):
        assert _tokenize("HTTP/1.1 200 OK") == ["http", "1", "1", "200", "ok"]

    def test_all_punctuation(self):
        assert _tokenize("...!!!") == []


# ---------------------------------------------------------------------------
# TestTokenF1
# ---------------------------------------------------------------------------


class TestTokenF1:
    def test_identical(self):
        p, r, f1 = _token_f1("the quick brown fox", "the quick brown fox")
        assert f1 == pytest.approx(1.0)

    def test_no_overlap(self):
        p, r, f1 = _token_f1("the quick brown fox", "a lazy red dog")
        assert f1 == pytest.approx(0.0)

    def test_partial_overlap(self):
        p, r, f1 = _token_f1("the quick brown fox", "the slow brown cat")
        assert 0.0 < f1 < 1.0
        assert 0.0 < p < 1.0
        assert 0.0 < r < 1.0

    def test_empty_reference(self):
        p, r, f1 = _token_f1("", "hello")
        assert f1 == 0.0

    def test_empty_hypothesis(self):
        p, r, f1 = _token_f1("hello", "")
        assert f1 == 0.0

    def test_precision_recall_semantics(self):
        # ref = {a, b, c}, hyp = {a, b, d, e} → common = {a, b}
        p, r, f1 = _token_f1("a b c", "a b d e")
        assert p == pytest.approx(2 / 4)  # 2 common / 4 hyp
        assert r == pytest.approx(2 / 3)  # 2 common / 3 ref


# ---------------------------------------------------------------------------
# TestLCSLength
# ---------------------------------------------------------------------------


class TestLCSLength:
    def test_identical(self):
        tokens = ["a", "b", "c"]
        assert _lcs_length(tokens, tokens) == 3

    def test_no_common(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_known_subsequence(self):
        a = ["a", "b", "c", "d", "e"]
        b = ["a", "c", "e"]
        assert _lcs_length(a, b) == 3

    def test_empty(self):
        assert _lcs_length([], ["a"]) == 0
        assert _lcs_length(["a"], []) == 0

    def test_single_match(self):
        assert _lcs_length(["x", "a", "y"], ["b", "a", "c"]) == 1


# ---------------------------------------------------------------------------
# TestRougeL
# ---------------------------------------------------------------------------


class TestRougeL:
    def test_identical(self):
        p, r, f = _rouge_l("the quick brown fox", "the quick brown fox")
        assert f == pytest.approx(1.0)

    def test_no_overlap(self):
        p, r, f = _rouge_l("the quick brown fox", "a lazy red dog")
        assert f == pytest.approx(0.0)

    def test_partial_overlap(self):
        p, r, f = _rouge_l("the cat sat on the mat", "the cat on the mat")
        assert 0.0 < f < 1.0

    def test_empty(self):
        p, r, f = _rouge_l("hello", "")
        assert f == 0.0


# ---------------------------------------------------------------------------
# TestLexicalScorer
# ---------------------------------------------------------------------------


class TestLexicalScorer:
    def setup_method(self):
        self.scorer = LexicalScorer()

    def test_identical_text_scores_one(self):
        task = _make_task(gold_answer="the quick brown fox")
        score = self.scorer.score(task, "the quick brown fox")
        assert score.quality == pytest.approx(1.0)
        assert score.overall == pytest.approx(1.0)

    def test_completely_different_scores_zero(self):
        task = _make_task(gold_answer="the quick brown fox")
        score = self.scorer.score(task, "a lazy red dog")
        assert score.quality == pytest.approx(0.0)
        assert score.overall == pytest.approx(0.0)

    def test_partial_overlap(self):
        task = _make_task(gold_answer="the quick brown fox jumps")
        score = self.scorer.score(task, "the slow brown fox sits")
        assert 0.0 < score.quality < 1.0
        assert 0.0 < score.overall < 1.0

    def test_empty_answer_scores_zero(self):
        task = _make_task(gold_answer="something meaningful")
        score = self.scorer.score(task, "")
        assert score.overall == 0.0

    def test_case_insensitive(self):
        task = _make_task(gold_answer="The Quick Brown Fox")
        s1 = self.scorer.score(task, "the quick brown fox")
        s2 = self.scorer.score(task, "THE QUICK BROWN FOX")
        assert s1.quality == pytest.approx(s2.quality)

    def test_score_fields_in_range(self):
        task = _make_task(gold_answer="answer with several key tokens here")
        score = self.scorer.score(task, "answer with some different tokens")
        assert 0.0 <= score.quality <= 1.0
        assert 0.0 <= score.completeness <= 1.0
        assert 0.0 <= score.accuracy <= 1.0
        assert 0.0 <= score.overall <= 1.0

    def test_score_batch(self):
        tasks = [_make_task(gold_answer="the answer"), _make_task(gold_answer="another")]
        answers = ["the answer", "something else"]
        results = self.scorer.score_batch(tasks, answers)
        assert len(results) == 2
        assert results[0].quality == pytest.approx(1.0)
        assert results[1].quality < 1.0

    def test_score_batch_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            self.scorer.score_batch([_make_task()], ["a", "b"])


# ---------------------------------------------------------------------------
# TestTaskScorerUnit
# ---------------------------------------------------------------------------


class TestTaskScorerUnit:
    def test_score_calls_gemini(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "my answer")
        mock_client.generate.assert_called_once()

    def test_score_uses_temperature_zero(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "my answer")
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0

    def test_score_uses_minimal_thinking(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "my answer")
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["thinking_level"] == "minimal"

    def test_score_uses_response_schema(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "my answer")
        call_kwargs = mock_client.generate.call_args
        schema = call_kwargs.kwargs["response_schema"]
        assert "quality" in schema["properties"]
        assert "completeness" in schema["properties"]
        assert "accuracy" in schema["properties"]

    def test_score_prompt_contains_task_fields(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task(
            question="What is photosynthesis?",
            gold_answer="Plants convert sunlight to energy.",
            scoring_rubric="5: Perfect. 1: Wrong.",
        )
        scorer.score(task, "It makes food for plants")
        # First positional arg to generate() is the messages list
        messages = mock_client.generate.call_args[0][0]
        user_msg = messages[0]["content"]
        assert "What is photosynthesis?" in user_msg
        assert "Plants convert sunlight to energy." in user_msg
        assert "5: Perfect. 1: Wrong." in user_msg
        assert "It makes food for plants" in user_msg

    def test_score_system_prompt_set(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "answer")
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["system_prompt"] is not None
        assert "evaluator" in call_kwargs.kwargs["system_prompt"].lower()

    def test_score_parses_response(self):
        scorer, _ = _make_scorer_with_mock(
            response=_mock_gemini_response(quality=5, completeness=3, accuracy=4)
        )
        task = _make_task()
        result = scorer.score(task, "my answer")
        assert result.quality == pytest.approx(1.0)
        assert result.completeness == pytest.approx(0.5)
        assert result.accuracy == pytest.approx(0.75)

    def test_score_returns_normalized_values(self):
        scorer, _ = _make_scorer_with_mock(
            response=_mock_gemini_response(quality=1, completeness=1, accuracy=1)
        )
        task = _make_task()
        result = scorer.score(task, "bad answer")
        assert result.quality == 0.0
        assert result.overall == 0.0


# ---------------------------------------------------------------------------
# TestCache
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_hit_avoids_llm_call(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "same answer")
        scorer.score(task, "same answer")
        assert mock_client.generate.call_count == 1

    def test_cache_miss_calls_llm(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "answer one")
        scorer.score(task, "answer two")
        assert mock_client.generate.call_count == 2

    def test_different_tasks_same_answer(self):
        scorer, mock_client = _make_scorer_with_mock()
        task1 = _make_task(task_id="task_001")
        task2 = _make_task(task_id="task_002")
        scorer.score(task1, "same answer")
        scorer.score(task2, "same answer")
        assert mock_client.generate.call_count == 2

    def test_cache_size(self):
        scorer, _ = _make_scorer_with_mock()
        assert scorer.cache_size == 0
        scorer.score(_make_task(), "answer")
        assert scorer.cache_size == 1

    def test_clear_cache(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "answer")
        assert scorer.cache_size == 1
        scorer.clear_cache()
        assert scorer.cache_size == 0
        scorer.score(task, "answer")
        assert mock_client.generate.call_count == 2

    def test_cached_result_identical(self):
        scorer, _ = _make_scorer_with_mock(
            response=_mock_gemini_response(quality=4, completeness=3, accuracy=5)
        )
        task = _make_task()
        r1 = scorer.score(task, "answer")
        r2 = scorer.score(task, "answer")
        assert r1 == r2


# ---------------------------------------------------------------------------
# TestBatchScoring
# ---------------------------------------------------------------------------


class TestBatchScoring:
    def test_batch_returns_correct_count(self):
        scorer, _ = _make_scorer_with_mock()
        tasks = [_make_task(task_id=f"t_{i}") for i in range(5)]
        answers = [f"answer {i}" for i in range(5)]
        results = scorer.score_batch(tasks, answers)
        assert len(results) == 5

    def test_batch_length_mismatch(self):
        scorer, _ = _make_scorer_with_mock()
        with pytest.raises(ValueError, match="Length mismatch"):
            scorer.score_batch([_make_task()], ["a", "b"])

    def test_batch_uses_cache(self):
        scorer, mock_client = _make_scorer_with_mock()
        task = _make_task()
        scorer.score(task, "pre-cached answer")
        assert mock_client.generate.call_count == 1

        results = scorer.score_batch([task], ["pre-cached answer"])
        assert mock_client.generate.call_count == 1  # no new call
        assert len(results) == 1

    def test_batch_preserves_order(self):
        responses = [
            _mock_gemini_response(quality=q, completeness=c, accuracy=a)
            for q, c, a in [(5, 5, 5), (1, 1, 1), (3, 3, 3)]
        ]
        mock_client = MagicMock()
        mock_client.generate.side_effect = responses
        scorer = TaskScorer(client=mock_client, max_workers=1)

        tasks = [_make_task(task_id=f"t_{i}") for i in range(3)]
        answers = [f"answer {i}" for i in range(3)]
        results = scorer.score_batch(tasks, answers)

        # With max_workers=1, calls are sequential
        assert results[0].quality == pytest.approx(1.0)
        assert results[1].quality == pytest.approx(0.0)
        assert results[2].quality == pytest.approx(0.5)

    def test_batch_empty(self):
        scorer, _ = _make_scorer_with_mock()
        results = scorer.score_batch([], [])
        assert results == []


# ---------------------------------------------------------------------------
# TestIntegration (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestIntegration:
    def test_real_score_returns_valid_range(self):
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task = dataset.tasks[0]
        scorer = TaskScorer()
        result = scorer.score(task, task.gold_answer)
        assert 0.0 <= result.quality <= 1.0
        assert 0.0 <= result.completeness <= 1.0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.overall <= 1.0

    def test_determinism(self):
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task = dataset.tasks[0]
        scorer = TaskScorer()
        r1 = scorer.score(task, "A vague answer about the topic.")
        scorer.clear_cache()
        r2 = scorer.score(task, "A vague answer about the topic.")
        assert r1.quality == r2.quality
        assert r1.completeness == r2.completeness
        assert r1.accuracy == r2.accuracy

    def test_good_answer_scores_higher(self):
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task = dataset.tasks[0]
        scorer = TaskScorer()
        good = scorer.score(task, task.gold_answer)
        bad = scorer.score(task, "I don't know.")
        assert good.overall > bad.overall


# ---------------------------------------------------------------------------
# TestCorrelation (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestCorrelation:
    @pytest.fixture
    def human_data(self):
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        task_map = {t.task_id: t for t in dataset.tasks}
        raw = json.loads(_HUMAN_SCORES_PATH.read_text())
        entries = []
        for entry in raw:
            task = task_map[entry["task_id"]]
            entries.append((task, entry["agent_answer"], entry["human_scores"]))
        return entries

    def test_llm_scorer_correlation(self, human_data):
        """AC: LLM judge overall scores correlate with human at r > 0.7."""
        scorer = TaskScorer()
        llm_scores = []
        human_scores = []
        for task, answer, hs in human_data:
            result = scorer.score(task, answer)
            llm_scores.append(result.overall)
            human_scores.append((hs["overall"] - 1) / 4.0)

        r = np.corrcoef(llm_scores, human_scores)[0, 1]
        assert r > 0.7, f"LLM scorer correlation {r:.3f} below threshold 0.7"

    def test_lexical_scorer_correlation(self, human_data):
        """AC: Lexical baseline scores correlate with human at r > 0.4."""
        scorer = LexicalScorer()
        lex_scores = []
        human_scores = []
        for task, answer, hs in human_data:
            result = scorer.score(task, answer)
            lex_scores.append(result.overall)
            human_scores.append((hs["overall"] - 1) / 4.0)

        r = np.corrcoef(lex_scores, human_scores)[0, 1]
        assert r > 0.4, f"Lexical scorer correlation {r:.3f} below threshold 0.4"
