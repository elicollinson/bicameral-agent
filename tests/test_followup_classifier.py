"""Tests for the FollowUpClassifier."""

import json
import time
from collections import Counter
from pathlib import Path

import pytest

from bicameral_agent.followup_classifier import FollowUpClassifier, FollowUpType
from bicameral_agent.schema import Message

DATASET_PATH = Path(__file__).parent / "fixtures" / "followup_dataset.json"


@pytest.fixture
def classifier():
    return FollowUpClassifier()


@pytest.fixture(scope="module")
def dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


# ── Accuracy tests on the labeled dataset ─────────────────────────────


class TestDatasetAccuracy:
    """Overall and per-class accuracy on the labeled dataset."""

    def test_overall_accuracy_at_least_75_percent(self, classifier, dataset):
        correct = 0
        total = len(dataset)
        for item in dataset:
            history = _build_history(item.get("context"))
            predicted = classifier.classify(item["message"], history)
            if predicted.value == item["label"]:
                correct += 1

        accuracy = correct / total
        assert accuracy >= 0.75, (
            f"Overall accuracy {accuracy:.1%} ({correct}/{total}) is below 75%"
        )

    def test_per_class_recall_at_least_50_percent(self, classifier, dataset):
        by_class: dict[str, dict[str, int]] = {}
        for item in dataset:
            label = item["label"]
            if label not in by_class:
                by_class[label] = {"correct": 0, "total": 0}
            by_class[label]["total"] += 1

            history = _build_history(item.get("context"))
            predicted = classifier.classify(item["message"], history)
            if predicted.value == label:
                by_class[label]["correct"] += 1

        for label, counts in by_class.items():
            recall = counts["correct"] / counts["total"]
            assert recall >= 0.50, (
                f"Recall for {label}: {recall:.1%} "
                f"({counts['correct']}/{counts['total']}) is below 50%"
            )

    def test_confusion_matrix_diagonal_dominant(self, classifier, dataset):
        """The diagonal of the confusion matrix should have the largest
        value in each row (i.e., no class is more often confused with
        another class than correctly classified)."""
        labels = [t.value for t in FollowUpType]
        matrix: dict[str, Counter[str]] = {l: Counter() for l in labels}

        for item in dataset:
            true_label = item["label"]
            history = _build_history(item.get("context"))
            predicted = classifier.classify(item["message"], history)
            matrix[true_label][predicted.value] += 1

        for true_label in labels:
            row = matrix[true_label]
            if row.total() == 0:
                continue
            diagonal = row[true_label]
            for pred_label, count in row.items():
                if pred_label != true_label:
                    assert diagonal >= count, (
                        f"Class '{true_label}' is confused with "
                        f"'{pred_label}' ({count}) more than correctly "
                        f"classified ({diagonal})"
                    )


# ── Edge case tests ───────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self, classifier):
        assert classifier.classify("") == FollowUpType.NEW_TASK

    def test_whitespace_only(self, classifier):
        assert classifier.classify("   \n\t  ") == FollowUpType.NEW_TASK

    def test_very_long_message(self, classifier):
        """10K char message should not crash or timeout."""
        long_msg = "explain more about " * 500  # ~10K chars
        result = classifier.classify(long_msg)
        assert isinstance(result, FollowUpType)

    def test_non_english(self, classifier):
        """Non-English input should return a valid type without crashing."""
        result = classifier.classify("これは日本語のメッセージです")
        assert isinstance(result, FollowUpType)

    def test_unicode_emoji(self, classifier):
        result = classifier.classify("👍 great job!")
        assert isinstance(result, FollowUpType)

    def test_mixed_case(self, classifier):
        result = classifier.classify("NO, THAT'S WRONG")
        assert result == FollowUpType.CORRECTION

    def test_single_word_no(self, classifier):
        result = classifier.classify("No")
        assert result == FollowUpType.CORRECTION

    def test_single_word_yes(self, classifier):
        result = classifier.classify("Yes")
        assert result == FollowUpType.ENCOURAGEMENT


# ── Context-aware tests ──────────────────────────────────────────────


class TestContextAware:
    def test_right_after_question_is_encouragement(self, classifier):
        history = [
            Message(role="assistant", content="Does that make sense?",
                    timestamp_ms=1000, token_count=5),
        ]
        result = classifier.classify("Right", history)
        assert result == FollowUpType.ENCOURAGEMENT

    def test_negation_suppresses_encouragement(self, classifier):
        result = classifier.classify("No, that's not good")
        assert result != FollowUpType.ENCOURAGEMENT


# ── Performance test ─────────────────────────────────────────────────


class TestPerformance:
    def test_classification_under_100ms(self, classifier, dataset):
        """Each classification should take <100ms (averaged over dataset)."""
        # Warm up
        classifier.classify("hello")

        start = time.perf_counter()
        for item in dataset:
            classifier.classify(item["message"])
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / len(dataset)) * 1000
        assert avg_ms < 100, f"Average classification took {avg_ms:.2f}ms (limit 100ms)"


# ── Type index mapping test ──────────────────────────────────────────


class TestTypeIndex:
    def test_all_types_have_index(self):
        for ftype in FollowUpType:
            idx = FollowUpClassifier.type_index(ftype)
            assert 0 <= idx <= 4

    def test_index_matches_encoder_order(self):
        assert FollowUpClassifier.type_index(FollowUpType.CORRECTION) == 0
        assert FollowUpClassifier.type_index(FollowUpType.REDIRECT) == 1
        assert FollowUpClassifier.type_index(FollowUpType.ELABORATION) == 2
        assert FollowUpClassifier.type_index(FollowUpType.NEW_TASK) == 3
        assert FollowUpClassifier.type_index(FollowUpType.ENCOURAGEMENT) == 4


# ── Helpers ───────────────────────────────────────────────────────────


def _build_history(context: list[dict] | None) -> list[Message]:
    """Convert optional context dicts to Message objects."""
    if not context:
        return []
    return [
        Message(
            role=c.get("role", "user"),
            content=c.get("content", ""),
            timestamp_ms=c.get("timestamp_ms", 0),
            token_count=c.get("token_count", 5),
        )
        for c in context
    ]
