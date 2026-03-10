"""Tests for the Research Gap Scanner tool primitive."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from bicameral_agent.gemini import GeminiResponse
from bicameral_agent.queue import Priority
from bicameral_agent.schema import Message
from bicameral_agent.tool_primitive import TokenBudget
from bicameral_agent.gap_scanner import (
    GapCategory,
    IdentifiedGap,
    MockSearchProvider,
    ResearchGapScanner,
    SearchResult,
    _format_conversation,
    _make_dedup_key,
    _max_priority,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BUDGET = TokenBudget(max_calls=3, max_input_tokens=5000, max_output_tokens=2000)


def _make_messages(content: str = "Tell me about superconductivity research") -> list[Message]:
    return [
        Message(role="user", content=content, timestamp_ms=1000, token_count=10),
        Message(
            role="assistant",
            content="Superconductivity is a phenomenon where materials conduct electricity with zero resistance. Recent claims about room-temperature superconductors have generated excitement.",
            timestamp_ms=2000,
            token_count=30,
        ),
        Message(
            role="user",
            content="What about LK-99? I heard it was confirmed as a superconductor.",
            timestamp_ms=3000,
            token_count=15,
        ),
    ]


def _make_state():
    return np.zeros(53, dtype=np.float32)


def _fake_response(content: str, input_tokens: int = 50, output_tokens: int = 100) -> GeminiResponse:
    return GeminiResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=50.0,
        finish_reason="STOP",
    )


def _mock_client(responses: list[GeminiResponse] | GeminiResponse | None = None) -> MagicMock:
    client = MagicMock(spec=["generate"])
    if isinstance(responses, list):
        client.generate.side_effect = responses
    else:
        client.generate.return_value = responses or _fake_response("{}")
    return client


def _gap_response(
    has_gaps: bool = True,
    gaps: list[dict] | None = None,
) -> GeminiResponse:
    """Create a fake LLM response for gap identification."""
    if gaps is None and has_gaps:
        gaps = [
            {
                "description": "LK-99 superconductor claim lacks evidence",
                "category": "core_claim",
                "search_query": "LK-99 room temperature superconductor replication results",
            },
            {
                "description": "No specific temperature data cited",
                "category": "supplementary",
                "search_query": "high temperature superconductor critical temperature records",
            },
        ]
    payload = {"has_gaps": has_gaps, "gaps": gaps or []}
    return _fake_response(json.dumps(payload))


def _ranking_response(
    overall_confidence: float = 0.75,
    ranked_results: list[dict] | None = None,
) -> GeminiResponse:
    """Create a fake LLM response for relevance ranking."""
    if ranked_results is None:
        ranked_results = [
            {
                "gap_description": "LK-99 superconductor claim lacks evidence",
                "title": "Room-temperature superconductor claims",
                "snippet": "LK-99 claims were not replicated.",
                "relevance_score": 0.9,
                "source": "nature:2023.lk99",
            },
        ]
    payload = {
        "overall_confidence": overall_confidence,
        "ranked_results": ranked_results,
    }
    return _fake_response(json.dumps(payload))


# ---------------------------------------------------------------------------
# Unit tests: Gap identification
# ---------------------------------------------------------------------------


class TestGapIdentification:
    def test_parses_structured_json(self):
        scanner = ResearchGapScanner()
        client = _mock_client([_gap_response(), _ranking_response()])
        result = scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.metadata.items_found >= 1
        assert result.queue_deposit is not None

    def test_no_gaps_returns_none_deposit(self):
        scanner = ResearchGapScanner()
        client = _mock_client(_gap_response(has_gaps=False))
        result = scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is None
        assert result.metadata.items_found == 0
        assert result.metadata.confidence == 0.8

    def test_categories_map_to_priorities(self):
        gaps_data = [
            {"description": "core gap", "category": "core_claim", "search_query": "q1"},
        ]
        scanner = ResearchGapScanner()
        client = _mock_client([
            _gap_response(gaps=gaps_data),
            _ranking_response(),
        ])
        result = scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.queue_deposit.priority == Priority.HIGH


# ---------------------------------------------------------------------------
# Unit tests: Budget compliance
# ---------------------------------------------------------------------------


class TestBudgetCompliance:
    def test_at_most_two_llm_calls(self):
        scanner = ResearchGapScanner()
        client = _mock_client([_gap_response(), _ranking_response()])
        scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert client.generate.call_count <= 2

    def test_no_gaps_only_one_call(self):
        scanner = ResearchGapScanner()
        client = _mock_client(_gap_response(has_gaps=False))
        scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert client.generate.call_count == 1


# ---------------------------------------------------------------------------
# Unit tests: Dedup key
# ---------------------------------------------------------------------------


class TestDedupKey:
    def test_same_gaps_same_key(self):
        gaps_a = [
            IdentifiedGap("gap1", GapCategory.CORE_CLAIM, "q1"),
            IdentifiedGap("gap2", GapCategory.SUPPLEMENTARY, "q2"),
        ]
        gaps_b = [
            IdentifiedGap("gap2", GapCategory.SUPPLEMENTARY, "q2"),
            IdentifiedGap("gap1", GapCategory.CORE_CLAIM, "q1"),
        ]
        assert _make_dedup_key(gaps_a) == _make_dedup_key(gaps_b)

    def test_different_gaps_different_key(self):
        gaps_a = [IdentifiedGap("gap1", GapCategory.CORE_CLAIM, "q1")]
        gaps_b = [IdentifiedGap("gap2", GapCategory.CORE_CLAIM, "q1")]
        assert _make_dedup_key(gaps_a) != _make_dedup_key(gaps_b)

    def test_key_has_prefix(self):
        gaps = [IdentifiedGap("gap1", GapCategory.CORE_CLAIM, "q1")]
        assert _make_dedup_key(gaps).startswith("gap_scanner:")


# ---------------------------------------------------------------------------
# Unit tests: Metadata fields
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_all_fields_populated(self):
        scanner = ResearchGapScanner()
        client = _mock_client([_gap_response(), _ranking_response()])
        result = scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        meta = result.metadata
        assert meta.tool_id == "research_gap_scanner"
        assert meta.action_taken != ""
        assert 0.0 <= meta.confidence <= 1.0
        assert meta.items_found >= 0
        assert 0.0 <= meta.estimated_relevance <= 1.0
        assert meta.tokens_consumed >= 0
        assert meta.execution_duration_ms >= 0


# ---------------------------------------------------------------------------
# Unit tests: Priority mapping
# ---------------------------------------------------------------------------


class TestPriorityMapping:
    def test_core_claim_maps_to_high(self):
        gaps = [IdentifiedGap("g", GapCategory.CORE_CLAIM, "q")]
        assert _max_priority(gaps) == Priority.HIGH

    def test_supplementary_maps_to_medium(self):
        gaps = [IdentifiedGap("g", GapCategory.SUPPLEMENTARY, "q")]
        assert _max_priority(gaps) == Priority.MEDIUM

    def test_nice_to_have_maps_to_low(self):
        gaps = [IdentifiedGap("g", GapCategory.NICE_TO_HAVE, "q")]
        assert _max_priority(gaps) == Priority.LOW

    def test_mixed_returns_highest(self):
        gaps = [
            IdentifiedGap("g1", GapCategory.NICE_TO_HAVE, "q1"),
            IdentifiedGap("g2", GapCategory.CORE_CLAIM, "q2"),
            IdentifiedGap("g3", GapCategory.SUPPLEMENTARY, "q3"),
        ]
        assert _max_priority(gaps) == Priority.HIGH


# ---------------------------------------------------------------------------
# Unit tests: MockSearchProvider
# ---------------------------------------------------------------------------


class TestMockSearchProvider:
    def test_keyword_matching_finds_results(self):
        provider = MockSearchProvider()
        results = provider.search("superconductivity high temperature hydrides")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_empty_for_nonsense(self):
        provider = MockSearchProvider()
        results = provider.search("xyzzy plugh twisty")
        assert results == []

    def test_max_results_respected(self):
        provider = MockSearchProvider()
        results = provider.search("superconductor fusion therapy research", max_results=2)
        assert len(results) <= 2

    def test_results_sorted_by_relevance(self):
        provider = MockSearchProvider()
        results = provider.search("CAR-T cell therapy cancer treatment")
        if len(results) > 1:
            scores = [r.relevance_score for r in results]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Unit tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_no_gaps_without_gap_tasks(self):
        """Tasks without known_gaps should mostly return None queue_deposit."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        no_gap_tasks = dataset.without_gaps()[:15]

        none_count = 0
        for task in no_gap_tasks:
            scanner = ResearchGapScanner()
            # Mock LLM to return no gaps for these tasks
            client = _mock_client(_gap_response(has_gaps=False))
            messages = [
                Message(role="user", content=task.question, timestamp_ms=1000, token_count=10),
                Message(role="assistant", content=task.gold_answer, timestamp_ms=2000, token_count=50),
            ]
            result = scanner.execute(messages, _make_state(), _DEFAULT_BUDGET, client)
            if result.queue_deposit is None:
                none_count += 1

        assert none_count >= 12, f"Expected >=12/15 None deposits, got {none_count}"


# ---------------------------------------------------------------------------
# Unit tests: Custom SearchProvider
# ---------------------------------------------------------------------------


class TestCustomSearchProvider:
    def test_custom_provider_via_protocol(self):
        class FakeProvider:
            def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
                return [
                    SearchResult(
                        title="Custom result",
                        snippet="From custom provider",
                        relevance_score=0.95,
                        source="custom:1",
                    )
                ]

        scanner = ResearchGapScanner(search_provider=FakeProvider())
        client = _mock_client([_gap_response(), _ranking_response()])
        result = scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None


# ---------------------------------------------------------------------------
# Unit tests: Conversation formatting
# ---------------------------------------------------------------------------


class TestConversationFormatting:
    def test_format_limits_to_10(self):
        messages = [
            Message(role="user", content=f"msg{i}", timestamp_ms=i * 1000, token_count=5)
            for i in range(15)
        ]
        formatted = _format_conversation(messages)
        lines = formatted.strip().split("\n")
        assert len(lines) == 10

    def test_format_includes_role(self):
        messages = _make_messages()
        formatted = _format_conversation(messages)
        assert "[user]:" in formatted
        assert "[assistant]:" in formatted


# ---------------------------------------------------------------------------
# Unit tests: Gaps found but no search results
# ---------------------------------------------------------------------------


class TestGapsNoResults:
    def test_gaps_no_search_results(self):
        """When gaps are found but search returns nothing."""

        class EmptyProvider:
            def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
                return []

        scanner = ResearchGapScanner(search_provider=EmptyProvider())
        client = _mock_client(_gap_response())
        result = scanner.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.metadata.confidence == 0.5
        assert result.metadata.estimated_relevance == 0.3
        # Only 1 LLM call (no ranking needed)
        assert client.generate.call_count == 1


# ---------------------------------------------------------------------------
# Integration tests (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestIntegration:
    def test_gap_identification_on_dataset(self):
        """Run against tasks with known gaps, verify >=70% gap identification."""
        from bicameral_agent.dataset import ResearchQADataset
        from bicameral_agent.gemini import GeminiClient
        from bicameral_agent.gap_scanner import _tokenize

        dataset = ResearchQADataset()
        gap_tasks = dataset.with_gaps()[:15]
        client = GeminiClient()
        budget = TokenBudget(max_calls=3, max_input_tokens=5000, max_output_tokens=2000)

        identified_count = 0
        for task in gap_tasks:
            scanner = ResearchGapScanner()
            messages = [
                Message(role="user", content=task.question, timestamp_ms=1000, token_count=10),
                Message(
                    role="assistant",
                    content=task.gold_answer[:200],  # truncate to leave gaps
                    timestamp_ms=2000,
                    token_count=50,
                ),
            ]
            result = scanner.execute(messages, _make_state(), budget, client)

            if result.queue_deposit is not None:
                # Fuzzy check: do gap descriptions overlap with known_gaps?
                deposit_tokens = set(_tokenize(result.queue_deposit.content))
                for known_gap in task.known_gaps or []:
                    gap_tokens = set(_tokenize(known_gap))
                    overlap = len(deposit_tokens & gap_tokens) / max(len(gap_tokens), 1)
                    if overlap > 0.2:
                        identified_count += 1
                        break

        rate = identified_count / len(gap_tasks)
        assert rate >= 0.7, f"Gap identification rate {rate:.0%} < 70%"

    def test_budget_and_latency(self):
        """Verify budget (<=3 calls, <5s) on real calls."""
        import time

        from bicameral_agent.gemini import GeminiClient

        scanner = ResearchGapScanner()
        client = GeminiClient()
        budget = TokenBudget(max_calls=3, max_input_tokens=5000, max_output_tokens=2000)

        start = time.monotonic()
        result = scanner.execute(_make_messages(), _make_state(), budget, client)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Execution took {elapsed:.1f}s, expected <5s"
        assert result.metadata.tokens_consumed > 0
