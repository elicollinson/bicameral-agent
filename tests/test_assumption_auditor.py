"""Tests for the Assumption Auditor tool primitive."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from bicameral_agent.gemini import GeminiResponse
from bicameral_agent.queue import Priority
from bicameral_agent.schema import Message
from bicameral_agent.tool_primitive import BudgetExceededError, TokenBudget
from bicameral_agent.assumption_auditor import (
    AssumptionAuditor,
    EvidenceVerdict,
    IdentifiedAssumption,
    RiskLevel,
    _compute_priority,
    _make_dedup_key,
)
from bicameral_agent.gap_scanner import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BUDGET = TokenBudget(max_calls=3, max_input_tokens=5000, max_output_tokens=2000)


def _make_messages(content: str = "LK-99 is a room-temperature superconductor") -> list[Message]:
    return [
        Message(role="user", content="Tell me about LK-99", timestamp_ms=1000, token_count=10),
        Message(
            role="assistant",
            content=content,
            timestamp_ms=2000,
            token_count=30,
        ),
        Message(
            role="user",
            content="So it's definitely confirmed then?",
            timestamp_ms=3000,
            token_count=10,
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


def _assumption_response(
    assumptions: list[dict] | None = None,
    all_safe: bool = False,
) -> GeminiResponse:
    """Create a fake LLM response for assumption extraction."""
    if assumptions is None:
        if all_safe:
            assumptions = [
                {
                    "description": "Physics is well-understood",
                    "risk_level": "safe",
                    "basis": "Established science",
                },
            ]
        else:
            assumptions = [
                {
                    "description": "LK-99 is a confirmed superconductor",
                    "risk_level": "high",
                    "basis": "Unverified claim treated as fact",
                    "search_query": "LK-99 superconductor replication results",
                },
                {
                    "description": "Room-temperature superconductivity is achievable",
                    "risk_level": "moderate",
                    "basis": "Plausible but unproven in general",
                },
            ]
    payload = {"assumptions": assumptions}
    return _fake_response(json.dumps(payload))


def _evidence_response(
    assessments: list[dict] | None = None,
) -> GeminiResponse:
    """Create a fake LLM response for evidence assessment."""
    if assessments is None:
        assessments = [
            {
                "assumption_description": "LK-99 is a confirmed superconductor",
                "verdict": "contradicting",
                "evidence_summary": "Independent labs failed to replicate LK-99 claims",
                "suggested_action": "revise",
                "source": "nature:2023.lk99",
            },
        ]
    payload = {"assessments": assessments}
    return _fake_response(json.dumps(payload))


# ---------------------------------------------------------------------------
# TestAssumptionExtraction
# ---------------------------------------------------------------------------


class TestAssumptionExtraction:
    def test_parses_structured_json(self):
        auditor = AssumptionAuditor()
        client = _mock_client([_assumption_response(), _evidence_response()])
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.metadata.items_found >= 1
        assert result.queue_deposit is not None

    def test_all_safe_returns_none_deposit(self):
        auditor = AssumptionAuditor()
        client = _mock_client(_assumption_response(all_safe=True))
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is None
        assert result.metadata.items_found == 0
        assert result.metadata.confidence == 0.8

    def test_high_risk_produces_deposit(self):
        assumptions = [
            {
                "description": "Unverified claim",
                "risk_level": "high",
                "basis": "No evidence provided",
                "search_query": "verify claim",
            },
        ]
        auditor = AssumptionAuditor()
        client = _mock_client([
            _assumption_response(assumptions=assumptions),
            _evidence_response(),
        ])
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None

    def test_moderate_risk_produces_medium_priority(self):
        assumptions = [
            {
                "description": "Plausible but untested",
                "risk_level": "moderate",
                "basis": "Reasonable but not verified",
            },
        ]
        auditor = AssumptionAuditor()
        client = _mock_client(_assumption_response(assumptions=assumptions))
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.queue_deposit.priority == Priority.MEDIUM


# ---------------------------------------------------------------------------
# TestPriorityMapping
# ---------------------------------------------------------------------------


class TestPriorityMapping:
    def test_high_contradicting_is_critical(self):
        assert _compute_priority(RiskLevel.HIGH, EvidenceVerdict.CONTRADICTING) == Priority.CRITICAL

    def test_high_no_evidence_is_high(self):
        assert _compute_priority(RiskLevel.HIGH, None) == Priority.HIGH

    def test_high_inconclusive_is_high(self):
        assert _compute_priority(RiskLevel.HIGH, EvidenceVerdict.INCONCLUSIVE) == Priority.HIGH

    def test_high_supporting_is_medium(self):
        assert _compute_priority(RiskLevel.HIGH, EvidenceVerdict.SUPPORTING) == Priority.MEDIUM

    def test_moderate_is_medium(self):
        assert _compute_priority(RiskLevel.MODERATE) == Priority.MEDIUM

    def test_safe_is_low(self):
        assert _compute_priority(RiskLevel.SAFE) == Priority.LOW


# ---------------------------------------------------------------------------
# TestBudgetCompliance
# ---------------------------------------------------------------------------


class TestBudgetCompliance:
    def test_at_most_two_llm_calls(self):
        auditor = AssumptionAuditor()
        client = _mock_client([_assumption_response(), _evidence_response()])
        auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert client.generate.call_count <= 2

    def test_all_safe_only_one_call(self):
        auditor = AssumptionAuditor()
        client = _mock_client(_assumption_response(all_safe=True))
        auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert client.generate.call_count == 1

    def test_budget_exceeded_fallback(self):
        """When 2nd LLM call raises BudgetExceededError, still returns result."""
        auditor = AssumptionAuditor()
        client = _mock_client([
            _assumption_response(),
        ])
        # Second call raises BudgetExceededError
        client.generate.side_effect = [
            _assumption_response(),
            BudgetExceededError("budget exceeded"),
        ]
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        # Should still produce a deposit (without evidence)
        assert result.queue_deposit is not None
        assert result.metadata.items_found >= 1


# ---------------------------------------------------------------------------
# TestSuggestedActions
# ---------------------------------------------------------------------------


class TestSuggestedActions:
    def test_contradicting_suggests_revise(self):
        assessments = [
            {
                "assumption_description": "test assumption",
                "verdict": "contradicting",
                "evidence_summary": "Evidence refutes this",
                "suggested_action": "revise",
                "source": "test:1",
            },
        ]
        assumptions = [
            {
                "description": "test assumption",
                "risk_level": "high",
                "basis": "unverified",
                "search_query": "test query",
            },
        ]

        class AlwaysFindsProvider:
            def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
                return [SearchResult(title="r", snippet="s", relevance_score=0.9, source="s:1")]

        auditor = AssumptionAuditor(search_provider=AlwaysFindsProvider())
        client = _mock_client([
            _assumption_response(assumptions=assumptions),
            _evidence_response(assessments=assessments),
        ])
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert "revise" in result.queue_deposit.content.lower()

    def test_moderate_defaults_to_validate(self):
        assumptions = [
            {
                "description": "moderate claim",
                "risk_level": "moderate",
                "basis": "Plausible",
            },
        ]
        auditor = AssumptionAuditor()
        client = _mock_client(_assumption_response(assumptions=assumptions))
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert "validate" in result.queue_deposit.content.lower()


# ---------------------------------------------------------------------------
# TestDedupKey
# ---------------------------------------------------------------------------


class TestDedupKey:
    def test_same_assumptions_same_key(self):
        a = [
            IdentifiedAssumption("a1", RiskLevel.HIGH, "b", "q"),
            IdentifiedAssumption("a2", RiskLevel.MODERATE, "b", None),
        ]
        b = [
            IdentifiedAssumption("a2", RiskLevel.MODERATE, "b", None),
            IdentifiedAssumption("a1", RiskLevel.HIGH, "b", "q"),
        ]
        assert _make_dedup_key(a) == _make_dedup_key(b)

    def test_different_assumptions_different_key(self):
        a = [IdentifiedAssumption("a1", RiskLevel.HIGH, "b", "q")]
        b = [IdentifiedAssumption("a2", RiskLevel.HIGH, "b", "q")]
        assert _make_dedup_key(a) != _make_dedup_key(b)

    def test_key_has_prefix(self):
        a = [IdentifiedAssumption("a1", RiskLevel.HIGH, "b", "q")]
        assert _make_dedup_key(a).startswith("assumption_auditor:")


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_all_fields_populated(self):
        auditor = AssumptionAuditor()
        client = _mock_client([_assumption_response(), _evidence_response()])
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        meta = result.metadata
        assert meta.tool_id == "assumption_auditor"
        assert meta.action_taken != ""
        assert 0.0 <= meta.confidence <= 1.0
        assert meta.items_found >= 0
        assert 0.0 <= meta.estimated_relevance <= 1.0
        assert meta.tokens_consumed >= 0
        assert meta.execution_duration_ms >= 0


# ---------------------------------------------------------------------------
# TestCleanConversations
# ---------------------------------------------------------------------------


class TestCleanConversations:
    def test_safe_tasks_return_none(self):
        """Tasks with safe assumptions should return None queue_deposit."""
        from bicameral_agent.dataset import ResearchQADataset

        dataset = ResearchQADataset()
        safe_tasks = dataset.with_assumptions()[:10]

        none_count = 0
        for task in safe_tasks:
            auditor = AssumptionAuditor()
            client = _mock_client(_assumption_response(all_safe=True))
            messages = [
                Message(role="user", content=task.question, timestamp_ms=1000, token_count=10),
                Message(role="assistant", content=task.gold_answer, timestamp_ms=2000, token_count=50),
            ]
            result = auditor.execute(messages, _make_state(), _DEFAULT_BUDGET, client)
            if result.queue_deposit is None:
                none_count += 1

        assert none_count >= 8, f"Expected >=8/10 None deposits, got {none_count}"


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_no_search_results_still_flags(self):
        """When search returns nothing, assumptions are still flagged."""

        class EmptyProvider:
            def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
                return []

        auditor = AssumptionAuditor(search_provider=EmptyProvider())
        client = _mock_client(_assumption_response())
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.metadata.items_found >= 1
        # Only 1 LLM call since no search results to assess
        assert client.generate.call_count == 1

    def test_empty_conversation_works(self):
        auditor = AssumptionAuditor()
        client = _mock_client(_assumption_response(all_safe=True))
        result = auditor.execute([], _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is None


# ---------------------------------------------------------------------------
# TestCustomSearchProvider
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

        auditor = AssumptionAuditor(search_provider=FakeProvider())
        client = _mock_client([_assumption_response(), _evidence_response()])
        result = auditor.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None


# ---------------------------------------------------------------------------
# Integration tests (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestIntegration:
    def test_assumption_detection_on_dataset(self):
        """Run against tasks with known assumptions, verify >=60% overlap."""
        from bicameral_agent.dataset import ResearchQADataset
        from bicameral_agent.gemini import GeminiClient

        dataset = ResearchQADataset()
        assumption_tasks = dataset.with_assumptions()[:15]
        client = GeminiClient()
        budget = TokenBudget(max_calls=3, max_input_tokens=5000, max_output_tokens=2000)

        def _tokenize(text: str) -> set[str]:
            import re
            return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}

        identified_count = 0
        for task in assumption_tasks:
            auditor = AssumptionAuditor()
            messages = [
                Message(role="user", content=task.question, timestamp_ms=1000, token_count=10),
                Message(
                    role="assistant",
                    content=task.gold_answer[:200],
                    timestamp_ms=2000,
                    token_count=50,
                ),
            ]
            result = auditor.execute(messages, _make_state(), budget, client)

            if result.queue_deposit is not None:
                deposit_tokens = _tokenize(result.queue_deposit.content)
                for known in task.known_assumptions or []:
                    known_tokens = _tokenize(known)
                    overlap = len(deposit_tokens & known_tokens) / max(len(known_tokens), 1)
                    if overlap > 0.2:
                        identified_count += 1
                        break

        rate = identified_count / len(assumption_tasks)
        assert rate >= 0.6, f"Assumption detection rate {rate:.0%} < 60%"
