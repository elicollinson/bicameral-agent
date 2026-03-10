"""Assumption Auditor tool primitive.

Identifies untested assumptions in conversation reasoning, rates their risk
level, optionally searches for evidence, and packages flagged assumptions
as a QueueItem with suggested actions.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass

from bicameral_agent.gap_scanner import MockSearchProvider, SearchProvider, SearchResult
from bicameral_agent.queue import Priority, QueueItem
from bicameral_agent.schema import Message
from bicameral_agent.tool_primitive import (
    BudgetExceededError,
    ToolMetadata,
    ToolPrimitive,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RiskLevel(str, enum.Enum):
    """Risk classification for an identified assumption."""

    SAFE = "safe"
    MODERATE = "moderate"
    HIGH = "high"


class SuggestedAction(str, enum.Enum):
    """Recommended action for a flagged assumption."""

    VALIDATE = "validate"
    HEDGE = "hedge"
    REVISE = "revise"


class EvidenceVerdict(str, enum.Enum):
    """Verdict from evidence assessment."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class IdentifiedAssumption:
    """A single assumption found in conversation reasoning."""

    description: str
    risk_level: RiskLevel
    basis: str
    search_query: str | None


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """Evidence assessment for a high-risk assumption."""

    assumption_description: str
    verdict: EvidenceVerdict
    evidence_summary: str
    suggested_action: SuggestedAction
    source: str


# ---------------------------------------------------------------------------
# Priority mapping
# ---------------------------------------------------------------------------


def _compute_priority(
    risk_level: RiskLevel,
    verdict: EvidenceVerdict | None = None,
) -> Priority:
    """Deterministic priority from risk level and evidence verdict.

    - high + contradicting → CRITICAL
    - high + no evidence or inconclusive → HIGH
    - high + supporting → MEDIUM (demoted)
    - moderate → MEDIUM
    """
    if risk_level == RiskLevel.HIGH:
        if verdict == EvidenceVerdict.CONTRADICTING:
            return Priority.CRITICAL
        if verdict == EvidenceVerdict.SUPPORTING:
            return Priority.MEDIUM
        return Priority.HIGH
    if risk_level == RiskLevel.MODERATE:
        return Priority.MEDIUM
    return Priority.LOW


# ---------------------------------------------------------------------------
# LLM prompts and schemas
# ---------------------------------------------------------------------------

_ASSUMPTION_EXTRACTION_SYSTEM = """\
You are an expert critical thinking analyst. Examine the conversation and \
identify every assumption being made — claims taken as true without explicit \
evidence or justification.

For each assumption:
- Describe it clearly
- Classify risk: safe (well-established, low consequence if wrong), \
moderate (plausible but untested), high (critical to reasoning and unverified)
- Provide a brief basis for the risk classification
- For high-risk assumptions only, generate a concise search query to find evidence"""

_ASSUMPTION_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "assumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["safe", "moderate", "high"],
                    },
                    "basis": {"type": "string"},
                    "search_query": {"type": ["string", "null"]},
                },
                "required": ["description", "risk_level", "basis"],
            },
        },
    },
    "required": ["assumptions"],
}

_EVIDENCE_ASSESSMENT_SYSTEM = """\
You are an evidence evaluator. For each assumption and its search results, \
determine:
- verdict: supporting (evidence confirms assumption), contradicting (evidence \
refutes it), or inconclusive (mixed or insufficient evidence)
- suggested_action: validate (gather more evidence), hedge (qualify the claim), \
or revise (change the reasoning)
- evidence_summary: brief summary of what the evidence shows
- source: the most relevant source reference"""

_EVIDENCE_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "assumption_description": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supporting", "contradicting", "inconclusive"],
                    },
                    "evidence_summary": {"type": "string"},
                    "suggested_action": {
                        "type": "string",
                        "enum": ["validate", "hedge", "revise"],
                    },
                    "source": {"type": "string"},
                },
                "required": [
                    "assumption_description",
                    "verdict",
                    "evidence_summary",
                    "suggested_action",
                    "source",
                ],
            },
        },
    },
    "required": ["assessments"],
}


# ---------------------------------------------------------------------------
# AssumptionAuditor
# ---------------------------------------------------------------------------


class AssumptionAuditor(ToolPrimitive):
    """Audits conversation for untested assumptions and assesses evidence.

    Uses up to 2 LLM calls:
    1. Extract assumptions from conversation history
    2. Assess evidence for high-risk assumptions (if search results found)
    """

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        super().__init__("assumption_auditor")
        self._search_provider = search_provider or MockSearchProvider()

    def _execute(self, conversation_history, reasoning_state, client):
        conv_text = _format_conversation(conversation_history)

        # Call 1: Extract assumptions
        assumptions = _extract_assumptions(conv_text, client)

        # Filter out safe assumptions
        flagged = [a for a in assumptions if a.risk_level != RiskLevel.SAFE]

        if not flagged:
            return ToolResult(
                queue_deposit=None,
                metadata=ToolMetadata(
                    tool_id=self.tool_id,
                    action_taken="audited conversation, all assumptions safe",
                    confidence=0.8,
                    items_found=0,
                    estimated_relevance=0.0,
                ),
            )

        # Search for evidence on high-risk assumptions
        high_risk = [a for a in flagged if a.risk_level == RiskLevel.HIGH]
        search_results: dict[str, list[SearchResult]] = {}
        for assumption in high_risk:
            if assumption.search_query:
                results = self._search_provider.search(assumption.search_query)
                if results:
                    search_results[assumption.description] = results

        # Call 2: Assess evidence (if we have search results)
        evidence_map: dict[str, EvidenceResult] = {}
        if search_results:
            try:
                evidence_map = _assess_evidence(high_risk, search_results, client)
            except BudgetExceededError:
                pass  # Graceful fallback: proceed without evidence assessment

        # Build priority for each flagged assumption
        priorities: list[Priority] = []
        content_lines = ["Assumptions requiring attention:"]

        for assumption in flagged:
            evidence = evidence_map.get(assumption.description)
            verdict = evidence.verdict if evidence else None
            priority = _compute_priority(assumption.risk_level, verdict)
            priorities.append(priority)

            action = (
                evidence.suggested_action.value
                if evidence
                else SuggestedAction.VALIDATE.value
            )
            line = (
                f"\n- [{assumption.risk_level.value}] {assumption.description}\n"
                f"  Basis: {assumption.basis}\n"
                f"  Action: {action}"
            )
            if evidence:
                line += (
                    f"\n  Evidence: {evidence.evidence_summary}"
                    f"\n  Verdict: {evidence.verdict.value}"
                    f"\n  Source: {evidence.source}"
                )
            content_lines.append(line)

        content = "\n".join(content_lines)
        max_priority = max(priorities)
        dedup_key = _make_dedup_key(flagged)

        return ToolResult(
            queue_deposit=QueueItem(
                content=content,
                priority=max_priority,
                source_tool_id=self.tool_id,
                token_count=len(content.split()),
                dedup_key=dedup_key,
            ),
            metadata=ToolMetadata(
                tool_id=self.tool_id,
                action_taken="audited assumptions and flagged risks",
                confidence=0.7 if evidence_map else 0.5,
                items_found=len(flagged),
                estimated_relevance=0.6 if evidence_map else 0.4,
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_conversation(history: list[Message]) -> str:
    """Format last 10 messages as [role]: content lines."""
    recent = history[-10:]
    lines = []
    for msg in recent:
        lines.append(f"[{msg.role}]: {msg.content}")
    return "\n".join(lines)


def _extract_assumptions(conv_text: str, client) -> list[IdentifiedAssumption]:
    """Call 1: Send conversation to LLM, parse structured assumption JSON."""
    response = client.generate(
        [{"role": "user", "content": f"Analyze this conversation for untested assumptions:\n\n{conv_text}"}],
        system_prompt=_ASSUMPTION_EXTRACTION_SYSTEM,
        thinking_level="low",
        temperature=0,
        max_output_tokens=1000,
        response_schema=_ASSUMPTION_EXTRACTION_SCHEMA,
    )

    parsed = json.loads(response.content)
    assumptions = []
    for item in parsed.get("assumptions", []):
        try:
            risk = RiskLevel(item["risk_level"])
        except (ValueError, KeyError):
            risk = RiskLevel.SAFE
        assumptions.append(
            IdentifiedAssumption(
                description=item["description"],
                risk_level=risk,
                basis=item.get("basis", ""),
                search_query=item.get("search_query"),
            )
        )
    return assumptions


def _assess_evidence(
    assumptions: list[IdentifiedAssumption],
    search_results: dict[str, list[SearchResult]],
    client,
) -> dict[str, EvidenceResult]:
    """Call 2: Assess search evidence for high-risk assumptions."""
    lines = ["## High-Risk Assumptions and Search Results"]
    for assumption in assumptions:
        results = search_results.get(assumption.description)
        if not results:
            continue
        lines.append(f"\n### Assumption: {assumption.description}")
        lines.append(f"Basis: {assumption.basis}")
        lines.append("Search results:")
        for r in results:
            lines.append(f"- **{r.title}** ({r.source}): {r.snippet}")

    content = "\n".join(lines)

    response = client.generate(
        [{"role": "user", "content": f"Assess the evidence for these assumptions:\n\n{content}"}],
        system_prompt=_EVIDENCE_ASSESSMENT_SYSTEM,
        thinking_level="minimal",
        temperature=0,
        max_output_tokens=400,
        response_schema=_EVIDENCE_ASSESSMENT_SCHEMA,
    )

    parsed = json.loads(response.content)
    evidence_map: dict[str, EvidenceResult] = {}
    for item in parsed.get("assessments", []):
        try:
            verdict = EvidenceVerdict(item["verdict"])
            action = SuggestedAction(item["suggested_action"])
        except (ValueError, KeyError):
            verdict = EvidenceVerdict.INCONCLUSIVE
            action = SuggestedAction.VALIDATE
        evidence_map[item["assumption_description"]] = EvidenceResult(
            assumption_description=item["assumption_description"],
            verdict=verdict,
            evidence_summary=item.get("evidence_summary", ""),
            suggested_action=action,
            source=item.get("source", ""),
        )
    return evidence_map


def _make_dedup_key(assumptions: list[IdentifiedAssumption]) -> str:
    """SHA-256 hash of sorted assumption descriptions, prefixed assumption_auditor:."""
    descriptions = sorted(a.description for a in assumptions)
    h = hashlib.sha256("|".join(descriptions).encode()).hexdigest()
    return f"assumption_auditor:{h}"
