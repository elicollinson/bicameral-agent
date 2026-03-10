"""Research Gap Scanner tool primitive.

Examines conversation history, identifies informational gaps (unsupported
claims, missing data), searches for relevant information, and packages
results as a QueueItem for context injection.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from bicameral_agent.queue import Priority, QueueItem
from bicameral_agent.schema import Message
from bicameral_agent.tool_primitive import ToolMetadata, ToolPrimitive, ToolResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class GapCategory(str, enum.Enum):
    """Classification of an identified gap, mapping to Priority levels."""

    CORE_CLAIM = "core_claim"
    SUPPLEMENTARY = "supplementary"
    NICE_TO_HAVE = "nice_to_have"


_CATEGORY_TO_PRIORITY = {
    GapCategory.CORE_CLAIM: Priority.HIGH,
    GapCategory.SUPPLEMENTARY: Priority.MEDIUM,
    GapCategory.NICE_TO_HAVE: Priority.LOW,
}


@dataclass(frozen=True, slots=True)
class IdentifiedGap:
    """A single informational gap found in conversation."""

    description: str
    category: GapCategory
    search_query: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result returned by a SearchProvider."""

    title: str
    snippet: str
    relevance_score: float
    source: str


# ---------------------------------------------------------------------------
# Search interface
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for pluggable search backends."""

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        ...


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter empty."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class MockSearchProvider:
    """Keyword-matching search over built-in research snippets."""

    _SNIPPETS: list[dict[str, str]] = [
        {"title": "High-temperature superconductivity in hydrides", "snippet": "Recent studies show hydrogen-rich compounds under extreme pressure can exhibit superconductivity above 200K, with LaH10 reaching critical temperatures near 250K.", "source": "arxiv:2023.superconductors"},
        {"title": "Room-temperature superconductor claims", "snippet": "LK-99 claims of room-temperature superconductivity were not replicated. Independent labs found the material is not a superconductor but exhibits diamagnetic properties from copper sulfide impurities.", "source": "nature:2023.lk99"},
        {"title": "Psilocybin therapy for depression", "snippet": "Clinical trials show psilocybin-assisted therapy produces rapid and sustained antidepressant effects. A single 25mg dose showed significant improvement in treatment-resistant depression at 3 weeks.", "source": "nejm:2022.psilocybin"},
        {"title": "MDMA-assisted therapy for PTSD", "snippet": "Phase 3 trials of MDMA-assisted therapy for PTSD showed 67% of participants no longer met PTSD diagnostic criteria after treatment, compared to 32% in placebo group.", "source": "nature:2023.mdma"},
        {"title": "Psychedelic mechanisms of action", "snippet": "Psychedelics primarily act on 5-HT2A serotonin receptors, promoting neuroplasticity and disrupting default mode network activity. This may explain their therapeutic effects on rigid thought patterns.", "source": "cell:2023.psychedelics"},
        {"title": "ITER fusion reactor progress", "snippet": "The ITER tokamak project aims to demonstrate net energy gain from fusion by 2035. First plasma is now expected around 2025, with deuterium-tritium operations planned for the 2030s.", "source": "iter:2024.progress"},
        {"title": "National Ignition Facility breakthrough", "snippet": "In December 2022, NIF achieved fusion ignition for the first time, producing 3.15 MJ of energy from 2.05 MJ of laser input — a gain factor of 1.54.", "source": "llnl:2023.nif"},
        {"title": "Compact fusion reactor designs", "snippet": "Companies like Commonwealth Fusion Systems are developing compact tokamaks using high-temperature superconducting magnets, potentially enabling smaller and cheaper fusion power plants.", "source": "science:2023.compact_fusion"},
        {"title": "CAR-T cell therapy advances", "snippet": "CAR-T therapies have shown remarkable efficacy in blood cancers, with complete remission rates of 40-54% in diffuse large B-cell lymphoma patients who failed prior treatments.", "source": "blood:2023.cart"},
        {"title": "CAR-T therapy side effects", "snippet": "Cytokine release syndrome occurs in 50-90% of CAR-T patients. Neurotoxicity affects 20-60%. Management protocols using tocilizumab and corticosteroids have improved safety profiles.", "source": "jco:2023.cart_safety"},
        {"title": "Solid tumor CAR-T challenges", "snippet": "CAR-T therapy faces major hurdles in solid tumors: hostile tumor microenvironment, antigen heterogeneity, T-cell exhaustion, and poor trafficking to tumor sites.", "source": "nature_rev:2023.solid_cart"},
        {"title": "Microplastics in human blood", "snippet": "A 2022 study detected microplastics in 77% of human blood samples tested. PET and polystyrene were the most common polymers found, raising concerns about systemic exposure.", "source": "env_int:2022.microplastics_blood"},
        {"title": "Microplastics health effects", "snippet": "Microplastics can cross biological barriers including the blood-brain barrier. Animal studies show inflammation, oxidative stress, and gut microbiome disruption, but human health effects remain under investigation.", "source": "lancet:2023.microplastics"},
        {"title": "Nanoplastics in bottled water", "snippet": "Researchers found approximately 240,000 nanoplastic particles per liter of bottled water using stimulated Raman scattering microscopy, far exceeding previous microplastic estimates.", "source": "pnas:2024.nanoplastics"},
        {"title": "Quantum error correction milestones", "snippet": "Google's quantum team demonstrated that increasing qubit count in their surface code actually reduced logical error rates, a key threshold for practical quantum error correction.", "source": "nature:2023.quantum_ec"},
        {"title": "CRISPR gene therapy approvals", "snippet": "Casgevy (exagamglogene autotemcel) became the first CRISPR-based therapy approved by FDA in December 2023 for sickle cell disease and transfusion-dependent beta-thalassemia.", "source": "fda:2023.crispr"},
        {"title": "mRNA vaccine platform advances", "snippet": "mRNA technology is being adapted beyond COVID-19 for cancer vaccines, rare diseases, and autoimmune conditions. Personalized cancer vaccines showed promising phase 2 results in melanoma.", "source": "science:2024.mrna"},
        {"title": "Artificial general intelligence timelines", "snippet": "Expert surveys show wide disagreement on AGI timelines. Median estimates range from 2040 to 2060, with significant uncertainty. Most researchers caution against specific predictions.", "source": "arxiv:2024.agi_survey"},
        {"title": "Large language model scaling laws", "snippet": "Scaling laws show predictable improvement in LLM performance with compute, data, and parameters. However, recent work suggests data quality and architecture innovations may matter more than raw scale.", "source": "arxiv:2024.scaling"},
        {"title": "Climate tipping points research", "snippet": "Multiple climate tipping points may be crossed between 1.5-2°C of warming, including collapse of Greenland and West Antarctic ice sheets, Amazon rainforest dieback, and permafrost thaw.", "source": "science:2023.tipping_points"},
    ]

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, str]]] = []
        for snippet_data in self._SNIPPETS:
            doc_tokens = set(
                _tokenize(snippet_data["title"])
                + _tokenize(snippet_data["snippet"])
            )
            if not doc_tokens:
                continue
            intersection = query_tokens & doc_tokens
            union = query_tokens | doc_tokens
            jaccard = len(intersection) / len(union)
            if jaccard > 0.1:
                scored.append((jaccard, snippet_data))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                title=item["title"],
                snippet=item["snippet"],
                relevance_score=round(score, 3),
                source=item["source"],
            )
            for score, item in scored[:max_results]
        ]


# ---------------------------------------------------------------------------
# LLM prompts and schemas
# ---------------------------------------------------------------------------

_GAP_IDENTIFICATION_SYSTEM = """\
You are an expert research analyst. Examine the conversation and identify \
informational gaps: unsupported claims, missing evidence, areas where \
additional data would strengthen the discussion.

Classify each gap:
- core_claim: Central claims lacking evidence (high priority)
- supplementary: Supporting details that would strengthen arguments (medium)
- nice_to_have: Interesting tangential information (low)

Generate a concise search query for each gap."""

_GAP_IDENTIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "has_gaps": {"type": "boolean"},
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["core_claim", "supplementary", "nice_to_have"],
                    },
                    "search_query": {"type": "string"},
                },
                "required": ["description", "category", "search_query"],
            },
        },
    },
    "required": ["has_gaps", "gaps"],
}

_RANKING_SYSTEM = """\
You are a research relevance judge. Given identified gaps and search results, \
filter and rank the results by relevance. Assign a relevance_score (0.0-1.0) \
to each result. Only include results with relevance_score >= 0.3. \
Provide an overall_confidence (0.0-1.0) reflecting how well the search \
results address the identified gaps."""

_RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_confidence": {"type": "number"},
        "ranked_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gap_description": {"type": "string"},
                    "title": {"type": "string"},
                    "snippet": {"type": "string"},
                    "relevance_score": {"type": "number"},
                    "source": {"type": "string"},
                },
                "required": [
                    "gap_description",
                    "title",
                    "snippet",
                    "relevance_score",
                    "source",
                ],
            },
        },
    },
    "required": ["overall_confidence", "ranked_results"],
}


# ---------------------------------------------------------------------------
# ResearchGapScanner
# ---------------------------------------------------------------------------


class ResearchGapScanner(ToolPrimitive):
    """Scans conversation for research gaps and searches for relevant info.

    Uses up to 2 LLM calls (within a budget of 3):
    1. Gap identification from conversation history
    2. Relevance ranking of search results (if any found)
    """

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        super().__init__("research_gap_scanner")
        self._search_provider = search_provider or MockSearchProvider()

    def _execute(self, conversation_history, reasoning_state, client):
        # Format conversation (last 10 messages)
        conv_text = _format_conversation(conversation_history)

        # Call 1: Identify gaps
        gaps = _identify_gaps(conv_text, client)

        if not gaps:
            return ToolResult(
                queue_deposit=None,
                metadata=ToolMetadata(
                    tool_id=self.tool_id,
                    action_taken="scanned conversation, no gaps found",
                    confidence=0.8,
                    items_found=0,
                    estimated_relevance=0.0,
                ),
            )

        # Search for each gap (no LLM call)
        all_search_results: dict[str, list[SearchResult]] = {}
        for gap in gaps:
            results = self._search_provider.search(gap.search_query)
            if results:
                all_search_results[gap.description] = results

        if not all_search_results:
            return ToolResult(
                queue_deposit=QueueItem(
                    content=_format_gaps_only(gaps),
                    priority=_max_priority(gaps),
                    source_tool_id=self.tool_id,
                    token_count=sum(len(g.description.split()) for g in gaps) * 2,
                    dedup_key=_make_dedup_key(gaps),
                ),
                metadata=ToolMetadata(
                    tool_id=self.tool_id,
                    action_taken="identified gaps but no search results found",
                    confidence=0.5,
                    items_found=len(gaps),
                    estimated_relevance=0.3,
                ),
            )

        # Call 2: Rank results
        ranked = _rank_results(gaps, all_search_results, client)
        overall_confidence = ranked.get("overall_confidence", 0.5)
        ranked_results = ranked.get("ranked_results", [])

        # Filter to relevant results
        relevant = [r for r in ranked_results if r.get("relevance_score", 0) >= 0.3]

        if not relevant:
            return ToolResult(
                queue_deposit=QueueItem(
                    content=_format_gaps_only(gaps),
                    priority=_max_priority(gaps),
                    source_tool_id=self.tool_id,
                    token_count=sum(len(g.description.split()) for g in gaps) * 2,
                    dedup_key=_make_dedup_key(gaps),
                ),
                metadata=ToolMetadata(
                    tool_id=self.tool_id,
                    action_taken="identified gaps, search results not relevant enough",
                    confidence=0.5,
                    items_found=len(gaps),
                    estimated_relevance=0.3,
                ),
            )

        content = _format_ranked_content(relevant)
        relevance_scores = [r["relevance_score"] for r in relevant]
        mean_relevance = sum(relevance_scores) / len(relevance_scores)

        return ToolResult(
            queue_deposit=QueueItem(
                content=content,
                priority=_max_priority(gaps),
                source_tool_id=self.tool_id,
                token_count=len(content.split()),
                dedup_key=_make_dedup_key(gaps),
            ),
            metadata=ToolMetadata(
                tool_id=self.tool_id,
                action_taken="identified gaps and found relevant research",
                confidence=overall_confidence,
                items_found=len(relevant),
                estimated_relevance=mean_relevance,
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


def _identify_gaps(conv_text: str, client) -> list[IdentifiedGap]:
    """Call 1: Send conversation to LLM, parse structured gap JSON."""
    response = client.generate(
        [{"role": "user", "content": f"Analyze this conversation for research gaps:\n\n{conv_text}"}],
        system_prompt=_GAP_IDENTIFICATION_SYSTEM,
        thinking_level="low",
        temperature=0,
        max_output_tokens=1000,
        response_schema=_GAP_IDENTIFICATION_SCHEMA,
    )

    parsed = json.loads(response.content)
    if not parsed.get("has_gaps", False):
        return []

    gaps = []
    for gap_data in parsed.get("gaps", []):
        try:
            category = GapCategory(gap_data["category"])
        except (ValueError, KeyError):
            category = GapCategory.NICE_TO_HAVE
        gaps.append(
            IdentifiedGap(
                description=gap_data["description"],
                category=category,
                search_query=gap_data["search_query"],
            )
        )
    return gaps


def _rank_results(
    gaps: list[IdentifiedGap],
    search_results: dict[str, list[SearchResult]],
    client,
) -> dict:
    """Call 2: Send gaps + search results to LLM for relevance ranking."""
    # Format input for ranking
    lines = ["## Identified Gaps"]
    for gap in gaps:
        lines.append(f"- [{gap.category.value}] {gap.description}")

    lines.append("\n## Search Results")
    for gap_desc, results in search_results.items():
        lines.append(f"\n### For gap: {gap_desc}")
        for r in results:
            lines.append(f"- **{r.title}** ({r.source}): {r.snippet}")

    content = "\n".join(lines)

    response = client.generate(
        [{"role": "user", "content": f"Rank these search results by relevance to the gaps:\n\n{content}"}],
        system_prompt=_RANKING_SYSTEM,
        thinking_level="minimal",
        temperature=0,
        max_output_tokens=400,
        response_schema=_RANKING_SCHEMA,
    )

    return json.loads(response.content)


def _max_priority(gaps: list[IdentifiedGap]) -> Priority:
    """Return the highest priority among gaps."""
    priorities = [_CATEGORY_TO_PRIORITY[g.category] for g in gaps]
    return max(priorities)


def _make_dedup_key(gaps: list[IdentifiedGap]) -> str:
    """SHA-256 hash of sorted gap descriptions, prefixed gap_scanner:."""
    descriptions = sorted(g.description for g in gaps)
    h = hashlib.sha256("|".join(descriptions).encode()).hexdigest()
    return f"gap_scanner:{h}"


def _format_gaps_only(gaps: list[IdentifiedGap]) -> str:
    """Format gaps without search results."""
    lines = ["Research gaps identified:"]
    for gap in gaps:
        lines.append(f"- [{gap.category.value}] {gap.description}")
    return "\n".join(lines)


def _format_ranked_content(ranked: list[dict]) -> str:
    """Format ranked results into a multi-gap content string for QueueItem."""
    lines = ["Research findings for identified gaps:"]
    for r in ranked:
        lines.append(
            f"\n**{r.get('gap_description', 'Unknown gap')}**\n"
            f"  {r['title']} (relevance: {r['relevance_score']:.1f})\n"
            f"  {r['snippet']}\n"
            f"  Source: {r['source']}"
        )
    return "\n".join(lines)
