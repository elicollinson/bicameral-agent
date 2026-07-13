"""Tests for scripts/analyze_search_experiment.py (issue #101).

Builds a synthetic two-run fixture (two runs x three conditions) with
hand-computable qualities and verifies the paired intersection, the
paired-delta math, tier splits, injection/URL counting and
transport-failure extraction.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import analyze_search_experiment as ase  # noqa: E402


def _payload(
    task_id: str,
    quality: float | None,
    injections: list[dict] | None = None,
) -> str:
    return json.dumps({
        "metadata": {"task_id": task_id},
        "outcome": {"quality_score": quality},
        "context_injections": injections or [],
    })


def _write_condition(path: Path, payloads: list[str]) -> None:
    table = pa.table({
        "episode_id": [f"ep-{i}" for i in range(len(payloads))],
        "payload": payloads,
    })
    pq.write_table(table, path)


def _injection(content: str, consumed: bool) -> dict:
    return {"content": content, "consumed": consumed}


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path]:
    """Two synthetic run dirs sharing 3 paired tasks (2 hard, 1 tricky).

    frames_hard_001, frames_hard_002 and crepe_tricky_001 complete in all
    six arms. frames_hard_003 is missing from nosearch/no_subconscious and
    has a null quality in search/random, so it must be excluded from the
    intersection either way.
    """
    nosearch = tmp_path / "nosearch"
    search = tmp_path / "search"
    nosearch.mkdir()
    search.mkdir()

    quality = {
        # (run, cond): {task: quality}
        ("nosearch", "no_subconscious"): {
            "frames_hard_001": 0.2,
            "frames_hard_002": 0.4,
            "crepe_tricky_001": 0.6,
        },
        ("nosearch", "random"): {
            "frames_hard_001": 0.5,
            "frames_hard_002": 0.5,
            "crepe_tricky_001": 0.5,
            "frames_hard_003": 1.0,
        },
        ("nosearch", "heuristic"): {
            "frames_hard_001": 0.4,
            "frames_hard_002": 0.4,
            "crepe_tricky_001": 1.0,
            "frames_hard_003": 1.0,
        },
        ("search", "no_subconscious"): {
            "frames_hard_001": 0.4,
            "frames_hard_002": 0.8,
            "crepe_tricky_001": 0.6,
            "frames_hard_003": 1.0,
        },
        ("search", "random"): {
            "frames_hard_001": 0.5,
            "frames_hard_002": 0.5,
            "crepe_tricky_001": 0.5,
            "frames_hard_003": None,
        },
        ("search", "heuristic"): {
            "frames_hard_001": 0.7,
            "frames_hard_002": 0.9,
            "crepe_tricky_001": 0.8,
            "frames_hard_003": 1.0,
        },
    }
    injections = {
        ("search", "heuristic"): {
            "frames_hard_001": [
                _injection("see https://example.com/a", consumed=True),
                _injection("no url here", consumed=False),
            ],
            "frames_hard_002": [
                _injection("plain finding", consumed=True),
            ],
        },
    }
    for (run, cond), tasks in quality.items():
        run_dir = nosearch if run == "nosearch" else search
        payloads = [
            _payload(t, q, injections.get((run, cond), {}).get(t))
            for t, q in tasks.items()
        ]
        _write_condition(run_dir / f"{cond}.parquet", payloads)

    (nosearch / "summary.json").write_text(
        json.dumps({"failures": [
            {"task_id": "frames_hard_009", "condition": "random", "error": "t"},
        ]})
    )
    (search / "summary.json").write_text(
        json.dumps({"failures": [
            {"task_id": "frames_hard_008", "condition": "no_subconscious", "error": "t"},
            {"task_id": "frames_hard_009", "condition": "no_subconscious", "error": "t"},
        ]})
    )
    return nosearch, search


def _analyze(nosearch: Path, search: Path) -> dict:
    return ase.analyze(
        nosearch=ase.load_run(nosearch),
        search=ase.load_run(search),
        nosearch_failures=ase.transport_failures(nosearch),
        search_failures=ase.transport_failures(search),
    )


def test_intersection_excludes_missing_and_unscored(runs) -> None:
    result = _analyze(*runs)
    assert result["paired_task_count"] == 3
    assert result["paired_tasks"] == [
        "crepe_tricky_001",
        "frames_hard_001",
        "frames_hard_002",
    ]
    assert result["tier_task_counts"] == {"all": 3, "tricky": 1, "hard": 2}


def test_paired_means_and_deltas(runs) -> None:
    result = _analyze(*runs)
    means = result["paired_means"]
    assert means["nosearch"]["no_subconscious"] == pytest.approx(0.4)
    assert means["search"]["heuristic"] == pytest.approx(0.8)

    # heuristic search - nosearch diffs: +0.3, +0.5, -0.2 -> mean +0.2.
    delta = result["search_vs_nosearch"]["heuristic"]["all"]
    diffs = [0.3, 0.5, -0.2]
    mean = sum(diffs) / 3
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / 2)
    margin = 4.303 * sd / math.sqrt(3)  # t_critical_95(df=2)
    assert delta["n"] == 3
    assert delta["mean"] == pytest.approx(mean)
    assert delta["ci_lower"] == pytest.approx(mean - margin, abs=1e-3)
    assert delta["ci_upper"] == pytest.approx(mean + margin, abs=1e-3)

    # random arm is identical across runs: delta exactly zero.
    assert result["search_vs_nosearch"]["random"]["all"]["mean"] == 0.0


def test_tier_split_of_within_run_contrast(runs) -> None:
    result = _analyze(*runs)
    contrast = result["heuristic_vs_no_subconscious"]
    # nosearch hard diffs: 0.4-0.2, 0.4-0.4 -> mean +0.1; tricky: 1.0-0.6.
    assert contrast["nosearch"]["hard"]["mean"] == pytest.approx(0.1)
    assert contrast["nosearch"]["hard"]["n"] == 2
    assert contrast["nosearch"]["tricky"]["mean"] == pytest.approx(0.4)
    # search hard diffs: 0.7-0.4, 0.9-0.8 -> mean +0.2.
    assert contrast["search"]["hard"]["mean"] == pytest.approx(0.2)


def test_injection_and_failure_stats(runs) -> None:
    result = _analyze(*runs)
    stats = result["injection_stats"]["search"]["heuristic"]
    assert stats == {
        "n_episodes": 4,
        "injections": 3,
        "consumed": 2,
        "url_bearing": 1,
    }
    assert result["injection_stats"]["nosearch"]["no_subconscious"] == {
        "n_episodes": 3,
        "injections": 0,
        "consumed": 0,
        "url_bearing": 0,
    }
    assert result["transport_failures"] == {
        "nosearch": {"no_subconscious": 0, "random": 1, "heuristic": 0},
        "search": {"no_subconscious": 2, "random": 0, "heuristic": 0},
    }


def test_duplicate_task_id_rejected(tmp_path: Path) -> None:
    path = tmp_path / "heuristic.parquet"
    _write_condition(path, [_payload("t1", 0.5), _payload("t1", 0.7)])
    with pytest.raises(ValueError, match="duplicate task_id"):
        ase.load_condition(path)


def _record(
    task_id: str,
    quality: float | None,
    injections: tuple[dict, ...],
    turn_messages: tuple[tuple[int, str, str], ...],
) -> ase.EpisodeRecord:
    return ase.EpisodeRecord(
        task_id=task_id,
        quality=quality,
        n_injections=len(injections),
        n_consumed=sum(1 for i in injections if i.get("consumed")),
        n_url_bearing=0,
        injections=injections,
        turn_messages=turn_messages,
    )


def _consumed(content: str) -> dict:
    return {"content": content, "consumed": True, "consumed_at_turn": 2}


@pytest.fixture
def utilization_records() -> dict[str, ase.EpisodeRecord]:
    """Five consumed-injection episodes covering every utilization rule.

    'verification' appears in three injections, so it crosses the
    document-frequency cutoff (max(3, ceil(0.25 * 5)) = 3) and must be
    treated as boilerplate.
    """
    return {
        # Utilized: 'scanderbeg', 'albanian', 'chieftain' are novel and reused.
        "token_hit": _record(
            "token_hit",
            1.0,
            (_consumed("verification: Scanderbeg the Albanian chieftain fought"),),
            (
                (1, "user", "who ended the war?"),
                (1, "assistant", "The pope ended a war."),
                (2, "user", "are you sure?"),
                (2, "assistant", "Per evidence: Scanderbeg, the Albanian chieftain."),
            ),
        ),
        # Utilized via verbatim URL despite only one reused token.
        "url_hit": _record(
            "url_hit",
            0.5,
            (_consumed("verification of claim: see https://ex.com/b now"),),
            (
                (1, "user", "question"),
                (1, "assistant", "First answer."),
                (2, "user", "source?"),
                (2, "assistant", "Per https://ex.com/b the claim holds."),
            ),
        ),
        # Not utilized: injected content ignored in the reply.
        "ignored": _record(
            "ignored",
            0.0,
            (_consumed("wolfram tantalum discovered near stockholm"),),
            (
                (1, "user", "question"),
                (1, "assistant", "First answer."),
                (2, "user", "sure?"),
                (2, "assistant", "Restating the first answer unchanged."),
            ),
        ),
        # Not utilized: 'gallium' was already in the pre-injection transcript
        # (scanner echo), so only 'xenon' is novel -> below the threshold.
        "pre_echo": _record(
            "pre_echo",
            0.25,
            (_consumed("gallium xenon measurements"),),
            (
                (1, "user", "question"),
                (1, "assistant", "It involves gallium."),
                (2, "user", "more?"),
                (2, "assistant", "Both gallium and xenon are involved."),
            ),
        ),
        # Not utilized: 'verification' is boilerplate, leaving one novel token.
        "boilerplate": _record(
            "boilerplate",
            0.75,
            (_consumed("verification of the osmium claim"),),
            (
                (1, "user", "question"),
                (1, "assistant", "First answer."),
                (2, "user", "check?"),
                (2, "assistant", "After verification, osmium stands."),
            ),
        ),
        # No consumed injection: excluded from the stats entirely.
        "no_injection": _record("no_injection", 1.0, (), ()),
    }


def test_utilization_classification(utilization_records) -> None:
    stats = ase.utilization_stats(utilization_records)
    assert stats["episodes_with_consumed_injections"] == 5
    assert "no_injection" not in stats["per_episode"]
    per = stats["per_episode"]
    assert per["token_hit"]["utilized"] is True
    assert "scanderbeg" in per["token_hit"]["used_tokens"]
    assert per["url_hit"]["utilized"] is True
    assert per["ignored"]["utilized"] is False
    assert per["pre_echo"]["utilized"] is False
    assert per["pre_echo"]["used_tokens"] == ["xenon"]
    assert per["boilerplate"]["utilized"] is False
    assert per["boilerplate"]["used_tokens"] == ["osmium"]
    assert stats["utilized_episodes"] == 2
    assert stats["url_citations"] == 1


def test_utilization_quality_split(utilization_records) -> None:
    stats = ase.utilization_stats(utilization_records)
    assert stats["quality_utilized"]["n"] == 2
    assert stats["quality_utilized"]["mean"] == pytest.approx(0.75)
    assert stats["quality_not_utilized"]["n"] == 3
    assert stats["quality_not_utilized"]["mean"] == pytest.approx(1.0 / 3)


def test_analyze_includes_utilization(runs) -> None:
    result = _analyze(*runs)
    stats = result["utilization"]["search"]
    # Two heuristic episodes carry consumed injections; the fixture payloads
    # have no messages, so nothing can be detected as utilized.
    assert stats["episodes_with_consumed_injections"] == 2
    assert stats["utilized_episodes"] == 0
