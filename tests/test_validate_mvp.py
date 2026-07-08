"""Tests for scripts/validate_mvp.py (issue #31).

Runs the validator against the committed artifacts (no live LLM calls) and
checks that the machine-readable determinations match both the expected
verdicts and the narrative report in docs/mvp_validation_2026-07.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import validate_mvp  # noqa: E402

_DOC = _REPO_ROOT / "docs" / "mvp_validation_2026-07.md"

EXPECTED_DETERMINATIONS = {
    "criterion_1": "FAIL",
    "criterion_2": "FAIL",
    "criterion_3": "PASS",
    "criterion_4": "PASS",
    "criterion_5": "UNRESOLVABLE",
    "criterion_6": "UNEVALUATED",
}


@pytest.fixture(scope="module")
def validation(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the validator once against the committed artifacts."""
    out = tmp_path_factory.mktemp("mvp") / "mvp_validation.json"
    rc = validate_mvp.main(
        [
            "--comparative-report",
            str(_REPO_ROOT / "data" / "comparative" / "report.json"),
            "--metrics-history",
            str(_REPO_ROOT / "data" / "mcts_training" / "metrics_history.json"),
            "--emergent-stats",
            str(_REPO_ROOT / "docs" / "figures" / "emergent" / "emergent_stats.json"),
            "--data-dir",
            str(_REPO_ROOT / "data"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    return json.loads(out.read_text())


def test_determinations_match_expected(validation: dict) -> None:
    assert validation["determinations"] == EXPECTED_DETERMINATIONS


def test_json_matches_doc_determinations(validation: dict) -> None:
    """The JSON verdicts and the stakeholder doc must agree, criterion by criterion."""
    text = _DOC.read_text()
    doc_determinations = {
        f"criterion_{m.group(1)}": m.group(2)
        for m in re.finditer(
            r"## Criterion (\d).*?\*\*Determination: (PASS|FAIL|UNRESOLVABLE|UNEVALUATED)",
            text,
            re.DOTALL,
        )
    }
    assert doc_determinations == validation["determinations"]


def test_headline_effect_sizes(validation: dict) -> None:
    """The two failing quality criteria fail on effect size, not on a data hiccup."""
    c1 = validation["criteria"]["criterion_1"]
    assert c1["relative_improvement_pct"] == pytest.approx(4.72, abs=0.01)
    assert c1["relative_improvement_pct"] < c1["threshold_relative_pct"]
    assert not c1["significant_at_95"]

    c2 = validation["criteria"]["criterion_2"]
    assert c2["relative_improvement_pct"] == pytest.approx(4.02, abs=0.01)
    assert c2["relative_improvement_pct"] < c2["threshold_relative_pct"]
    assert not c2["significant_at_95"]


def test_criterion_6_reports_no_synchronous_data(validation: dict) -> None:
    c6 = validation["criteria"]["criterion_6"]
    modes = c6["committed_episodes_by_injection_mode"]
    assert "synchronous" not in modes
    assert modes.get("breakpoint", 0) > 0
    assert "ABTestRunner" in c6["command_to_produce_data"]


def test_criterion_6_resolves_when_ab_data_lands(tmp_path: Path) -> None:
    """Once a synchronous-arm ABTestResult JSON is committed, C6 becomes decidable."""

    def condition(name: str, derailment_mean: float) -> dict:
        return {
            "name": name,
            "derailments": {
                "mean": derailment_mean,
                "std": 0.5,
                "ci_lower": derailment_mean - 0.2,
                "ci_upper": derailment_mean + 0.2,
                "n": 50,
            },
        }

    result = {
        "conditions": [condition("synchronous", 1.4), condition("breakpoint", 0.6)],
        "episode_metrics": [],
        "best_condition": "breakpoint",
        "justification": "",
    }
    (tmp_path / "ab_test").mkdir()
    (tmp_path / "ab_test" / "report.json").write_text(json.dumps(result))

    c6 = validate_mvp.derailment_criterion(tmp_path)
    assert c6["determination"] == "PASS"
    assert c6["breakpoint_derailments_mean"] == pytest.approx(0.6)
    assert c6["synchronous_derailments_mean"] == pytest.approx(1.4)
