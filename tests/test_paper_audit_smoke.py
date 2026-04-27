"""Smoke test — every \\VAR in the advanced template resolves on the canonical fixture.

Audit A1: on a canonical_347, single-seed, advanced-template build the
residual unresolved-key set after the
``expand_inputs → collect_unresolved → is_missing_ok`` filter MUST be
empty.  A regression on this contract indicates either a missing
producer (real bug) or a missing-OK allow-list entry (allow-list
omission).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from report.combine_new import merge_ablation_report
from report.inject import collect_unresolved, expand_inputs
from report.inject_tables import inject_tables
from report.missing import is_missing_ok

_FIXTURE = Path(__file__).parent / "fixtures" / "canonical_347_n1.json"
_TEMPLATE = Path(__file__).resolve().parents[1] / "report" / "template_advanced.tex"


def _fixture_metrics() -> dict[str, object]:
    metrics: dict[str, object] = json.loads(_FIXTURE.read_text())
    config = MagicMock()
    config.output_dir = "/nonexistent"  # heal from results/bug_timeline.json
    merge_ablation_report(config, metrics)
    return metrics


def test_advanced_template_no_required_unresolved_on_canonical_fixture() -> None:
    """Every unresolved \\VAR must be on the missing-OK allow-list."""
    metrics = _fixture_metrics()
    # Mirror stages.paper: emit every table_* tabular into metrics so
    # \VAR{table_*} keys are populated before unresolved-collection.
    for k, v in inject_tables(metrics).items():
        metrics[k] = v
    expanded = expand_inputs(_TEMPLATE.read_text(encoding="utf-8"), _TEMPLATE.parent)
    unresolved = collect_unresolved(expanded, metrics)
    blockers = [k for k in unresolved if not is_missing_ok(k.split(":", 1)[0])]
    assert blockers == [], (
        f"canonical_347 advanced fixture has {len(blockers)} unresolved blocker keys: "
        f"{blockers[:10]}"
    )
