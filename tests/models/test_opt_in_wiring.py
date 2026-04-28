"""Smoke tests for the P1/P3/P4 opt-in wiring.

None of these tests run the actual training/eval (CPU-only CI).  They
confirm the modules import cleanly, the config flags round-trip, and
the orchestrator stage can be invoked without torch.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.config import load_config
from core.types import AblationReport, AblationRun


def test_config_roundtrip_flags() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(str(repo / "configs/default.json"))
    assert cfg.rag_enabled is False
    assert cfg.gat_enabled is False
    assert cfg.foundation_enabled is False
    assert len(cfg.bug_flags) == 17  # PR-C added bugs 14–17
    assert all(cfg.bug_flags[f"bug_{i}"] for i in range(1, 18))


def test_ablation_types_construct() -> None:
    r = AblationRun(
        run_id="x", bug_id="all_on", seed=0, f1=0.8, ned=0.1, em=0.7,
    )
    rep = AblationReport(baseline_f1=0.8, runs=[r])
    assert rep.baseline_f1 == 0.8
    assert rep.runs[0].bug_id == "all_on"
    assert isinstance(rep.per_bug_delta, dict)
    assert isinstance(rep.interaction, dict)


def test_gat_assigner_importable() -> None:
    # Must import even if torch is missing — AssignerInput/FieldAssignment
    # dataclasses are declared under TYPE_CHECKING guards on torch.
    from models.focus_gat import AssignerInput, FieldAssignment, gat_assign
    assert AssignerInput is not None
    assert FieldAssignment is not None
    assert callable(gat_assign)


def test_bug_timeline_v2_schema(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    data = json.loads((repo / "results" / "bug_timeline.json").read_text())
    assert data["schema_version"] >= 2
    for bug in data["bugs"]:
        assert "f1_delta_measured" in bug
        assert bug["f1_delta_measured"] == bug["f1_before"]  # alias preserved
        assert "ci_low" in bug and "ci_high" in bug


def test_ablate_bugs_stage_importable() -> None:
    from stages.ablate_bugs import ablate_bugs, stage_ablate_bugs
    assert callable(ablate_bugs)
    assert callable(stage_ablate_bugs)
