"""Integration tests that render actual PDFs for the new figure emitters.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: drive every ``render_*`` in :mod:`report.figures_extra`,
    :mod:`report.figures_bugs`, and :mod:`report.figures_attn`
    against a synthetic artefact directory, confirming that (i)
    present sources produce non-empty PDFs and (ii) absent sources
    degrade to a :class:`UserWarning` + ``None`` return (the
    best-effort contract the paper stage relies on).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest


@pytest.fixture
def synthetic_results_dir(tmp_path: Path) -> Path:
    """Populate ``tmp_path`` with minimal artefacts the emitters consume."""
    (tmp_path / "combined_metrics.json").write_text(json.dumps({
        "donut_f1": 0.78, "pipeline_f1": 0.74, "rulebased_f1": 0.54,
        "donut_f1_company": 0.92, "donut_f1_date": 0.95,
        "donut_f1_address": 0.55, "donut_f1_total": 0.70,
        "pipeline_f1_company": 0.90, "pipeline_f1_date": 0.94,
        "pipeline_f1_address": 0.48, "pipeline_f1_total": 0.64,
        "rulebased_f1_company": 0.88, "rulebased_f1_date": 0.85,
        "rulebased_f1_address": 0.00, "rulebased_f1_total": 0.42,
    }))
    (tmp_path / "pipeline_metrics.json").write_text(json.dumps({
        "empty_detection_fraction": 0.032,
        "per_receipt_error_fraction": 0.016,
    }))
    (tmp_path / "assigner_metrics.json").write_text(json.dumps({
        "train_loss": [1.20, 0.95, 0.72, 0.55, 0.41],
        "val_loss": [1.30, 1.01, 0.80, 0.66, 0.60],
        "best_epoch": 4, "stopped_at_epoch": 5, "best_val_loss": 0.60,
        "n_params": 380_000,
    }))
    (tmp_path / "telemetry_donut.jsonl").write_text(
        "\n".join(json.dumps({"ts": 1.0 + i, "gpu_util_pct": 70 + i})
                 for i in range(10)),
    )
    # Overlay figure is, by name, an *overlay* — both traces are
    # required (review item S3).  Earlier fixtures provided only the
    # DONUT trace, which silently produced a one-line "overlay"; the
    # renderer now refuses, so the test fixture must include both.
    (tmp_path / "telemetry_pipeline.jsonl").write_text(
        "\n".join(json.dumps({"ts": 1.0 + i, "gpu_util_pct": 40 + i})
                 for i in range(10)),
    )
    repo = Path(__file__).resolve().parent.parent
    (tmp_path / "bug_timeline.json").write_text(
        (repo / "results" / "bug_timeline.json").read_text(),
    )
    return tmp_path


def test_figure_emitters_write_pdfs(synthetic_results_dir: Path) -> None:
    """With matplotlib available, every emitter writes a non-empty PDF."""
    pytest.importorskip("matplotlib")
    from report.figures_bugs import render_bug_timeline, render_telemetry_overlay
    from report.figures_extra import (
        render_assigner_loss_curve,
        render_f1_by_system,
        render_pipeline_diagnostics,
    )

    out = synthetic_results_dir
    emitted = [
        render_f1_by_system(str(out), str(out)),
        render_assigner_loss_curve(str(out), str(out)),
        render_pipeline_diagnostics(str(out), str(out)),
        render_bug_timeline(str(out), str(out)),
        render_telemetry_overlay(str(out), str(out)),
    ]
    assert all(e is not None for e in emitted)
    for path in emitted:
        assert path is not None
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_figure_emitters_skip_missing_sources(tmp_path: Path) -> None:
    """Every emitter returns ``None`` (and warns) on missing source JSON."""
    pytest.importorskip("matplotlib")
    from report.figures_bugs import render_bug_timeline, render_telemetry_overlay
    from report.figures_extra import (
        render_assigner_loss_curve,
        render_f1_by_system,
        render_pipeline_diagnostics,
    )

    for fn in (
        render_f1_by_system, render_assigner_loss_curve,
        render_pipeline_diagnostics, render_bug_timeline,
        render_telemetry_overlay,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert fn(str(tmp_path), str(tmp_path)) is None


def test_attention_heatmap_from_json_fallback(tmp_path: Path) -> None:
    """Heatmap emitter reads the ``.json`` fallback if ``.npz`` is absent."""
    pytest.importorskip("matplotlib")
    from report.figures_attn import render_attention_heatmap

    (tmp_path / "attention_samples.json").write_text(json.dumps({
        "image_paths": ["a.png", "b.png"],
        "attn": [
            [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1],
             [0.2, 0.3, 0.5], [0.33, 0.33, 0.34]],
            [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8],
             [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]],
        ],
    }))
    out = render_attention_heatmap(str(tmp_path), str(tmp_path))
    assert out is not None
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0
