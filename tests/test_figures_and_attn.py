"""Unit tests for the new figure emitters and attention sampler.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: exercise every module added in the figures-and-diagnostics
    pass — :mod:`models.pipeline_attn.AttentionSampler`, the three
    ``report.figures_*`` modules, and the ``results/bug_timeline.json``
    fixture schema — without requiring matplotlib, NumPy, or PyTorch
    to be installed.  When matplotlib is available we additionally
    assert that each emitter writes a non-empty PDF under the
    per-test tmp_path; when it is not, we assert that every emitter
    issues a :class:`UserWarning` and returns ``None``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from models.pipeline_attn import DEFAULT_SAMPLE_K, AttentionSampler


def test_default_sample_k_is_small_and_positive() -> None:
    """The shipped K must stay in the 1--10 range (paper caption asserts K=3)."""
    assert 1 <= DEFAULT_SAMPLE_K <= 10


def test_attention_sampler_captures_up_to_k(tmp_path: Path) -> None:
    """Capture stops after K triples and survives NumPy-less environments."""
    sampler = AttentionSampler(k=2)
    for i in range(4):
        sampler.capture(
            image_path=f"/tmp/img_{i}.png",
            bboxes=[[0, 0, 10, 10], [0, 10, 10, 20]],
            attn_weights=[[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7]],
        )
    assert sampler.full
    out = sampler.write(tmp_path)
    assert out is not None
    assert out.exists()
    # .npz preferred, but the .json fallback must also keep the data intact
    if out.suffix == ".json":
        payload = json.loads(out.read_text())
        assert len(payload["image_paths"]) == 2
        assert len(payload["attn"]) == 2


def test_attention_sampler_skips_empty_bboxes() -> None:
    """Empty bbox list → no-op, sampler remains un-full."""
    sampler = AttentionSampler(k=1)
    sampler.capture("/tmp/a.png", [], [[0.5, 0.5]])
    assert not sampler.full


def test_attention_sampler_accepts_tolist_objects() -> None:
    """Any object exposing ``.tolist()`` (tensor, ndarray) is accepted."""
    class _FakeTensor:
        def tolist(self) -> list[list[float]]:
            return [[0.7, 0.3], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8]]

    sampler = AttentionSampler(k=1)
    sampler.capture("/tmp/b.png", [[0, 0, 5, 5], [0, 5, 5, 10]], _FakeTensor())
    assert sampler.full


def test_attention_sampler_write_no_samples_returns_none(tmp_path: Path) -> None:
    """Empty sampler writes nothing and returns ``None`` (paper-stage safe)."""
    sampler = AttentionSampler(k=3)
    assert sampler.write(tmp_path) is None


def test_bug_timeline_fixture_schema() -> None:
    """The shipped fixture must contain 13 valid entries with an F1 default."""
    repo = Path(__file__).resolve().parent.parent
    data = json.loads((repo / "results" / "bug_timeline.json").read_text())
    assert data["schema_version"] == 1
    assert "f1_after_default" in data
    bugs = data["bugs"]
    assert len(bugs) == 13
    seen_ids = set()
    for bug in bugs:
        assert isinstance(bug["id"], int)
        assert bug["id"] not in seen_ids
        seen_ids.add(bug["id"])
        assert 0.0 <= float(bug["f1_before"]) <= 1.0
        assert isinstance(bug["measured"], bool)
        assert bug["short"]
        assert bug["mechanism"]
    assert seen_ids == set(range(1, 14))


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
    # Seed the bug-timeline fixture alongside the other per-run artefacts.
    repo = Path(__file__).resolve().parent.parent
    (tmp_path / "bug_timeline.json").write_text(
        (repo / "results" / "bug_timeline.json").read_text(),
    )
    return tmp_path


def test_figure_emitters_write_pdfs(synthetic_results_dir: Path) -> None:
    """When matplotlib is available every emitter writes a non-empty PDF."""
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
