"""Tests for the v4 LaTeX-engine hardening.

Pin the new contracts that v4 introduces:

  * :mod:`report.missing` allow-list / strict-mode logic
  * :mod:`report.check_artefacts` build-gate detection
  * :mod:`report.best_epoch` per-stage best-epoch surfacing
  * Training-table per-stage rows resolve to real values when the
    artefacts are present rather than ``\\textit{n/a}``
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from report.best_epoch import (
    _read_donut_best,
    _read_trocr_best,
    _read_yolo_best,
    merge_best_epochs,
)
from report.check_artefacts import scan_figures_dir, scan_paper
from report.missing import (
    MISSING_OK_KEYS,
    filter_blockers,
    is_missing_ok,
    render_missing_cell,
)


def test_render_missing_cell_emits_typed_marker() -> None:
    # Underscores must be escaped as \_ so LaTeX text mode does not treat
    # them as math-mode subscript operators (Missing $ inserted).
    out = render_missing_cell("donut_f1")
    assert out == "\\MissingCell{donut\\_f1}"


def test_render_missing_cell_sanitises_key() -> None:
    """LaTeX-unsafe characters in the key get sanitised; underscores are escaped as ``\\_``."""
    out = render_missing_cell("foo bar$\\baz")
    assert "$" not in out
    assert out.startswith("\\MissingCell{")
    assert out.endswith("}")
    # Only escape sequences (\_ for underscores) should appear inside the arg —
    # raw unescaped underscores must not remain.
    inner = out[len("\\MissingCell{"):-1]
    assert "_" not in inner.replace("\\_", "")


def test_missing_ok_allow_list() -> None:
    """Allow-listed keys do not count as build blockers."""
    assert is_missing_ok("donut_f1_std")
    assert is_missing_ok("foundation_em_company")
    assert is_missing_ok("rag_on_f1")
    assert is_missing_ok("gat_assigner_f1")
    assert not is_missing_ok("donut_f1")
    assert not is_missing_ok("pipeline_f1_company")


def test_filter_blockers_strips_allow_listed() -> None:
    unresolved = [
        "donut_f1", "donut_f1_std", "rag_on_f1", "pipeline_f1",
        "foundation_em_address",
    ]
    blockers = filter_blockers(unresolved)
    assert blockers == ["donut_f1", "pipeline_f1"]


def test_missing_ok_keys_is_frozen() -> None:
    """Allow-list is immutable so accidental mutation can't widen it."""
    assert isinstance(MISSING_OK_KEYS, frozenset)


def test_check_artefacts_detects_unresolved_VAR(tmp_path: Path) -> None:
    paper = tmp_path / "paper_filled.tex"
    paper.write_text(
        "\\documentclass{article}\n"
        "Headline F1: \\VAR{donut_f1}\n"
        "Allow-listed: \\VAR{donut_f1_std}\n"
        "End.\n"
    )
    findings = scan_paper(paper)
    # ``donut_f1`` is a blocker; ``donut_f1_std`` is on the allow-list.
    assert findings.get("unresolved_VAR") == ["donut_f1"]


def test_check_artefacts_detects_dangling_refs(tmp_path: Path) -> None:
    paper = tmp_path / "p.tex"
    paper.write_text("see Sec.~?? for details. and Fig.~?? too.")
    findings = scan_paper(paper)
    assert "dangling_refs" in findings


def test_check_artefacts_detects_dangling_citations(tmp_path: Path) -> None:
    paper = tmp_path / "p.tex"
    paper.write_text("recent work [?] showed ...")
    findings = scan_paper(paper)
    assert "dangling_citations" in findings


def test_check_artefacts_clean_paper(tmp_path: Path) -> None:
    paper = tmp_path / "p.tex"
    paper.write_text(
        "All resolved. F1=0.85. See Sec.~5. cite~[1]."
    )
    findings = scan_paper(paper)
    assert findings == {}


def test_check_artefacts_finds_empty_pdfs(tmp_path: Path) -> None:
    figs = tmp_path / "figures"
    figs.mkdir()
    (figs / "good.pdf").write_bytes(b"%PDF-1.4 nontrivial")
    (figs / "empty.pdf").write_bytes(b"")
    out = scan_figures_dir(figs)
    assert out == ["empty.pdf"]


def test_best_epoch_yolo_from_results_csv(tmp_path: Path) -> None:
    """``_read_yolo_best`` recovers best epoch from Ultralytics csv."""
    yolo_dir = tmp_path / "yolo" / "run"
    yolo_dir.mkdir(parents=True)
    csv_path = yolo_dir / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)"])
        w.writerow([0, 0.20, 0.50])
        w.writerow([1, 0.55, 0.80])  # best
        w.writerow([2, 0.45, 0.75])
    best, total = _read_yolo_best(tmp_path)
    assert best == 2  # 1-indexed for paper presentation
    assert total == 3


def test_best_epoch_yolo_missing_csv(tmp_path: Path) -> None:
    best, total = _read_yolo_best(tmp_path)
    assert best is None
    assert total is None


def test_best_epoch_trocr_from_trainer_state(tmp_path: Path) -> None:
    """``_read_trocr_best`` finds best epoch in HF Trainer state."""
    trocr_dir = tmp_path / "trocr"
    trocr_dir.mkdir()
    state = {
        "epoch": 8.0,
        "best_metric": 0.91,
        "log_history": [
            {"epoch": 1.0, "eval_crop_cer_f1": 0.50},
            {"epoch": 5.0, "eval_crop_cer_f1": 0.91},  # best
            {"epoch": 8.0, "eval_crop_cer_f1": 0.85},
        ],
    }
    (trocr_dir / "trainer_state.json").write_text(json.dumps(state))
    best, total = _read_trocr_best(tmp_path)
    assert best == 5
    assert total == 8


def test_best_epoch_donut_from_training_log(tmp_path: Path) -> None:
    """``_read_donut_best`` argmaxes eval_f1 across the training log."""
    log = {
        "epochs": [1, 2, 3, 4, 5],
        "eval_f1": [0.40, 0.55, 0.70, 0.78, 0.72],  # best at epoch 4
        "eval_loss": [3.0, 2.0, 1.5, 1.0, 1.2],
    }
    (tmp_path / "training_log.json").write_text(json.dumps(log))
    best, total = _read_donut_best(tmp_path)
    assert best == 4
    assert total == 5


def test_merge_best_epochs_emits_pipeline_label(tmp_path: Path) -> None:
    """End-to-end: merge populates per-stage keys + pipeline label."""
    from core.types import ExpConfig
    cfg = ExpConfig(
        seed=42, base_model="x", trocr_model="x", yolo_model="x",
        image_size=(640, 640), yolo_img_size=640, max_length=8,
        trocr_max_len=8, epochs_donut=1, epochs_yolo=1,
        epochs_trocr=1, epochs_assigner=1, batch_size=1, grad_accum=1,
        lr=1e-3, lr_decoder=1e-3, warmup_steps=0, weight_decay=0.0,
        label_smoothing=0.0, precision="bf16", patience=1,
        max_grad_norm=1.0, fields=[], new_tokens=[], sroie_url="",
        data_dir=str(tmp_path), output_dir=str(tmp_path),
        paper_template="x", paper_output="y",
    )
    # Set up only the assigner-stopped indicator so the pipeline-label
    # path runs even without yolo/trocr/donut artefacts.
    metrics: dict[str, object] = {"assigner_stopped_at": 12}
    merge_best_epochs(cfg, metrics)
    assert metrics.get("assigner_epochs_run") == 12
    assert metrics.get("pipeline_best_epoch_label") == "\\textit{see sub-stages}"


def test_training_table_resolves_per_stage_best_epochs() -> None:
    """v4 fix: per-stage best-epoch cells render the real value."""
    from report.inject_tables import render_training_table
    metrics: dict[str, object] = {
        "donut_epochs": 30, "donut_best_epoch": 18,
        "yolo_epochs_run": 50, "yolo_best_epoch": 32,
        "trocr_epochs_run": 10, "trocr_best_epoch": 7,
        "assigner_epochs_run": 25, "assigner_best_epoch": 14,
        "pipeline_epochs": 85,
        "pipeline_best_epoch_label": "\\textit{see sub-stages}",
    }
    out = render_training_table(metrics)
    # Each stage's best/total epoch is in the table.
    for v in (30, 18, 50, 32, 10, 7, 25, 14, 85):
        assert str(v) in out, f"missing epoch {v} in training table"
    # Pipeline best-epoch row carries the composite label.
    assert "see sub-stages" in out
    # No silent em-dashes anywhere.
    assert "---" not in out


def test_mean_std_directive_renders_pm() -> None:
    """v4 inject directive: ``mean_std_pct1`` renders ``X ± Y``."""
    from report.inject import inject_results
    template = "F1 = \\VAR{donut_f1:mean_std_pct1}"
    out = inject_results(template, {"donut_f1_mean": 0.852, "donut_f1_std": 0.007})
    assert "85.2\\%" in out
    assert "\\pm" in out
    assert "0.7\\%" in out


def test_mean_std_directive_collapses_when_std_zero() -> None:
    """When the std is zero (n=1 run), the mean alone is rendered.

    Single-seed runs do not have inter-seed variance to report; the
    directive degrades gracefully so n=1 PDFs read naturally instead
    of always carrying a meaningless ``± 0.0\\%`` tail.
    """
    from report.inject import inject_results
    template = "F1 = \\VAR{donut_f1:mean_std_pct1}"
    out = inject_results(template, {"donut_f1_mean": 0.852, "donut_f1_std": 0.0})
    assert "85.2\\%" in out
    assert "\\pm" not in out


def test_mean_std_directive_unresolved_when_mean_missing() -> None:
    """Directive-format keys (``key:directive``) are excluded from collect_unresolved.

    ``apply_formatters`` owns directive resolution; counting them in
    ``collect_unresolved`` before formatters run produces false positives.
    A template with only directive-format ``\\VAR{}`` keys reports zero
    unresolved even when the underlying metric keys are absent.
    """
    from report.inject import collect_unresolved
    template = "F1 = \\VAR{donut_f1:mean_std_pct1}"
    unresolved = collect_unresolved(template, {})
    assert unresolved == []
