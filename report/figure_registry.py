"""Figure registry — central index of every emitter the paper may cite.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: provide the LaTeX injector a ``figure_exists(stem) -> bool``
    check so the ``\\iffigure{name}{yes}{fallback}`` macro can
    gracefully degrade when a figure couldn't be rendered (usually
    because the source data was absent on a partial run).  Also
    enumerates the expected figure catalogue for MANIFEST cross-
    reference and for ``tests/test_figures_new.py`` to iterate over.

Every entry is ``(stem, source_description)``; the figure emitters
themselves decide whether to actually produce output given the data
available at paper-stage start.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FigureSpec:
    """One expected figure and a one-liner describing its data source."""

    stem: str
    module: str
    source: str


FIGURE_CATALOG: tuple[FigureSpec, ...] = (
    FigureSpec("fig_training_curves", "figures",
               "training_log.json per epoch losses"),
    FigureSpec("fig_gpu_telemetry", "figures",
               "telemetry_donut.jsonl GPU util/vram/temp"),
    FigureSpec("fig_per_field_confusion", "figures",
               "combined_metrics.json per-field F1"),
    FigureSpec("fig_attention_heatmap", "figures_attn",
               "attention_samples.npz cross-attention"),
    FigureSpec("fig_bug_timeline", "figures_bugs",
               "bug_timeline.json F1 before/after"),
    FigureSpec("fig_pareto", "figures_extra",
               "combined_metrics.json Pareto frontier"),
    FigureSpec("fig_donut_curves", "figures_curves",
               "curves/donut_loss.csv + donut_lr.csv"),
    FigureSpec("fig_yolo_curves", "figures_curves",
               "curves/yolo_map.csv"),
    FigureSpec("fig_f1_grouped", "figures_f1",
               "combined_metrics.json per-field F1 + CI"),
    FigureSpec("fig_calibration", "figures_calibration",
               "assigner_diagnostics.json ECE bins"),
    FigureSpec("fig_latency", "figures_latency",
               "metrics/latency_*.json percentile burst"),
    FigureSpec("fig_cost", "figures_cost",
               "metrics/cost_*.json USD/Wh per stage"),
    FigureSpec("fig_errors", "figures_errors",
               "per_field_errors.jsonl 8-category breakdown"),
    FigureSpec("fig_yolo_pr", "figures_yolo",
               "yolo_metrics.json PR curve + per-class AP"),
    FigureSpec("fig_trocr", "figures_trocr",
               "trocr_metrics.json CER/WER scatter"),
    FigureSpec("fig_assigner", "figures_assigner",
               "assigner_diagnostics.json entropy/top-k"),
    FigureSpec("fig_gpu_series", "figures_gpu",
               "curves/gpu_util.csv time series"),
    FigureSpec("fig_samples", "figures_samples",
               "predictions/*.jsonl 12-receipt qualitative grid"),
)


def figure_exists(figures_dir: Path, stem: str) -> bool:
    """Return True iff ``<figures_dir>/<stem>.pdf`` exists and is non-empty."""
    p = figures_dir / f"{stem}.pdf"
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def registry_summary(figures_dir: Path) -> dict[str, bool]:
    """Return ``{stem: exists}`` for every entry in FIGURE_CATALOG."""
    return {spec.stem: figure_exists(figures_dir, spec.stem) for spec in FIGURE_CATALOG}


def expected_stems() -> tuple[str, ...]:
    """Tuple of every stem in the catalogue — used by hygiene tests."""
    return tuple(spec.stem for spec in FIGURE_CATALOG)
