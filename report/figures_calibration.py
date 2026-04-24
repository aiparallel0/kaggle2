"""Reliability-diagram calibration figure — one panel per field.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_calibration.pdf`` from
    ``metrics/assigner_diagnostics.json``.  Four sub-panels (one per
    field) showing calibration (confidence-vs-accuracy binned) with
    the ECE value annotated.  Degrades gracefully when the diagnostic
    sidecar is missing or lacks the per-field reliability bins.
"""
from __future__ import annotations

import logging
from pathlib import Path

from report.figures_common import (
    COL_DOUBLE,
    HAS_MPL,
    PALETTE,
    guard_empty,
    load_json,
    plt,
    save_fig,
    set_paper_style,
)

log = logging.getLogger("kaggle2")

_FIELDS = ("company", "date", "address", "total")


def _bins_from_diag(diag: dict[str, object], field: str) -> list[tuple[float, float, int]]:
    """Return ``[(avg_conf, accuracy, count), ...]`` for a field."""
    raw = diag.get(f"reliability_bins_{field}")
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float, int]] = []
    for entry in raw:
        if not isinstance(entry, list | tuple) or len(entry) < 3:
            continue
        try:
            out.append((float(entry[0]), float(entry[1]), int(entry[2])))
        except (TypeError, ValueError):
            continue
    return out


def render_calibration(run_dir: Path) -> Path | None:
    """Four-panel reliability diagram; skip if no assigner diagnostics."""
    if not HAS_MPL:
        return None
    set_paper_style()
    diag = load_json(run_dir / "metrics" / "assigner_diagnostics.json")
    if diag is None:
        diag = load_json(run_dir / "assigner_diagnostics.json")
    if guard_empty(diag, "calibration"):
        return None
    assert diag is not None
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 0.55 * COL_DOUBLE))
    any_bins = False
    ece_val = diag.get("ece")
    for ax, field in zip(axes.flat, _FIELDS, strict=True):
        bins = _bins_from_diag(diag, field)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.6)
        if bins:
            any_bins = True
            xs = [b[0] for b in bins]
            ys = [b[1] for b in bins]
            sizes = [max(6, b[2] / 4) for b in bins]
            ax.scatter(xs, ys, s=sizes, color=PALETTE[0], edgecolor="black",
                       linewidth=0.4, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(field)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.25, linewidth=0.4)
    if isinstance(ece_val, int | float):
        fig.suptitle(f"Reliability diagrams (ECE={float(ece_val):.3f})", y=1.02)
    else:
        fig.suptitle("Reliability diagrams", y=1.02)
    if not any_bins:
        log.info("figures_calibration: no reliability bins in diagnostics — empty panels")
    return save_fig(fig, run_dir / "figures", "fig_calibration")
