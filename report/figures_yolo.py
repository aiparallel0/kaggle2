"""YOLO detection figure — mAP bars, per-class AP, PR curve.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_yolo_pr.pdf`` from
    ``metrics/yolo_metrics.json`` (the sidecar written by
    :mod:`core.metrics_yolo`).  Three-panel layout: headline mAP
    bars, per-class AP, PR curve.
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


def render_yolo(run_dir: Path) -> Path | None:
    """Three-panel YOLO diagnostics figure."""
    if not HAS_MPL:
        return None
    set_paper_style()
    diag = load_json(run_dir / "metrics" / "yolo_metrics.json")
    if diag is None:
        diag = load_json(run_dir / "yolo_metrics.json")
    if guard_empty(diag, "yolo"):
        return None
    assert diag is not None
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 0.35 * COL_DOUBLE))
    # Panel 1: headline mAP bars.
    axes[0].bar(
        ["mAP@.5", "mAP@.5:.95"],
        [float(diag.get("map50", 0.0) or 0.0),
         float(diag.get("map5095", 0.0) or 0.0)],
        color=[PALETTE[0], PALETTE[2]],
        edgecolor="black", linewidth=0.4,
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("mAP")
    axes[0].set_title("headline mAP")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.4)
    # Panel 2: per-class AP bars.
    pca_raw = diag.get("per_class_ap")
    if isinstance(pca_raw, dict) and pca_raw:
        names = list(pca_raw.keys())
        vals = [float(pca_raw[n]) for n in names]
        axes[1].bar(names, vals, color=PALETTE[3],
                    edgecolor="black", linewidth=0.4)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("AP@.5")
        axes[1].set_title("per-class AP")
        axes[1].tick_params(axis="x", labelrotation=20)
    else:
        axes[1].text(0.5, 0.5, "no per-class AP", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=7, color="gray")
        axes[1].set_axis_off()
    # Panel 3: PR curve.
    prec = diag.get("pr_curve_precision")
    rec = diag.get("pr_curve_recall")
    if (isinstance(prec, list) and isinstance(rec, list)
            and len(prec) == len(rec) and prec):
        axes[2].plot(rec, prec, color=PALETTE[6], linewidth=1.2)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("recall")
        axes[2].set_ylabel("precision")
        axes[2].set_title("PR curve")
        axes[2].grid(True, alpha=0.25, linewidth=0.4)
    else:
        axes[2].text(0.5, 0.5, "no PR curve", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=7, color="gray")
        axes[2].set_axis_off()
    return save_fig(fig, run_dir / "figures", "fig_yolo_pr")
