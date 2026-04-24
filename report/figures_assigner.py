"""Attention-assigner diagnostics figure — entropy, top-k, ECE.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_assigner.pdf`` from
    ``metrics/assigner_diagnostics.json``.  Three panels:
    per-field attention-entropy bars, top-1/3/5 accuracy, and
    the ECE / MCE / Brier triple as a small headline table.
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


def render_assigner(run_dir: Path) -> Path | None:
    """Three-panel assigner diagnostics figure."""
    if not HAS_MPL:
        return None
    set_paper_style()
    diag = load_json(run_dir / "metrics" / "assigner_diagnostics.json")
    if diag is None:
        diag = load_json(run_dir / "assigner_diagnostics.json")
    if guard_empty(diag, "assigner"):
        return None
    assert diag is not None
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 0.35 * COL_DOUBLE))
    # Panel 1: per-field attention entropy.
    ent_raw = diag.get("entropy_per_field")
    ent = ent_raw if isinstance(ent_raw, dict) else {}
    if ent:
        names = list(ent.keys())
        axes[0].bar(
            names, [float(ent[n]) for n in names],
            color=PALETTE[0], edgecolor="black", linewidth=0.4,
        )
        axes[0].set_ylabel("attention entropy (bits)")
        axes[0].set_title("per-field attention entropy")
        axes[0].tick_params(axis="x", labelrotation=20)
        axes[0].grid(axis="y", alpha=0.25, linewidth=0.4)
    else:
        axes[0].set_axis_off()
    # Panel 2: top-k accuracy.
    axes[1].bar(
        ["top-1", "top-3", "top-5"],
        [float(diag.get("top1_acc", 0.0) or 0.0),
         float(diag.get("top3_acc", 0.0) or 0.0),
         float(diag.get("top5_acc", 0.0) or 0.0)],
        color=[PALETTE[0], PALETTE[2], PALETTE[3]],
        edgecolor="black", linewidth=0.4,
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("top-k assignment accuracy")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.4)
    # Panel 3: ECE/MCE/Brier headline bars.
    axes[2].bar(
        ["ECE", "MCE", "Brier"],
        [float(diag.get("ece", 0.0) or 0.0),
         float(diag.get("mce", 0.0) or 0.0),
         float(diag.get("brier", 0.0) or 0.0)],
        color=[PALETTE[5], PALETTE[6], PALETTE[7]],
        edgecolor="black", linewidth=0.4,
    )
    axes[2].set_ylabel("error")
    axes[2].set_title("calibration error")
    axes[2].grid(axis="y", alpha=0.25, linewidth=0.4)
    return save_fig(fig, run_dir / "figures", "fig_assigner")
