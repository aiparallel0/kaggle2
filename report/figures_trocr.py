"""TrOCR diagnostics figure — CER/WER bars + per-field breakdown.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_trocr.pdf`` from ``metrics/trocr_metrics.json``
    (written by :mod:`core.metrics_trocr`).  Two panels: CER/WER mean
    headline bars, and per-field CER/WER grouped bars.
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


def render_trocr(run_dir: Path) -> Path | None:
    """Two-panel TrOCR diagnostics figure."""
    if not HAS_MPL:
        return None
    set_paper_style()
    diag = load_json(run_dir / "metrics" / "trocr_metrics.json")
    if diag is None:
        diag = load_json(run_dir / "trocr_metrics.json")
    if guard_empty(diag, "trocr"):
        return None
    assert diag is not None
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 0.4 * COL_DOUBLE))
    # Panel 1: headline CER/WER.
    axes[0].bar(
        ["CER mean", "CER total", "WER mean"],
        [float(diag.get("cer_mean", 0.0) or 0.0),
         float(diag.get("cer_total", 0.0) or 0.0),
         float(diag.get("wer_mean", 0.0) or 0.0)],
        color=[PALETTE[0], PALETTE[2], PALETTE[6]],
        edgecolor="black", linewidth=0.4,
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("error rate")
    axes[0].set_title("TrOCR headline")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.4)
    # Panel 2: per-field CER vs WER grouped bars.
    cer_pf_raw = diag.get("cer_per_field")
    wer_pf_raw = diag.get("wer_per_field")
    cer_pf = cer_pf_raw if isinstance(cer_pf_raw, dict) else {}
    wer_pf = wer_pf_raw if isinstance(wer_pf_raw, dict) else {}
    fields = sorted(set(cer_pf.keys()) | set(wer_pf.keys()))
    if fields:
        x = list(range(len(fields)))
        cer_vals = [float(cer_pf.get(f, 0.0) or 0.0) for f in fields]
        wer_vals = [float(wer_pf.get(f, 0.0) or 0.0) for f in fields]
        axes[1].bar([i - 0.18 for i in x], cer_vals, width=0.35,
                    color=PALETTE[0], label="CER", edgecolor="black", linewidth=0.3)
        axes[1].bar([i + 0.18 for i in x], wer_vals, width=0.35,
                    color=PALETTE[6], label="WER", edgecolor="black", linewidth=0.3)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(fields, rotation=20, ha="right")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("error rate")
        axes[1].set_title("per-field error rate")
        axes[1].legend(frameon=False, loc="upper right")
        axes[1].grid(axis="y", alpha=0.25, linewidth=0.4)
    else:
        axes[1].text(0.5, 0.5, "no per-field data", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=7, color="gray")
        axes[1].set_axis_off()
    return save_fig(fig, run_dir / "figures", "fig_trocr")
