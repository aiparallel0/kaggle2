"""Per-field F1 grouped bars with bootstrap-CI whiskers + significance stars.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_f1_grouped.pdf`` from ``combined_metrics.json`` —
    a single grouped-bar chart with three bars per field (DONUT,
    Pipeline, Rule-baseline), 95% bootstrap CI whiskers, and
    significance stars derived from the paired-bootstrap p-value
    keys (``p_donut_vs_pipeline_<field>``).  Never raises.
"""
from __future__ import annotations

import logging
from pathlib import Path

from report.figures_common import (
    COL_SINGLE,
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
_MODELS = (
    ("donut", "DONUT", PALETTE[0]),
    ("pipeline", "Pipeline", PALETTE[2]),
    ("rulebased", "Rule-based", PALETTE[5]),
)


def _star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def render_f1_grouped(run_dir: Path) -> Path | None:
    """Grouped-bars figure with CI whiskers + significance stars."""
    if not HAS_MPL:
        return None
    set_paper_style()
    metrics = load_json(run_dir / "metrics" / "combined_metrics.json")
    if metrics is None:
        metrics = load_json(run_dir / "combined_metrics.json")
    if guard_empty(metrics, "f1_grouped"):
        return None
    assert metrics is not None
    fig, ax = plt.subplots(figsize=(COL_SINGLE * 1.3, COL_SINGLE * 0.8))
    n_fields = len(_FIELDS)
    group_w = 0.26
    for mi, (prefix, label, color) in enumerate(_MODELS):
        xs = [i + (mi - 1) * group_w for i in range(n_fields)]
        ys = [float(metrics.get(f"{prefix}_f1_{f}", 0.0) or 0.0) for f in _FIELDS]
        los = [float(metrics.get(f"f1_{f}_ci_lo", 0.0) or 0.0) for f in _FIELDS]
        his = [float(metrics.get(f"f1_{f}_ci_hi", 0.0) or 0.0) for f in _FIELDS]
        yerr_lo = [max(0.0, y - lo) for y, lo in zip(ys, los, strict=True)]
        yerr_hi = [max(0.0, hi - y) for y, hi in zip(ys, his, strict=True)]
        ax.bar(xs, ys, width=group_w, color=color, label=label,
               edgecolor="black", linewidth=0.4)
        ax.errorbar(
            xs, ys, yerr=[yerr_lo, yerr_hi],
            fmt="none", ecolor="black", capsize=1.6, linewidth=0.5,
        )
    # Significance stars above DONUT bars using paired-bootstrap p-values.
    for fi, field in enumerate(_FIELDS):
        p = metrics.get(f"p_donut_vs_pipeline_{field}")
        if isinstance(p, int | float):
            star = _star(float(p))
            if star:
                y = float(metrics.get(f"donut_f1_{field}", 0.0) or 0.0)
                ax.text(fi - group_w, y + 0.02, star, ha="center", fontsize=8)
    ax.set_xticks(range(n_fields))
    ax.set_xticklabels(_FIELDS)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Token F1")
    ax.set_title("Per-field F1 with 95% bootstrap CI")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    return save_fig(fig, run_dir / "figures", "fig_f1_grouped")
