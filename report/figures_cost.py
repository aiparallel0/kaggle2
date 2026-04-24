"""Cost + energy figure: USD/Wh per stage + cost-vs-F1 Pareto.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_cost.pdf`` from ``metrics/cost_*.json`` and
    ``metrics/combined_metrics.json``.  Two panels: per-stage
    USD/Wh stacked bars, and a cost-vs-F1 Pareto scatter.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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


def _collect_costs(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for root in (run_dir / "metrics", run_dir):
        if not root.is_dir():
            continue
        for p in root.glob("cost_*.json"):
            data = load_json(p)
            if data is not None:
                out[p.stem.removeprefix("cost_")] = data
    return out


def render_cost(run_dir: Path) -> Path | None:
    """Two-panel cost figure; skip when no ``cost_*.json`` sidecars exist."""
    if not HAS_MPL:
        return None
    set_paper_style()
    per_stage = _collect_costs(run_dir)
    if guard_empty(per_stage, "cost"):
        return None
    metrics = load_json(run_dir / "metrics" / "combined_metrics.json") or {}
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 0.4 * COL_DOUBLE))
    stages = list(per_stage.keys())
    usd = [float(per_stage[s].get("usd", 0.0) or 0.0) for s in stages]
    wh = [float(per_stage[s].get("energy_wh", 0.0) or 0.0) for s in stages]
    x = list(range(len(stages)))
    axes[0].bar([i - 0.18 for i in x], usd, width=0.35,
                color=PALETTE[0], label="USD")
    ax2 = axes[0].twinx()
    ax2.bar([i + 0.18 for i in x], wh, width=0.35,
            color=PALETTE[2], label="Wh")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, rotation=20, ha="right")
    axes[0].set_ylabel("USD")
    ax2.set_ylabel("energy (Wh)")
    axes[0].set_title("per-stage cost + energy")
    axes[0].spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # Panel 2: cost-vs-F1 Pareto — one point per system label.
    for i, system in enumerate(("donut", "pipeline", "rulebased")):
        f1 = float(metrics.get(f"{system}_f1", 0.0) or 0.0)
        cost = float(metrics.get(f"{system}_usd", 0.0) or 0.0)
        if f1 == 0.0 and cost == 0.0:
            continue
        axes[1].scatter(cost, f1, s=64, color=PALETTE[i],
                        edgecolor="black", linewidth=0.4, label=system)
    axes[1].set_xlabel("total USD")
    axes[1].set_ylabel("macro F1")
    axes[1].set_title("cost–quality Pareto")
    axes[1].grid(True, alpha=0.25, linewidth=0.4)
    axes[1].legend(frameon=False, loc="lower right", fontsize=6)
    return save_fig(fig, run_dir / "figures", "fig_cost")
