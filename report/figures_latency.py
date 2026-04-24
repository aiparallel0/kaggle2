"""Latency violin + throughput + Pareto figure.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_latency.pdf`` from ``metrics/latency_<system>.json``
    — a 1×3 panel: box/violin of p50/p95/p99 per system, a bar-chart
    of batch1 vs batch8 throughput, and a scatter of F1-vs-p95-latency
    coloured by USD/img (the Pareto frontier the paper's Discussion
    cites).  Torch-free; uses :mod:`report.figures_common`.
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


def _collect_latency(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load every ``latency_<system>.json`` into ``{system: data}``."""
    out: dict[str, dict[str, Any]] = {}
    for root in (run_dir / "metrics", run_dir):
        if not root.is_dir():
            continue
        for p in root.glob("latency_*.json"):
            data = load_json(p)
            if data is not None:
                out[p.stem.removeprefix("latency_")] = data
    return out


def render_latency(run_dir: Path) -> Path | None:
    """Three-panel latency figure; skip if no latency sidecars."""
    if not HAS_MPL:
        return None
    set_paper_style()
    per_sys = _collect_latency(run_dir)
    if guard_empty(per_sys, "latency"):
        return None
    metrics = load_json(run_dir / "metrics" / "combined_metrics.json") or {}
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 0.35 * COL_DOUBLE))
    systems = list(per_sys.keys())
    # Panel 1: p50/p95/p99 bars per system.
    x = range(len(systems))
    p50 = [float(per_sys[s].get("p50_ms", 0.0) or 0.0) for s in systems]
    p95 = [float(per_sys[s].get("p95_ms", 0.0) or 0.0) for s in systems]
    p99 = [float(per_sys[s].get("p99_ms", 0.0) or 0.0) for s in systems]
    axes[0].bar([i - 0.25 for i in x], p50, width=0.2, color=PALETTE[0], label="p50")
    axes[0].bar(list(x), p95, width=0.2, color=PALETTE[2], label="p95")
    axes[0].bar([i + 0.25 for i in x], p99, width=0.2, color=PALETTE[6], label="p99")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(systems, rotation=20, ha="right")
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title("inference latency")
    axes[0].legend(frameon=False, loc="upper left")
    # Panel 2: throughput bars.
    b1 = [float(per_sys[s].get("throughput_batch1", 0.0) or 0.0) for s in systems]
    b8 = [float(per_sys[s].get("throughput_batch8", 0.0) or 0.0) for s in systems]
    axes[1].bar([i - 0.15 for i in x], b1, width=0.3, color=PALETTE[3], label="batch=1")
    axes[1].bar([i + 0.15 for i in x], b8, width=0.3, color=PALETTE[4], label="batch=8")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(systems, rotation=20, ha="right")
    axes[1].set_ylabel("throughput (img/s)")
    axes[1].set_title("throughput")
    axes[1].legend(frameon=False, loc="upper right")
    # Panel 3: F1 vs p95 Pareto frontier, one dot per system.
    for i, s in enumerate(systems):
        f1 = float(metrics.get(f"{s}_f1", 0.0) or 0.0)
        axes[2].scatter(p95[i], f1, s=48, color=PALETTE[i % len(PALETTE)],
                        edgecolor="black", linewidth=0.4, label=s)
    axes[2].set_xlabel("p95 latency (ms)")
    axes[2].set_ylabel("macro F1")
    axes[2].set_title("latency–quality Pareto")
    axes[2].grid(True, alpha=0.25, linewidth=0.4)
    axes[2].legend(frameon=False, loc="lower right", fontsize=6)
    return save_fig(fig, run_dir / "figures", "fig_latency")
