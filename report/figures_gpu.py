"""GPU telemetry time-series figure — util / VRAM / temperature / power.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_gpu_series.pdf`` from ``curves/gpu_util.csv``,
    ``curves/gpu_vram.csv``, ``curves/gpu_temp.csv``,
    ``curves/gpu_power.csv`` — the four scalars the pynvml-backed
    tracker emits per training step.  Four-panel layout, each panel
    a simple time series.  Any missing CSV yields an empty panel
    with an explanatory annotation.
"""
from __future__ import annotations

import logging
from pathlib import Path

from report.figures_common import (
    COL_DOUBLE,
    HAS_MPL,
    PALETTE,
    guard_empty,
    load_csv,
    plt,
    save_fig,
    set_paper_style,
)

log = logging.getLogger("kaggle2")


def _plot_axis(
    ax: object, rows: list[dict[str, float]], color: str,
    ylabel: str, title: str,
) -> None:
    if not rows or not HAS_MPL:
        ax.text(0.5, 0.5, f"no {title} data", ha="center", va="center",  # type: ignore[attr-defined]
                transform=ax.transAxes, fontsize=7, color="gray")  # type: ignore[attr-defined]
        ax.set_axis_off()  # type: ignore[attr-defined]
        return
    xs = [r.get("step", 0.0) for r in rows]
    ys = [r.get("value", 0.0) for r in rows]
    ax.plot(xs, ys, color=color, linewidth=0.9)  # type: ignore[attr-defined]
    ax.set_ylabel(ylabel)  # type: ignore[attr-defined]
    ax.set_title(title)  # type: ignore[attr-defined]
    ax.set_xlabel("step")  # type: ignore[attr-defined]
    ax.grid(True, alpha=0.25, linewidth=0.4)  # type: ignore[attr-defined]


def render_gpu_series(run_dir: Path) -> Path | None:
    """Four-panel GPU telemetry figure; skip when no GPU CSVs exist."""
    if not HAS_MPL:
        return None
    set_paper_style()
    curves = run_dir / "curves"
    util = load_csv(curves / "gpu_util.csv")
    vram = load_csv(curves / "gpu_vram.csv")
    temp = load_csv(curves / "gpu_temp.csv")
    power = load_csv(curves / "gpu_power.csv")
    if guard_empty(util + vram + temp + power, "gpu_series"):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 0.55 * COL_DOUBLE))
    _plot_axis(axes[0, 0], util, PALETTE[0], "%", "GPU utilisation")
    _plot_axis(axes[0, 1], vram, PALETTE[2], "MiB", "VRAM used")
    _plot_axis(axes[1, 0], temp, PALETTE[6], "°C", "temperature")
    _plot_axis(axes[1, 1], power, PALETTE[3], "W", "power draw")
    fig.suptitle("GPU telemetry during training", y=1.02)
    return save_fig(fig, run_dir / "figures", "fig_gpu_series")
