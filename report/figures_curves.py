"""Training-curve multi-panel figures — loss / LR / grad-norm / throughput.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_<model>_curves.pdf`` from ``curves/*.csv`` emitted
    by :class:`core.tracking.Tracker`.  Each figure is a 2×2 multi-
    panel with a best-checkpoint vertical line annotation.  Never
    raises: absent CSVs → a warning log and a skipped panel.
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


def _plot_series(ax: object, rows: list[dict[str, float]], color: str, label: str) -> None:
    if not rows or not HAS_MPL:
        return
    xs = [r.get("step", 0.0) for r in rows]
    ys = [r.get("value", 0.0) for r in rows]
    ax.plot(xs, ys, color=color, linewidth=1.2, label=label)  # type: ignore[attr-defined]


def render_training_curves(run_dir: Path, model: str) -> Path | None:
    """Four-panel figure for ``<run_dir>/curves/<model>_{loss,lr,gradnorm,tput}.csv``."""
    if not HAS_MPL:
        return None
    set_paper_style()
    curves_dir = run_dir / "curves"
    loss = load_csv(curves_dir / f"{model}_loss.csv")
    lr = load_csv(curves_dir / f"{model}_lr.csv")
    gn = load_csv(curves_dir / f"{model}_gradnorm.csv")
    tput = load_csv(curves_dir / f"{model}_tput.csv")
    if guard_empty(loss + lr + gn + tput, f"{model}_curves"):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 0.55 * COL_DOUBLE))
    _plot_series(axes[0, 0], loss, PALETTE[0], "loss")
    axes[0, 0].set_title("training loss")
    axes[0, 0].set_xlabel("step")
    _plot_series(axes[0, 1], lr, PALETTE[1], "lr")
    axes[0, 1].set_title("learning rate")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("step")
    _plot_series(axes[1, 0], gn, PALETTE[2], "grad-norm")
    axes[1, 0].set_title("gradient norm (post-clip)")
    axes[1, 0].set_xlabel("step")
    _plot_series(axes[1, 1], tput, PALETTE[3], "samples/s")
    axes[1, 1].set_title("throughput")
    axes[1, 1].set_xlabel("step")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25, linewidth=0.4)
    fig.suptitle(f"{model.upper()} training diagnostics", y=1.02)
    return save_fig(fig, run_dir / "figures", f"fig_{model}_curves")


def render_all_curves(run_dir: Path) -> list[Path]:
    """Render one figure per model that has any CSV evidence under curves/."""
    out: list[Path] = []
    for model in ("donut", "yolo", "trocr", "assigner"):
        p = render_training_curves(run_dir, model)
        if p is not None:
            out.append(p)
    return out
