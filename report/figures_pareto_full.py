"""PR-E — Full Pareto sweep figure (assigner_size × dataset × seed).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: aggregates the 24-cell sweep (4 sizes × 2 datasets × 3 seeds) into
    an F1-vs-log(params) curve so reviewers can read off the
    saturation point.  Sourced from ``runs/_sweep_aggregate.json``
    written by :mod:`stages.sweep`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("kaggle2")


def emit_pareto_full_figure(
    config: ExpConfig, sweep_summary: Path | None = None,
) -> Path | None:
    """Emit ``figures/fig_pareto_full.pdf`` from the sweep aggregate."""
    src = sweep_summary or Path("runs/_sweep_aggregate.json")
    if not src.is_file():
        return None
    try:
        with src.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("figures_pareto_full: cannot read %s (%s)", src, exc)
        return None
    rows = data.get("rows")
    if not isinstance(rows, list):
        return None
    out_path = Path(config.output_dir) / "figures" / "fig_pareto_full.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _emit_curve(rows, out_path):
        return None
    return out_path


def _emit_curve(rows: list[object], out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    by_dataset: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        ds = str(r.get("dataset", "?"))
        params = r.get("params_m")
        f1 = r.get("f1_mean")
        if isinstance(params, int | float) and isinstance(f1, int | float):
            by_dataset.setdefault(ds, []).append(
                (float(params), float(f1)),
            )
    if not by_dataset:
        return False
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for ds, pts in by_dataset.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=ds)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Macro-F1")
    ax.set_xscale("log")
    ax.grid(visible=True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True
