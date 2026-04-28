"""PR-D — F1-vs-params Pareto scatter for the competitor comparison.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: scatter of ``params_m`` vs ``global_f1`` for the headline four
    arms (DONUT / pipeline / LayoutLMv3 / GPT-4V) so reviewers can
    read the Pareto frontier at a glance.  Sourced exclusively from
    ``runs/<id>/combined_metrics.json``; never raises.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("kaggle2")

_SYSTEMS = (
    ("donut", "DONUT"),
    ("pipeline", "YOLO+TrOCR+Attn"),
    ("layoutlmv3", "LayoutLMv3"),
    ("gpt4v", "GPT-4V"),
)


def emit_pareto_figure(config: ExpConfig) -> Path | None:
    """Emit ``figures/fig_pareto.pdf`` from ``combined_metrics.json``."""
    cm = Path(config.output_dir) / "combined_metrics.json"
    if not cm.is_file():
        return None
    try:
        with cm.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("figures_pareto: cannot read %s (%s)", cm, exc)
        return None
    points: list[tuple[str, float, float]] = []
    for key, label in _SYSTEMS:
        params = data.get(f"{key}_params_m")
        f1 = data.get(f"{key}_global_f1")
        if isinstance(params, int | float) and isinstance(f1, int | float):
            points.append((label, float(params), float(f1)))
    if len(points) < 2:
        return None
    out_path = Path(config.output_dir) / "figures" / "fig_pareto.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _emit_scatter(points, out_path):
        return None
    return out_path


def _emit_scatter(
    points: list[tuple[str, float, float]], out_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for label, params, f1 in points:
        ax.scatter(params, f1, label=label, s=40)
        ax.annotate(label, (params, f1), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Macro-F1")
    ax.set_xscale("log")
    ax.grid(visible=True, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True
