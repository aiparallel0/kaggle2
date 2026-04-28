"""PR-D — Carbon-emissions figure (kgCO2e by stage).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: reads ``runs/<id>/env/env_snapshot.json`` and emits a stacked
    bar chart of kgCO2e per training stage (DONUT / YOLO / TrOCR /
    assigner).  Falls back to a textual placeholder when matplotlib
    or the input JSON is unavailable so the paper compile never
    aborts on a CI box without GUI deps.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("kaggle2")

_DEFAULT_GRID_FACTOR = 0.475  # global average kgCO2e per kWh (2023).
_STAGES = ("donut", "yolo", "trocr", "assigner")


def emit_carbon_figure(config: ExpConfig) -> Path | None:
    """Emit ``figures/fig_carbon.pdf`` from the env snapshot.

    Returns the figure path on success, ``None`` when the input is
    missing or matplotlib is not installed.
    """
    snap = Path(config.output_dir) / "env" / "env_snapshot.json"
    if not snap.is_file():
        return None
    try:
        with snap.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("figures_carbon: cannot read %s (%s)", snap, exc)
        return None
    grid_raw = config.extra.get(
        "grid_factor",
        config.carbon_grid_factor_kgco2e_per_kwh or _DEFAULT_GRID_FACTOR,
    )
    grid = float(grid_raw) if isinstance(grid_raw, int | float) else _DEFAULT_GRID_FACTOR
    tdp = float(data.get("gpu_tdp_w", 0.0))
    durations = data.get("stage_seconds", {})
    if not isinstance(durations, dict):
        return None
    bars: dict[str, float] = {}
    for stage in _STAGES:
        secs = float(durations.get(stage, 0.0))
        bars[stage] = tdp * secs / 3600.0 / 1000.0 * grid
    out_path = Path(config.output_dir) / "figures" / "fig_carbon.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _emit_bars(bars, out_path):
        return None
    return out_path


def _emit_bars(bars: dict[str, float], out_path: Path) -> bool:
    """Render ``bars`` as a one-axis bar chart; return True on success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    keys = list(bars.keys())
    vals = [bars[k] for k in keys]
    ax.bar(keys, vals, color="#4f81bd")
    ax.set_ylabel("kgCO$_2$e")
    ax.set_title("Per-stage carbon footprint")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True
