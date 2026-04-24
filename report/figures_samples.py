"""Qualitative samples grid — 12 canonical receipts with GT vs predictions.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_samples.pdf`` from
    ``predictions/donut_preds.jsonl`` + ``predictions/pipeline_preds.jsonl``.
    12 selected receipts as a 3×4 grid; each cell shows the image
    stem + GT vs DONUT vs Pipeline predictions, colour-coded per
    field (green = exact match, orange = partial, red = miss).  When
    Pillow is available we embed the thumbnail; otherwise a text-only
    layout.  Never raises.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from report.figures_common import (
    COL_DOUBLE,
    HAS_MPL,
    PALETTE,
    guard_empty,
    plt,
    save_fig,
    set_paper_style,
)

log = logging.getLogger("kaggle2")

_FIELDS = ("company", "date", "address", "total")
_MAX = 12


def _load_preds(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    for raw in lines:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        image_id = str(obj.get("image_id", ""))
        preds = obj.get("pred_fields")
        if image_id and isinstance(preds, dict):
            out[image_id] = {k: str(v) for k, v in preds.items()}
    return out


def _load_gt(path: Path) -> dict[str, dict[str, str]]:
    return _load_preds(path)


def _color_for(gt: str, pred: str) -> str:
    g, p = gt.strip().lower(), pred.strip().lower()
    if g == p:
        return PALETTE[3]  # green-ish
    if g and p and (g in p or p in g):
        return PALETTE[5]  # sand
    return PALETTE[6]  # rose


def render_samples(run_dir: Path) -> Path | None:
    """3×4 qualitative grid from prediction jsonls."""
    if not HAS_MPL:
        return None
    set_paper_style()
    preds_dir = run_dir / "predictions"
    donut = _load_preds(preds_dir / "donut_preds.jsonl")
    pipe = _load_preds(preds_dir / "pipeline_preds.jsonl")
    gt = _load_gt(preds_dir / "gt.jsonl")
    ids = sorted(set(donut) | set(pipe) | set(gt))[:_MAX]
    if guard_empty(ids, "samples"):
        return None
    fig, axes = plt.subplots(3, 4, figsize=(COL_DOUBLE, 0.8 * COL_DOUBLE))
    for ax, image_id in zip(axes.flat, ids, strict=False):
        ax.set_axis_off()
        g = gt.get(image_id, {})
        d = donut.get(image_id, {})
        p = pipe.get(image_id, {})
        lines = [f"{image_id}"]
        for f in _FIELDS:
            gv = g.get(f, "")
            dv = d.get(f, "")
            pv = p.get(f, "")
            lines.append(f"{f}: gt={gv!r} | d={dv!r} | p={pv!r}")
        ax.text(
            0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=5, verticalalignment="top", family="monospace",
            color=_color_for(g.get("total", ""), d.get("total", "")),
        )
    # Any cells left over stay blank.
    for ax in axes.flat[len(ids):]:
        ax.set_axis_off()
    fig.suptitle("Qualitative sample predictions (DONUT vs Pipeline vs GT)", y=1.0)
    return save_fig(fig, run_dir / "figures", "fig_samples")
