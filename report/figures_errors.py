"""Stacked-bar error-type decomposition — one bar per (model, field).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_errors.pdf`` from
    ``predictions/per_field_errors.jsonl`` written at eval time by
    :mod:`core.metrics_errors`.  Four sub-panels (one per field) each
    with one stacked bar per model, showing the 8-category breakdown.
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
_CATEGORIES = (
    "correct", "partial", "wrong_span", "wrong_normalization",
    "hallucination", "ocr_error", "assigner_error",
    "missed_detection", "postprocess_error",
)


def _load_error_records(run_dir: Path) -> list[dict[str, object]]:
    for cand in (
        run_dir / "predictions" / "per_field_errors.jsonl",
        run_dir / "per_field_errors.jsonl",
    ):
        if not cand.is_file():
            continue
        try:
            lines = cand.read_text().splitlines()
        except OSError:
            continue
        out: list[dict[str, object]] = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        if out:
            return out
    return []


def _counts(records: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = {}
    for r in records:
        model = str(r.get("model", "unknown"))
        field = str(r.get("field", "unknown"))
        cat = str(r.get("category", "correct"))
        key = (model, field)
        out.setdefault(key, dict.fromkeys(_CATEGORIES, 0))
        out[key][cat] = out[key].get(cat, 0) + 1
    return out


def render_errors(run_dir: Path) -> Path | None:
    """Stacked-bar figure, four sub-panels (one per field)."""
    if not HAS_MPL:
        return None
    set_paper_style()
    records = _load_error_records(run_dir)
    if guard_empty(records, "errors"):
        return None
    counts = _counts(records)
    models = sorted({m for (m, _) in counts})
    if not models:
        return None
    fig, axes = plt.subplots(
        2, 2, figsize=(COL_DOUBLE, 0.75 * COL_DOUBLE),
        constrained_layout=True,
    )
    for idx, (ax, field) in enumerate(zip(axes.flat, _FIELDS, strict=True)):
        bottom = [0.0] * len(models)
        for ci, cat in enumerate(_CATEGORIES):
            vals = [float(counts.get((m, field), {}).get(cat, 0)) for m in models]
            ax.bar(models, vals, bottom=bottom,
                   color=PALETTE[ci % len(PALETTE)], label=cat,
                   edgecolor="black", linewidth=0.3)
            bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
        ax.set_title(field)
        if idx % 2 == 0:
            ax.set_ylabel("count")
    # Single shared legend + suptitle, anchored via constrained_layout so
    # they never collide with subplot titles or x-tick labels.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Error-type decomposition per field per model")
    fig.legend(
        handles, labels, loc="outside upper center", ncol=5, fontsize=6,
        frameon=False,
    )
    return save_fig(fig, run_dir / "figures", "fig_errors")
