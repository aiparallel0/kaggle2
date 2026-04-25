"""Qualitative samples grid — curated receipts with GT vs predictions.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_samples.pdf`` from
    ``predictions/donut_preds.jsonl`` + ``predictions/pipeline_preds.jsonl``.
    The previous renderer crammed twelve receipts into a 3×4 grid of
    5-pt red monospace text with no embedded images — illegible at any
    print zoom (review item S2).  We instead show at most nine
    receipts in a 3×3 grid (or four text-only when no thumbnail is
    available), embed the receipt thumbnail when Pillow is installed,
    and render the GT/DONUT/Pipeline JSON below in 9-pt black
    monospace, with mismatched fields highlighted in red.  Never
    raises.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from report.figures_common import (
    COL_DOUBLE,
    HAS_MPL,
    guard_empty,
    plt,
    save_fig,
    set_paper_style,
)

try:  # Pillow is optional — text-only fallback below.
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

log = logging.getLogger("kaggle2")

_FIELDS = ("company", "date", "address", "total")
_MAX_WITH_IMAGES = 9
_MAX_TEXT_ONLY = 4


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


def _image_path_for(run_dir: Path, image_id: str) -> Path | None:
    """Best-effort lookup of the receipt thumbnail for ``image_id``."""
    candidates = (
        run_dir / "predictions" / "samples" / f"{image_id}.jpg",
        run_dir / "predictions" / "samples" / f"{image_id}.png",
        run_dir / "samples" / f"{image_id}.jpg",
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def _block_text(  # noqa: PLR0913 - readable to keep all four lines together
    ax: object, label: str, fields: dict[str, str],
    gt: dict[str, str], y0: float,
) -> None:
    """Render one ``label: ...`` block with red highlighting on mismatches."""
    ax.text(0.02, y0, f"{label}:", transform=ax.transAxes,  # type: ignore[attr-defined]
            fontsize=8, fontweight="bold", family="monospace")
    for i, f in enumerate(_FIELDS):
        v = fields.get(f, "")
        ref = gt.get(f, "")
        bad = label != "GT" and (v.strip() != ref.strip())
        ax.text(  # type: ignore[attr-defined]
            0.02, y0 - 0.04 * (i + 1),
            f"  {f}: {v!s:.60s}",
            transform=ax.transAxes,  # type: ignore[attr-defined]
            fontsize=8, family="monospace",
            color="#B00020" if bad else "black",
        )


def render_samples(run_dir: Path) -> Path | None:
    """3×3 qualitative grid (or text-only) from prediction jsonls."""
    if not HAS_MPL:
        return None
    set_paper_style()
    preds_dir = run_dir / "predictions"
    donut = _load_preds(preds_dir / "donut_preds.jsonl")
    pipe = _load_preds(preds_dir / "pipeline_preds.jsonl")
    gt = _load_preds(preds_dir / "gt.jsonl")
    ids = sorted(set(donut) | set(pipe) | set(gt))
    if guard_empty(ids, "samples"):
        return None
    cap = _MAX_WITH_IMAGES if _HAS_PIL else _MAX_TEXT_ONLY
    ids = ids[:cap]
    rows = 3 if _HAS_PIL else 2
    cols = 3 if _HAS_PIL else 2
    fig, axes = plt.subplots(rows, cols, figsize=(COL_DOUBLE, 0.95 * COL_DOUBLE))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, image_id in zip(axes_flat, ids, strict=False):
        ax.set_axis_off()
        img_path = _image_path_for(run_dir, image_id) if _HAS_PIL else None
        if img_path is not None:
            try:
                ax.imshow(Image.open(img_path))
            except (OSError, ValueError) as exc:  # pragma: no cover
                log.info("samples: cannot open %s (%s)", img_path, exc)
                img_path = None
        ax.set_title(image_id, fontsize=8, family="monospace", loc="left")
        # Below-image text block (only when no image; otherwise overlay).
        if img_path is None:
            g, d, p = gt.get(image_id, {}), donut.get(image_id, {}), pipe.get(image_id, {})
            _block_text(ax, "GT", g, g, 0.95)
            _block_text(ax, "DONUT", d, g, 0.70)
            _block_text(ax, "Pipeline", p, g, 0.45)
    for ax in axes_flat[len(ids):]:
        ax.set_axis_off()
    fig.suptitle("Qualitative sample predictions (GT vs DONUT vs Pipeline)", y=1.0)
    return save_fig(fig, run_dir / "figures", "fig_samples")
