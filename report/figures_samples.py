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


def _load_preds(
    path: Path, key: str = "pred_fields",
) -> dict[str, dict[str, str]]:
    """Read ``image_id -> {field: value}`` from ``donut_preds.jsonl``-style files.

    ``key`` selects which payload column to extract — ``pred_fields`` for
    model output, ``gt_fields`` for the ground-truth values that
    :func:`stages.eval_producers.write_preds_jsonl` writes alongside
    each prediction (there is no separate ``gt.jsonl``).
    """
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
        preds = obj.get(key)
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


def _block_text(
    ax: object, label: str, fields: dict[str, str],
    gt: dict[str, str], y0: float,
) -> None:
    """Render one ``label: ...`` block as a single multi-line text artist.

    Using one ``ax.text`` call per block lets matplotlib drive line
    spacing from the font's actual ascent/descent, so blocks no longer
    visually collide the way the previous per-line layout did when
    ``0.04`` axes-units < the rendered line height (review issue 1).
    Mismatched fields are surfaced by appending a trailing ``  ✗`` marker
    and recolouring the whole block red — colour-per-line would require
    multiple text artists and re-introduce the spacing bug.
    """
    any_bad = False
    body_lines: list[str] = []
    for f in _FIELDS:
        v = fields.get(f, "")
        ref = gt.get(f, "")
        bad = label != "GT" and (v.strip() != ref.strip())
        any_bad = any_bad or bad
        body_lines.append(f"  {f}: {v!s:.55s}{'  ✗' if bad else ''}")
    ax.text(  # type: ignore[attr-defined]
        0.02, y0, f"{label}:", transform=ax.transAxes,  # type: ignore[attr-defined]
        fontsize=7, fontweight="bold", family="monospace",
        verticalalignment="top",
    )
    ax.text(  # type: ignore[attr-defined]
        0.02, y0 - 0.06, "\n".join(body_lines),
        transform=ax.transAxes,  # type: ignore[attr-defined]
        fontsize=7, family="monospace",
        color="#B00020" if any_bad else "black",
        verticalalignment="top",
        linespacing=1.05,
    )


def render_samples(run_dir: Path) -> Path | None:
    """3×3 qualitative grid (or text-only) from prediction jsonls."""
    if not HAS_MPL:
        return None
    set_paper_style()
    preds_dir = run_dir / "predictions"
    donut = _load_preds(preds_dir / "donut_preds.jsonl")
    pipe = _load_preds(preds_dir / "pipeline_preds.jsonl")
    # ``write_preds_jsonl`` writes each ground-truth value alongside the
    # prediction under the ``gt_fields`` key — there is no separate
    # ``gt.jsonl`` artefact (review issue 1: empty GT blocks).
    gt = (
        _load_preds(preds_dir / "donut_preds.jsonl", key="gt_fields")
        or _load_preds(preds_dir / "pipeline_preds.jsonl", key="gt_fields")
    )
    ids = sorted(set(donut) | set(pipe) | set(gt))
    if guard_empty(ids, "samples"):
        return None
    cap = _MAX_WITH_IMAGES if _HAS_PIL else _MAX_TEXT_ONLY
    ids = ids[:cap]
    rows = 3 if _HAS_PIL else 2
    cols = 3 if _HAS_PIL else 2
    # Slightly taller figure so each cell has room for three text blocks
    # at non-colliding y-positions (review issue 1: overlapping blocks).
    fig, axes = plt.subplots(rows, cols, figsize=(COL_DOUBLE, 1.15 * COL_DOUBLE))
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
        ax.set_title(image_id, fontsize=7, family="monospace", loc="left")
        # Below-image text block (only when no image; otherwise overlay).
        if img_path is None:
            g, d, p = gt.get(image_id, {}), donut.get(image_id, {}), pipe.get(image_id, {})
            # Each block is roughly 6 text lines (label + 4 fields + spacer)
            # at fontsize=7 ≈ 0.32 axes-units.  Three blocks at y0 = 0.97,
            # 0.65, 0.33 leave a clear margin between them.
            _block_text(ax, "GT", g, g, 0.97)
            _block_text(ax, "DONUT", d, g, 0.65)
            _block_text(ax, "Pipeline", p, g, 0.33)
    for ax in axes_flat[len(ids):]:
        ax.set_axis_off()
    fig.suptitle("Qualitative sample predictions (GT vs DONUT vs Pipeline)", y=1.0)
    return save_fig(fig, run_dir / "figures", "fig_samples")
