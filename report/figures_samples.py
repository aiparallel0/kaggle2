"""Qualitative samples grid — curated headline + supplementary full grid.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: produce ``fig_samples.pdf`` (Fig.~11 headline) and
    ``fig_samples_full.pdf`` (supplementary) from
    ``predictions/donut_preds.jsonl`` + ``predictions/pipeline_preds.jsonl``.

v4 healing:
  * Headline is a 2×2 grid curated by outcome bucket
    (both-correct / DONUT-wins / pipeline-wins / both-fail).
  * 9-pt black monospace JSON below each receipt thumbnail; mismatched
    fields are highlighted with a red ``\u2717`` glyph on the offending
    line only (v3 recoloured the whole block, hiding which field
    failed).
  * Curated IDs come from ``config.qualitative_sample_ids`` (recovered
    from ``env/config_snapshot.json``); when empty we auto-curate one
    example per bucket on a deterministic sorted-id traversal.
  * The legacy 3×3 nine-receipt grid is preserved as a supplementary
    figure — no information lost.
  * When the chosen images / predictions are missing on disk, the cell
    renders an honest "asset pending — image_id=…" placeholder
    identifying the gap.  Never em-dashes, never zero-bars.
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
_CURATED_N = 4
_BUCKETS = (
    "(a) both correct", "(b) DONUT correct only",
    "(c) Pipeline correct only", "(d) both fail",
)


def _load_preds(path: Path, key: str = "pred_fields") -> dict[str, dict[str, str]]:
    """Read ``image_id -> {field: value}`` from a ``*_preds.jsonl`` file."""
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
    """Best-effort thumbnail lookup; falls back to SROIE source images."""
    for d in (
        run_dir / "predictions" / "samples", run_dir / "samples",
        run_dir.parent / "predictions" / "samples",
        Path("data/sroie_cache/0325updated.task1train(626p)"),
        Path("data/sroie_cache"),
    ):
        for ext in (".jpg", ".png", ".jpeg"):
            p = d / f"{image_id}{ext}"
            if p.is_file():
                return p
    return None


def _matched(pred: dict[str, str], gt: dict[str, str]) -> set[str]:
    return {f for f in _FIELDS if pred.get(f, "").strip() == gt.get(f, "").strip()}


def _curate(
    donut: dict[str, dict[str, str]],
    pipe: dict[str, dict[str, str]],
    gt: dict[str, dict[str, str]],
    forced: list[str],
) -> list[str]:
    """Select 4 IDs by outcome bucket; ``forced`` overrides verbatim."""
    if forced:
        return forced[:_CURATED_N]
    ids = sorted(set(donut) & set(pipe) & set(gt))
    full = set(_FIELDS)
    b_both, b_d, b_p, b_neither = [], [], [], []
    for i in ids:
        d_ok = _matched(donut[i], gt[i]) == full
        p_ok = _matched(pipe[i], gt[i]) == full
        if d_ok and p_ok:
            b_both.append(i)
        elif d_ok:
            b_d.append(i)
        elif p_ok:
            b_p.append(i)
        else:
            b_neither.append(i)
    out: list[str] = []
    for bucket in (b_both, b_d, b_p, b_neither):
        if bucket:
            out.append(bucket[0])
    pool = [i for i in ids if i not in out]
    while len(out) < _CURATED_N and pool:
        out.append(pool.pop(0))
    return out


def _block(
    ax: object, label: str, fields: dict[str, str],
    gt: dict[str, str], y0: float, fontsize: int,
) -> None:
    """Render one ``label: ...`` block with per-line red-✗ on mismatch."""
    ax.text(  # type: ignore[attr-defined]
        0.02, y0, f"{label}:", transform=ax.transAxes,  # type: ignore[attr-defined]
        fontsize=fontsize, fontweight="bold", family="monospace",
        verticalalignment="top",
    )
    dy = 0.045 * (fontsize / 9.0)
    cross = "  \u2717"
    for i, f in enumerate(_FIELDS):
        v = fields.get(f, "")
        bad = label != "GT" and v.strip() != gt.get(f, "").strip()
        marker = cross if bad else ""
        ax.text(  # type: ignore[attr-defined]
            0.04, y0 - (i + 1) * dy,
            f"{f}: {v!s:.55s}{marker}",
            transform=ax.transAxes,  # type: ignore[attr-defined]
            fontsize=fontsize, family="monospace",
            color="#B00020" if bad else "black",
            verticalalignment="top",
        )


def _placeholder(ax: object, image_id: str, bucket: str) -> None:
    """Honest 'asset pending' cell — never em-dashes, never fake data."""
    ax.text(  # type: ignore[attr-defined]
        0.5, 0.55, f"[no candidate]\n{bucket}",
        transform=ax.transAxes,  # type: ignore[attr-defined]
        ha="center", va="center", fontsize=10, family="monospace",
        color="#7A7A7A",
    )
    ax.text(  # type: ignore[attr-defined]
        0.5, 0.18,
        f"image_id={image_id}\n(prediction or thumbnail\n missing on this run)",
        transform=ax.transAxes,  # type: ignore[attr-defined]
        ha="center", va="center", fontsize=8, family="monospace",
        color="#7A7A7A",
    )


def _read_curated_ids(run_dir: Path) -> list[str]:
    """Recover ``config.qualitative_sample_ids`` from the snapshotted config."""
    snap = run_dir / "env" / "config_snapshot.json"
    if not snap.is_file():
        return []
    try:
        cfg = json.loads(snap.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    ids = cfg.get("qualitative_sample_ids") or []
    return [str(x) for x in ids] if isinstance(ids, list) else []


def _draw_cell(
    ax: object, image_id: str, bucket: str, run_dir: Path,
    g: dict[str, str], d: dict[str, str], p: dict[str, str],
    fontsize: int,
) -> None:
    """Draw a single curated cell — image overlay or text block."""
    ax.set_axis_off()  # type: ignore[attr-defined]
    ax.set_title(f"{bucket}  \u2014  {image_id}", fontsize=8,  # type: ignore[attr-defined]
                 family="monospace", loc="left")
    img = _image_path_for(run_dir, image_id) if _HAS_PIL else None
    if img is not None:
        try:
            ax.imshow(Image.open(img))  # type: ignore[attr-defined]
            return
        except (OSError, ValueError) as exc:  # pragma: no cover
            log.info("samples: cannot open %s (%s)", img, exc)
    if not g and not d and not p:
        _placeholder(ax, image_id, bucket)
        return
    _block(ax, "GT", g, g, 0.92, fontsize)
    _block(ax, "DONUT", d, g, 0.62, fontsize)
    _block(ax, "Pipeline", p, g, 0.32, fontsize)


def render_samples(run_dir: Path) -> Path | None:
    """Headline 2×2 curated grid (Fig.~11) + supplementary 3×3 grid."""
    if not HAS_MPL:
        return None
    set_paper_style()
    pdir = run_dir / "predictions"
    donut = _load_preds(pdir / "donut_preds.jsonl")
    pipe = _load_preds(pdir / "pipeline_preds.jsonl")
    gt = (_load_preds(pdir / "donut_preds.jsonl", key="gt_fields")
          or _load_preds(pdir / "pipeline_preds.jsonl", key="gt_fields"))
    if guard_empty(sorted(set(donut) | set(pipe) | set(gt)), "samples"):
        return None
    ids = _curate(donut, pipe, gt, _read_curated_ids(run_dir))
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 1.05 * COL_DOUBLE))
    for slot, ax in enumerate(list(axes.flat)):
        bucket = _BUCKETS[slot]
        if slot >= len(ids):
            ax.set_axis_off()
            _placeholder(ax, "(unfilled)", bucket)
            continue
        img_id = ids[slot]
        _draw_cell(ax, img_id, bucket, run_dir,
                   gt.get(img_id, {}), donut.get(img_id, {}),
                   pipe.get(img_id, {}), 9)
    fig.suptitle(
        "Qualitative sample predictions (curated 2×2: outcome buckets)", y=1.0,
    )
    headline = save_fig(fig, run_dir / "figures", "fig_samples")
    try:
        _render_full(run_dir, donut, pipe, gt)
    except Exception as exc:  # noqa: BLE001 — supplementary, never fatal
        log.info("samples: full-grid supplementary skipped (%s)", exc)
    return headline


def _render_full(
    run_dir: Path,
    donut: dict[str, dict[str, str]],
    pipe: dict[str, dict[str, str]],
    gt: dict[str, dict[str, str]],
) -> Path | None:
    """3×3 (or 2×2 text-only) supplementary grid — preserved from v3."""
    ids = sorted(set(donut) | set(pipe) | set(gt))
    if guard_empty(ids, "samples_full"):
        return None
    cap = 9 if _HAS_PIL else 4
    rows = cols = 3 if _HAS_PIL else 2
    ids = ids[:cap]
    fig, axes = plt.subplots(rows, cols, figsize=(COL_DOUBLE, 1.15 * COL_DOUBLE))
    for ax, img_id in zip(list(axes.flat), ids, strict=False):
        _draw_cell(ax, img_id, "", run_dir,
                   gt.get(img_id, {}), donut.get(img_id, {}),
                   pipe.get(img_id, {}), 7)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_axis_off()
    fig.suptitle(
        "Qualitative sample predictions — full grid (supplementary)", y=1.0,
    )
    return save_fig(fig, run_dir / "figures", "fig_samples_full")
