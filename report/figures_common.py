"""Shared figure-emitter primitives — palette, style, save, JSON loaders.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: every ``report/figures_*.py`` module imports the same matplotlib
    configuration, the same colorblind-safe palette (Paul Tol muted),
    and the same ``save_fig`` helper.  Keeping this shared lets each
    emitter stay under the 166-LOC cap and guarantees visual
    consistency across the paper's 14+ figures.  The module is always
    importable even when matplotlib is absent — callers guard on the
    ``HAS_MPL`` re-export.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("kaggle2")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

__all__ = [
    "COL_DOUBLE",
    "COL_SINGLE",
    "HAS_MPL",
    "PALETTE",
    "guard_empty",
    "load_csv",
    "load_json",
    "plt",
    "save_fig",
    "set_paper_style",
]

# Paul Tol "muted" colorblind-safe palette.  Order:
# indigo, cyan, teal, green, olive, sand, rose, wine, purple.
PALETTE: tuple[str, ...] = (
    "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499",
)

# IEEE column widths (inches).
COL_SINGLE: float = 3.45
COL_DOUBLE: float = 7.16


def set_paper_style() -> None:
    """Apply IEEE-paper matplotlib defaults.  No-op without matplotlib."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save_fig(fig: Any, out_dir: Path, stem: str) -> Path | None:
    """Save ``fig`` as both PDF (vector) and PNG (raster fallback).

    Returns the PDF path on success, None on failure.  Never raises —
    figure-emitter contract forbids crashing the paper stage.
    """
    if not HAS_MPL or fig is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    try:
        fig.savefig(pdf_path)
        fig.savefig(png_path)
    except (OSError, ValueError) as exc:
        log.warning("figures_common: save failed for %s (%s)", stem, exc)
        plt.close(fig)
        return None
    plt.close(fig)
    return pdf_path


def load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file; return ``None`` on any error (never raise)."""
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("figures_common: cannot read %s (%s)", path, exc)
        return None
    return obj if isinstance(obj, dict) else None


def load_csv(path: Path) -> list[dict[str, float]]:
    """Tiny CSV→list-of-dicts loader (avoids pandas dependency)."""
    if not path.is_file():
        return []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        log.warning("figures_common: cannot read %s (%s)", path, exc)
        return []
    if not lines:
        return []
    header = lines[0].split(",")
    out: list[dict[str, float]] = []
    for raw in lines[1:]:
        parts = raw.split(",")
        if len(parts) != len(header):
            continue
        row: dict[str, float] = {}
        for k, v in zip(header, parts, strict=True):
            try:
                row[k] = float(v)
            except ValueError:
                continue
        if row:
            out.append(row)
    return out


def guard_empty(data: object, name: str) -> bool:
    """Log + return True when ``data`` is missing so callers can early-out."""
    if not data:
        log.info("figures_common: skipping %s (no data)", name)
        return True
    return False
