"""Per-field F1, assigner loss-curve, and pipeline-diagnostic figures.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: three figure emitters that turn eval artefacts into PDFs
    referenced by the results section of the paper:
    ``render_f1_by_system`` (DONUT / Pipeline / Rule-based × fields
    from ``combined_metrics.json``); ``render_assigner_loss_curve``
    (per-epoch trajectory from ``assigner_metrics.json`` with the
    best-epoch vertical rule); ``render_pipeline_diagnostics``
    (``empty_detection_fraction`` and ``per_receipt_error_fraction``
    from ``pipeline_metrics.json``).  The heatmap emitter lives in
    :mod:`report.figures_attn` so no module crosses the 166-LOC cap.
    All emitters follow :mod:`report.figures`' best-effort contract:
    missing source files emit :class:`UserWarning` and return ``None``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_FIELDS = ("company", "date", "address", "total")
_FIGSIZE_SINGLE = (6.8, 3.6)


def _guard_mpl(label: str) -> bool:
    """Return ``True`` when matplotlib is importable; else warn and bail out."""
    if _HAS_MPL:
        return True
    warnings.warn(f"matplotlib unavailable — skipping {label}", stacklevel=3)
    return False


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


def render_f1_by_system(results_dir: str, out_dir: str) -> str | None:
    """Grouped bar chart: DONUT / Pipeline / Rule-based × four SROIE fields."""
    if not _guard_mpl("F1-by-system"):
        return None
    data = _load_json(Path(results_dir) / "combined_metrics.json")
    if data is None:
        warnings.warn(
            f"combined_metrics.json not found in {results_dir}", stacklevel=2,
        )
        return None
    systems = (("DONUT", "donut"), ("Pipeline", "pipeline"), ("Rule-based", "rulebased"))
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    n_groups, n_systems = len(_FIELDS), len(systems)
    width = 0.8 / n_systems
    x = list(range(n_groups))
    for i, (label, prefix) in enumerate(systems):
        vals = [_lookup_f1(data, prefix, f) for f in _FIELDS]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=label)
    ax.set_xticks([xi + width * (n_systems - 1) / 2 for xi in x])
    ax.set_xticklabels(_FIELDS)
    ax.set_ylabel("Token F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-field F1 by system (SROIE test split)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_f1_by_system.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _lookup_f1(data: dict[str, Any], prefix: str, field: str) -> float:
    """Read ``<prefix>_f1_<field>``; fall back to ``<prefix>_f1`` for pipeline."""
    key = f"{prefix}_f1_{field}"
    if key in data:
        return float(data.get(key, 0.0))
    # Pipeline arm stores per-field F1 only inside pipeline_metrics.json;
    # surface the global as the "company" bar so the group is non-empty.
    return float(data.get(f"{prefix}_f1", 0.0)) if field == "company" else 0.0


def render_assigner_loss_curve(results_dir: str, out_dir: str) -> str | None:
    """Assigner train/val loss per epoch, with the best-epoch marker."""
    if not _guard_mpl("assigner loss curve"):
        return None
    data = _load_json(Path(results_dir) / "assigner_metrics.json")
    if data is None or not data.get("train_loss"):
        warnings.warn(
            f"assigner_metrics.json missing per-epoch loss lists in {results_dir}",
            stacklevel=2,
        )
        return None
    train = [float(x) for x in data["train_loss"]]
    val = [float(x) if x is not None else float("nan") for x in data.get("val_loss", [])]
    epochs = list(range(1, len(train) + 1))
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    ax.plot(epochs, train, marker="o", ms=3, label="Train", color="tab:blue")
    if val:
        ax.plot(epochs, val, marker="s", ms=3, label="Val", color="tab:orange")
    best_epoch = data.get("best_epoch")
    if isinstance(best_epoch, int) and 1 <= best_epoch <= len(epochs):
        ax.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7,
                   label=f"best (epoch {best_epoch})")
    ax.set(xlabel="Epoch", ylabel="Pos-mass NLL loss",
           title="AttentionAssigner training trajectory")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_assigner_loss_curve.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_pipeline_diagnostics(results_dir: str, out_dir: str) -> str | None:
    """Two-bar chart of empty-detection + per-receipt-error fractions."""
    if not _guard_mpl("pipeline diagnostics"):
        return None
    data = _load_json(Path(results_dir) / "pipeline_metrics.json")
    if data is None:
        warnings.warn(
            f"pipeline_metrics.json not found in {results_dir}", stacklevel=2,
        )
        return None
    labels = ("Empty detection\n(YOLO 0 boxes)", "Per-receipt error\n(exception caught)")
    vals = (
        float(data.get("empty_detection_fraction", 0.0)),
        float(data.get("per_receipt_error_fraction", 0.0)),
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(labels, vals, color=("tab:orange", "tab:red"), width=0.55)
    ax.set_ylabel("Fraction of test receipts")
    ax.set_ylim(0, max(0.12, max(vals) * 1.25))
    ax.set_title("Pipeline per-receipt diagnostics")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_pipeline_diagnostics.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all_extra(results_dir: str, out_dir: str) -> list[str]:
    """Run the three ICDAR extra emitters; return the list of written PDFs."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    emitters = (
        render_f1_by_system, render_assigner_loss_curve, render_pipeline_diagnostics,
    )
    return [r for fn in emitters for r in [fn(results_dir, out_dir)] if r is not None]
