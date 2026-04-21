"""Matplotlib figure generators for the paper (best-effort, no raise).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: produces fig_training_curves, fig_gpu_telemetry, fig_per_field_confusion
    from training_log.json, telemetry_donut.jsonl, combined_metrics.json.
    2-in/1-out contract; missing source files log at INFO, never raise.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

log = logging.getLogger("kaggle2")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if raw:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:  # noqa: BLE001
                pass
    return rows


def render_training_curves(results_dir: str, out_dir: str) -> str | None:
    """Plot DONUT train/eval loss + eval F1 vs epoch for the paper."""
    if not _HAS_MPL:
        warnings.warn("matplotlib unavailable — skipping training curves", stacklevel=2)
        return None
    data = _load_json(Path(results_dir) / "training_log.json")
    if not data or not data.get("epochs"):
        log.info(
            "training_log.json missing/empty in %s — skipping training curves "
            "(run: train)", results_dir,
        )
        return None
    epochs = data["epochs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for key, label, marker in [("train_loss", "Train", "o"), ("eval_loss", "Eval", "s")]:
        if data.get(key):
            ax1.plot(epochs, data[key], label=f"{label} loss", marker=marker, ms=3)
    ax1.set(xlabel="Epoch", ylabel="Loss", title="DONUT Loss")
    ax1.legend(fontsize=8)
    if data.get("eval_f1"):
        ax2.plot(epochs, data["eval_f1"], color="tab:green", marker="^", ms=3)
    ax2.set(xlabel="Epoch", ylabel="F1", title="DONUT Eval F1")
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_training_curves.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_gpu_telemetry(results_dir: str, out_dir: str) -> str | None:
    """Plot GPU util/memory/power/temp over wall-clock time (Table II data)."""
    if not _HAS_MPL:
        warnings.warn("matplotlib unavailable — skipping GPU telemetry", stacklevel=2)
        return None
    rows = _load_jsonl(Path(results_dir) / "telemetry_donut.jsonl")
    if not rows:
        log.info(
            "telemetry_donut.jsonl not found or empty in %s — skipping GPU telemetry "
            "(run: train)", results_dir,
        )
        return None
    t0 = float(rows[0].get("ts", 0))
    ts = [(float(r.get("ts", t0)) - t0) / 60.0 for r in rows]
    series = [
        ([float(r.get("gpu_util_pct", 0)) for r in rows], "GPU Util (%)", "tab:blue"),
        ([float(r.get("gpu_mem_used_mb", 0)) / 1024 for r in rows], "Mem (GB)", "tab:orange"),
        ([float(r.get("gpu_power_w", 0)) for r in rows], "Power (W)", "tab:red"),
        ([float(r.get("gpu_temp_c", 0)) for r in rows], "Temp (°C)", "tab:purple"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
    for ax, (vals, ylabel, color) in zip(
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], series, strict=False
    ):
        ax.plot(ts, vals, color=color, lw=0.8)
        ax.set(ylabel=ylabel, xlabel="Time (min)")
    fig.suptitle("GPU Telemetry — DONUT Training", fontsize=10)
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_gpu_telemetry.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_per_field_confusion(results_dir: str, out_dir: str) -> str | None:
    """Stacked bar of per-field F1 per system (DONUT vs Pipeline)."""
    if not _HAS_MPL:
        warnings.warn("matplotlib unavailable — skipping confusion figure", stacklevel=2)
        return None
    data = _load_json(Path(results_dir) / "combined_metrics.json")
    if data is None:
        log.info(
            "combined_metrics.json not found in %s — skipping per-field confusion "
            "(run: eval)", results_dir,
        )
        return None
    fields = ["company", "date", "address", "total"]
    systems = [("DONUT", "donut_f1"), ("Pipeline", "pipeline_f1")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, (name, pfx) in zip(axes, systems, strict=False):
        vals = [float(data.get(f"{pfx}_{f}", 0.0)) for f in fields]
        ax.bar(fields, vals, color="tab:blue")
        ax.set(title=name, ylim=(0, 1))
        ax.set_ylabel("F1" if ax is axes[0] else "")
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_per_field_confusion.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(results_dir: str, out_dir: str = "") -> list[str]:
    """Render all available paper figures; skip silently when data is missing."""
    out = out_dir or results_dir
    Path(out).mkdir(parents=True, exist_ok=True)
    fns = (render_training_curves, render_gpu_telemetry, render_per_field_confusion)
    return [r for fn in fns for r in [fn(results_dir, out)] if r is not None]
