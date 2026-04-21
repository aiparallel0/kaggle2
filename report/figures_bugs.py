"""Nice-to-have figures: bug-timeline and DONUT-vs-pipeline telemetry overlay.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: two auxiliary emitters alongside :mod:`report.figures_extra`.
    ``render_bug_timeline`` reads ``results/bug_timeline.json`` and
    plots per-bug F1 before/after (the ``measured`` flag is rendered
    as filled vs. hollow markers so readers can separate empirically
    reproduced values from mechanistic lower bounds — Section IV).
    ``render_telemetry_overlay`` overlays DONUT vs. pipeline GPU
    utilisation on a common minutes-axis from
    ``telemetry_donut.jsonl`` / ``telemetry_pipeline.jsonl``,
    complementing the full four-panel figure from
    :func:`report.figures.render_gpu_telemetry`.  Both emitters are
    best-effort: missing source files warn and return ``None``.
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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
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
            except json.JSONDecodeError:
                pass
    return rows


def render_bug_timeline(results_dir: str, out_dir: str) -> str | None:
    """Horizontal F1-before/F1-after timeline across the thirteen bugs.

    Reads ``bug_timeline.json`` (see :mod:`results/bug_timeline.json`
    for schema).  If ``combined_metrics.json`` is also present, its
    ``pipeline_f1`` is used as the common post-fix F1 in place of the
    fixture's ``f1_after_default`` so the figure stays consistent
    with the headline table at paper-build time.
    """
    if not _HAS_MPL:
        warnings.warn("matplotlib unavailable — skipping bug timeline", stacklevel=2)
        return None
    data = _load_json(Path(results_dir) / "bug_timeline.json")
    if data is None or "bugs" not in data:
        warnings.warn(
            f"bug_timeline.json not found in {results_dir}", stacklevel=2,
        )
        return None
    bugs = data["bugs"]
    f1_after = _resolve_f1_after(Path(results_dir), data)
    ids = [b["id"] for b in bugs]
    shorts = [b.get("short", f"Bug {b['id']}") for b in bugs]
    befores = [float(b.get("f1_before", 0.0)) for b in bugs]
    measured = [bool(b.get("measured", False)) for b in bugs]
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.35 * len(bugs) + 1.2)))
    y = list(range(len(bugs)))
    for yi, before, meas in zip(y, befores, measured, strict=True):
        ax.plot(
            [before, f1_after], [yi, yi],
            color="tab:gray", linestyle=":", lw=1.0, zorder=1,
        )
        ax.scatter(
            before, yi, s=45, zorder=2,
            facecolor="tab:red" if meas else "white",
            edgecolor="tab:red", linewidth=1.2,
        )
        ax.scatter(f1_after, yi, s=45, zorder=2,
                   facecolor="tab:green", edgecolor="tab:green")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i}. {s}" for i, s in zip(ids, shorts, strict=True)],
                       fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Global token F1 (SROIE test)")
    ax.set_title("F1 before (red) vs. after (green) each of the 13 bug fixes")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_bug_timeline.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _resolve_f1_after(results_dir: Path, fixture: dict[str, Any]) -> float:
    """Pick post-fix F1: live ``combined_metrics.json`` > fixture default."""
    combined = _load_json(results_dir / "combined_metrics.json")
    if combined and "pipeline_f1" in combined:
        try:
            return float(combined["pipeline_f1"])
        except (TypeError, ValueError):
            pass
    return float(fixture.get("f1_after_default", 0.74))


def render_telemetry_overlay(results_dir: str, out_dir: str) -> str | None:
    """DONUT vs. pipeline GPU utilisation on a common minutes-axis."""
    if not _HAS_MPL:
        warnings.warn(
            "matplotlib unavailable — skipping telemetry overlay", stacklevel=2,
        )
        return None
    donut = _load_jsonl(Path(results_dir) / "telemetry_donut.jsonl")
    pipe = _load_jsonl(Path(results_dir) / "telemetry_pipeline.jsonl")
    if not donut and not pipe:
        warnings.warn(
            f"no telemetry_*.jsonl files found in {results_dir}", stacklevel=2,
        )
        return None
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for rows, label, color in (
        (donut, "DONUT", "tab:blue"), (pipe, "Pipeline", "tab:orange"),
    ):
        if not rows:
            continue
        t0 = float(rows[0].get("ts", 0.0))
        ts = [(float(r.get("ts", t0)) - t0) / 60.0 for r in rows]
        util = [float(r.get("gpu_util_pct", 0.0)) for r in rows]
        ax.plot(ts, util, color=color, lw=0.9, label=label)
    ax.set(xlabel="Wall-clock time (min)", ylabel="GPU utilisation (%)",
           title="GPU-util overlay: DONUT vs. YOLO+TrOCR+Assigner pipeline")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = str(Path(out_dir) / "fig_telemetry_overlay.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all_bugs_telemetry(results_dir: str, out_dir: str) -> list[str]:
    """Run the two nice-to-have emitters; return written PDF paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    emitters = (render_bug_timeline, render_telemetry_overlay)
    return [r for fn in emitters for r in [fn(results_dir, out_dir)] if r is not None]
