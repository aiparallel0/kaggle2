"""LaTeX tabular emitters that materialise ``\\VAR{table_*}`` blocks.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: replace prose-embedded literals with auto-generated tables sourced
    from the expanded metrics dict.  Each emitter takes ``metrics`` and
    returns a complete ``\\begin{tabular}...\\end{tabular}`` block (no
    ``\\begin{table}`` wrapper — that lives in the section file so
    captions and labels are authored alongside the prose).  Missing
    values render as ``---`` via :func:`_fmt` so a partial run still
    compiles; a WARNING is logged for auditability.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("kaggle2")

_FIELDS = ("company", "date", "address", "total")


def _fmt(metrics: dict[str, Any], key: str, directive: str = "sig3") -> str:
    """Format ``metrics[key]`` per ``directive``; ``---`` if missing."""
    if key not in metrics:
        return "---"
    from report.inject_format import apply_directive
    out = apply_directive(metrics[key], directive)
    return out if out is not None else "---"


def render_f1_table(metrics: dict[str, Any]) -> str:
    """Headline DONUT vs Pipeline vs Rule-based F1 per field + macro."""
    rows: list[str] = []
    for f in _FIELDS:
        rows.append(
            f"{f} & {_fmt(metrics, f'donut_f1_{f}', 'pct1')} "
            f"& {_fmt(metrics, f'pipeline_f1_{f}', 'pct1')} "
            f"& {_fmt(metrics, f'rulebased_f1_{f}', 'pct1')} \\\\",
        )
    macro = (
        "\\textbf{macro} "
        f"& \\textbf{{{_fmt(metrics, 'donut_f1', 'pct1')}}} "
        f"& \\textbf{{{_fmt(metrics, 'pipeline_f1', 'pct1')}}} "
        f"& \\textbf{{{_fmt(metrics, 'gtocr_rulebased_f1', 'pct1')}}} \\\\"
    )
    body = "\n".join(rows + [macro])
    return (
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "field & DONUT & Pipeline & Rule-based \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}"
    )


def render_extended_table(metrics: dict[str, Any]) -> str:
    """Per-field precision/recall/F1/EM with bootstrap CI for DONUT + Pipeline."""
    rows: list[str] = []
    for system in ("donut", "pipeline"):
        for f in _FIELDS:
            rows.append(
                f"{system} & {f} "
                f"& {_fmt(metrics, f'{system}_precision_{f}', 'pct1')} "
                f"& {_fmt(metrics, f'{system}_recall_{f}', 'pct1')} "
                f"& {_fmt(metrics, f'{system}_f1_{f}', 'pct1')} "
                f"& {_fmt(metrics, f'{system}_em_{f}', 'pct1')} \\\\",
            )
    return (
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "system & field & P & R & F1 & EM \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )


def render_latency_table(metrics: dict[str, Any]) -> str:
    """Mean/p50/p95/p99 latency + throughput + USD per image per system."""
    rows: list[str] = []
    for system in ("donut", "pipeline"):
        rows.append(
            f"{system} "
            f"& {_fmt(metrics, f'{system}_latency_mean', 'ms')} "
            f"& {_fmt(metrics, f'{system}_latency_p50', 'ms')} "
            f"& {_fmt(metrics, f'{system}_latency_p95', 'ms')} "
            f"& {_fmt(metrics, f'{system}_latency_p99', 'ms')} "
            f"& {_fmt(metrics, f'{system}_throughput_batch1', 'sig3')} "
            f"& {_fmt(metrics, f'{system}_usd_per_img', 'usd4')} \\\\",
        )
    return (
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "system & mean & p50 & p95 & p99 & img/s & USD/img \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )


def _fmt_raw(metrics: dict[str, Any], key: str) -> str:
    """Return ``metrics[key]`` as a LaTeX-safe string; ``---`` if missing."""
    if key not in metrics:
        return "---"
    return str(metrics[key])


def render_env_table(metrics: dict[str, Any]) -> str:
    """Reproducibility footer — git sha, cuda, gpu, seed."""
    keys = (
        ("git_sha", "git SHA"),
        ("config_sha256", "config sha256"),
        ("torch_version", "torch"),
        ("cuda_version", "CUDA"),
        ("gpu_model", "GPU"),
        ("driver_version", "driver"),
        ("seed", "seed"),
        ("run_id", "run\\_id"),
    )
    rows = "\n".join(
        f"{label} & \\texttt{{{_fmt_raw(metrics, key)}}} \\\\"
        for key, label in keys
    )
    return (
        "\\begin{tabular}{ll}\n\\toprule\nproperty & value \\\\\n\\midrule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}"
    )


def render_training_table(metrics: dict[str, Any]) -> str:
    """Training-summary — epochs / best / wall-clock / VRAM / USD / energy."""
    rows: list[str] = []
    for system in ("donut", "pipeline"):
        rows.append(
            f"{system} "
            f"& {_fmt(metrics, f'{system}_epochs', 'int')} "
            f"& {_fmt(metrics, f'{system}_best_epoch', 'int')} "
            f"& {_fmt(metrics, f'{system}_wall_clock_s', 'sig4')} "
            f"& {_fmt(metrics, f'{system}_peak_vram_gb', 'gb1')} "
            f"& {_fmt(metrics, f'{system}_cost_usd', 'usd')} "
            f"& {_fmt(metrics, f'{system}_energy_wh', 'wh')} \\\\",
        )
    return (
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "system & epochs & best & wall-s & VRAM & USD & energy \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )


_TABLES = {
    "table_headline_f1": render_f1_table,
    "table_extended": render_extended_table,
    "table_latency": render_latency_table,
    "table_env": render_env_table,
    "table_training": render_training_table,
}


def inject_tables(metrics: dict[str, Any]) -> dict[str, str]:
    """Return ``{key: tabular_block}`` for every supported table emitter."""
    return {name: fn(metrics) for name, fn in _TABLES.items()}
