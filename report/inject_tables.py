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
    """Format ``metrics[key]`` per ``directive``; ``\\MissingCell`` if missing.

    Replaces the legacy ``---`` em-dash backstop with a typed marker that
    is red in the PDF and counted by ``check_artefacts`` as a build
    blocker — unless the key is on the
    :data:`report.missing.MISSING_OK_KEYS` / ``MISSING_OK_PREFIXES``
    allow-list, in which case the cell renders ``\\textit{n/a}`` to
    document an intentional skip without inflating the blocker count.
    """
    if key not in metrics:
        from report.missing import is_missing_ok, render_missing_cell
        return "\\textit{n/a}" if is_missing_ok(key) else render_missing_cell(key)
    from report.inject_format import apply_directive
    out = apply_directive(metrics[key], directive)
    if out is not None:
        return out
    from report.missing import render_missing_cell
    return render_missing_cell(key)


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
    """Return ``metrics[key]`` as a LaTeX-safe string.

    Resolves through the env-snapshot key naming used by
    :func:`report.combine_ext.merge_env`, which prefixes top-level
    fields with ``env_`` and host-info fields with ``host_``.  We
    therefore consult ``key`` first (back-compat / direct writers),
    then ``env_<key>``, then ``host_<key>`` before declaring the
    cell missing — closing the v3 regression where every Table~XIV
    cell rendered as ``---`` despite the producer running.

    Missing values render as a typed ``\\MissingCell`` (red, audited
    by ``check_artefacts``) rather than the silent ``---`` em-dash.
    """
    for candidate in (key, f"env_{key}", f"host_{key}"):
        if candidate in metrics:
            v = metrics[candidate]
            # Empty strings count as missing — an env-snapshot writer
            # logs a warning when nvidia-smi is unavailable; we don't
            # want a blank cell silently rendering as "everything ok".
            if v in (None, ""):
                continue
            return str(v)
    from report.missing import is_missing_ok, render_missing_cell
    return "\\textit{n/a}" if is_missing_ok(key) else render_missing_cell(key)


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
    """Training-summary — epochs / best / wall-clock / VRAM / USD / energy.

    Rows cover both the headline systems (DONUT, Pipeline) and each
    pipeline sub-stage (YOLO, TrOCR, assigner) so every per-stage
    ``cost_<stage>.json`` value emitted by the train-stage telemetry
    has a one-to-one consumer in Table~\\ref{tab:training}.
    """
    rows: list[str] = []
    for system in ("donut", "pipeline"):
        # ``best_epoch`` for the headline pipeline row is genuinely
        # composite (three sub-stages with three separate bests); the
        # cell therefore reads from ``pipeline_best_epoch_label`` —
        # a raw LaTeX string — when present, falling back to the
        # numeric ``{system}_best_epoch`` for DONUT's single stage.
        if system == "pipeline" and "pipeline_best_epoch_label" in metrics:
            best_cell = str(metrics["pipeline_best_epoch_label"])
        else:
            best_cell = _fmt(metrics, f"{system}_best_epoch", "int")
        rows.append(
            f"{system} "
            f"& {_fmt(metrics, f'{system}_epochs', 'int')} "
            f"& {best_cell} "
            f"& {_fmt(metrics, f'{system}_wall_clock_s', 'sig4')} "
            f"& {_fmt(metrics, f'{system}_peak_vram_gb', 'gb1')} "
            f"& {_fmt(metrics, f'{system}_cost_usd', 'usd')} "
            f"& {_fmt(metrics, f'{system}_energy_wh', 'wh')} "
            f"& {_fmt(metrics, f'{system}_co2_kg', 'sig4')} \\\\",
        )
    rows.append("\\midrule")
    for stage in ("yolo", "trocr", "assigner"):
        # Each stage HAS its own best-checkpoint epoch — YOLO via
        # Ultralytics' ``best.pt`` / ``results.csv`` argmax over
        # mAP\textsubscript{50-95}, TrOCR via HF Trainer's
        # ``load_best_model_at_end`` + ``trainer_state.best_metric``,
        # and the assigner via its early-stopping val-loss tracker.
        # ``report.best_epoch.merge_best_epochs`` extracts all three at
        # paper-stage time and folds the values into ``metrics`` under
        # ``{stage}_best_epoch`` / ``{stage}_epochs_run`` so the cells
        # below resolve to real measurements rather than ``n/a``.
        rows.append(
            f"\\quad {stage} "
            f"& {_fmt(metrics, f'{stage}_epochs_run', 'int')} "
            f"& {_fmt(metrics, f'{stage}_best_epoch', 'int')} "
            f"& {_fmt(metrics, f'{stage}_train_minutes', 'sig4')} "
            f"& {_fmt(metrics, f'{stage}_peak_vram_gb', 'gb1')} "
            f"& {_fmt(metrics, f'{stage}_cost_usd', 'usd')} "
            f"& {_fmt(metrics, f'{stage}_energy_kwh', 'sig4')} "
            f"& {_fmt(metrics, f'{stage}_co2_kg', 'sig4')} \\\\",
        )
    return (
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        "system & epochs & best & wall & VRAM & USD & energy & CO\\textsubscript{2} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )


_TABLES = {
    "table_headline_f1": render_f1_table,
    "table_extended": render_extended_table,
    "table_latency": render_latency_table,
    "table_env": render_env_table,
    "table_training": render_training_table,
}


def _competitors() -> Any:  # lazy import keeps the LOC budget here
    from report.inject_competitors import render_competitors_table
    return render_competitors_table


def inject_tables(metrics: dict[str, Any]) -> dict[str, str]:
    """Return ``{key: tabular_block}`` for every supported table emitter."""
    out = {name: fn(metrics) for name, fn in _TABLES.items()}
    out["table_competitors"] = _competitors()(metrics)
    return out
