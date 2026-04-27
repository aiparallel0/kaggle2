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


def _is_canonical(metrics: dict[str, Any]) -> bool:
    """A1: canonical-strip — drop rulebased columns/rows entirely."""
    return metrics.get("test_set_kind") == "canonical_347"


def render_f1_table(metrics: dict[str, Any]) -> str:
    """Headline DONUT vs Pipeline (vs Rule-based on basic variant) F1.

    On the canonical_347 advanced variant the rule-based column is
    dropped entirely (no GT boxes to feed it) — A1.  On every other
    test_set_kind the rule-based column is rendered as before.
    """
    canonical = _is_canonical(metrics)
    rows: list[str] = []
    for f in _FIELDS:
        cells = (
            f"{f} & {_fmt(metrics, f'donut_f1_{f}', 'pct1')} "
            f"& {_fmt(metrics, f'pipeline_f1_{f}', 'pct1')}"
        )
        if not canonical:
            cells += f" & {_fmt(metrics, f'rulebased_f1_{f}', 'pct1')}"
        rows.append(cells + " \\\\")
    macro = (
        "\\textbf{macro} "
        f"& \\textbf{{{_fmt(metrics, 'donut_f1', 'pct1')}}} "
        f"& \\textbf{{{_fmt(metrics, 'pipeline_f1', 'pct1')}}}"
    )
    if not canonical:
        macro += f" & \\textbf{{{_fmt(metrics, 'gtocr_rulebased_f1', 'pct1')}}}"
    macro += " \\\\"
    body = "\n".join(rows + [macro])
    if canonical:
        return (
            "\\begin{tabular}{lcc}\n\\toprule\n"
            "field & DONUT & Pipeline \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}"
        )
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


_BIB_KEYS_CACHE: frozenset[str] | None = None


def _bib_keys() -> frozenset[str]:
    """Read citation keys from ``report/references.bib`` (cached).

    Used by :func:`_sanitise_tabular` to verify every ``\\cite{}`` in
    an emitted tabular resolves to a known bib entry — silent
    ``[?]`` markers in the PDF are otherwise indistinguishable from
    typeset content (Audit A2 / Issues 5/6/9).
    """
    global _BIB_KEYS_CACHE  # noqa: PLW0603 — module-level cache, not state
    if _BIB_KEYS_CACHE is not None:
        return _BIB_KEYS_CACHE
    import re as _re
    from pathlib import Path
    bib = Path(__file__).resolve().parent / "references.bib"
    if not bib.exists():
        _BIB_KEYS_CACHE = frozenset()
        return _BIB_KEYS_CACHE
    text = bib.read_text(encoding="utf-8", errors="replace")
    keys = _re.findall(r"^\s*@\w+\s*\{\s*([^,\s]+)", text, flags=_re.MULTILINE)
    _BIB_KEYS_CACHE = frozenset(keys)
    return _BIB_KEYS_CACHE


def _sanitise_tabular(s: str) -> str:
    """Validate an emitted ``\\begin{tabular}…\\end{tabular}`` block.

    Raises :class:`core.errors.EvalError` (a hard build failure, not a
    silent render-corrupt — Audit A2) if any of the following invariants
    is violated:

    * Balanced ``{`` / ``}`` braces (modulo escaped ``\\{`` / ``\\}``).
    * Every ``\\cite{key}`` resolves against ``report/references.bib``;
      a stray cite would otherwise typeset as ``[?]`` in the PDF.
    * No stray ``\\VAR{}`` placeholders — every key the emitter
      referenced was either resolved or routed through the
      :func:`_fmt` ``\\MissingCell``/``\\textit{n/a}`` backstop.

    Empty strings pass through (a deliberately-empty emitter under a
    non-applicable variant — see e.g. ``table_competitors`` outside
    canonical_347).  Returns the input unchanged on success.
    """
    if not s:
        return s
    import re as _re

    from core.errors import EvalError
    # Balanced braces — count unescaped { vs }.
    stripped = _re.sub(r"\\[{}]", "", s)
    if stripped.count("{") != stripped.count("}"):
        raise EvalError(
            "tabular validation failed: unbalanced braces "
            f"({stripped.count('{')} '{{' vs {stripped.count('}')} '}}'); "
            "check the emitter for un-escaped '%' or stray '{' characters.",
        )
    if "\\VAR{" in s:
        raise EvalError(
            "tabular validation failed: stray \\VAR{} placeholder in "
            "emitted block — every key must resolve through _fmt() or "
            "render as \\MissingCell{}/\\textit{n/a}.",
        )
    cites = set(_re.findall(r"\\cite\{([^}]+)\}", s))
    bib = _bib_keys()
    if bib:
        # Comma-separated cites are valid LaTeX (e.g. \cite{a,b}).
        flat = {k.strip() for group in cites for k in group.split(",")}
        unresolved = sorted(k for k in flat if k and k not in bib)
        if unresolved:
            raise EvalError(
                "tabular validation failed: \\cite{} keys do not resolve "
                f"against references.bib: {unresolved[:5]}",
            )
    return s


def _competitors() -> Any:  # lazy import keeps the LOC budget here
    from report.inject_competitors import render_competitors_table
    return render_competitors_table


def render_bug_ablation_table(metrics: dict[str, Any]) -> str:
    """Per-bug ΔF1 table sourced from ``results/bug_timeline.json``.

    Renders the 13-bug timeline as a tabular for ``\\VAR{table_bug_ablation}``.
    ΔF1 is computed at emit-time as ``ceiling - f1_before`` (Audit B2;
    matches :func:`report.combine_new._emit_from_bug_timeline`).  bug_7
    (val/test leakage) is an *over-reporting* bug, so its ΔF1 is
    negative — the caption (in ``bugs.tex``) explains the sign.

    The fixture file ``results/bug_timeline.json`` is the source of
    truth; the function reads it directly so the table can render
    even before ``stage_paper`` populates the per-bug merger keys.
    """
    import json as _json
    from pathlib import Path as _Path
    fixture = _Path(__file__).resolve().parents[1] / "results" / "bug_timeline.json"
    if not fixture.exists():
        return ""
    try:
        data: dict[str, Any] = _json.loads(fixture.read_text())
    except (OSError, ValueError):
        return ""
    bugs = data.get("bugs") or []
    ceiling = (
        metrics.get("pipeline_f1")
        or metrics.get("donut_f1")
        or data.get("f1_after_default")
        or 0.0
    )
    try:
        ceiling_f = float(ceiling)
    except (TypeError, ValueError):
        ceiling_f = 0.0
    rows: list[str] = []
    for entry in bugs:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        before = entry.get("f1_before")
        if not isinstance(idx, int) or not isinstance(before, int | float):
            continue
        delta = ceiling_f - float(before)
        # bug_7 (val/test leakage) over-reports F1 above ceiling, so
        # ``delta`` is negative — render with an ASCII minus inside
        # math mode so tectonic typesets a true unicode minus.  The
        # caption in ``bugs.tex`` documents the sign convention.
        rows.append(
            f"bug\\_{idx} & ${delta:.4f}$ "
            f"& $[{delta:.4f},\\,{delta:.4f}]$ \\\\",
        )
    if not rows:
        return ""
    return (
        "\\begin{tabular}{llc}\\toprule\n"
        "Bug & $\\Delta$F1 & 95\\% CI \\\\ \\midrule\n"
        + "\n".join(rows)
        + f"\n\\midrule\nceiling & {ceiling_f:.4f} & --- \\\\\n"
        "\\bottomrule\\end{tabular}"
    )


def inject_tables(metrics: dict[str, Any]) -> dict[str, str]:
    """Return ``{key: tabular_block}`` for every supported table emitter.

    Every emitted block is passed through :func:`_sanitise_tabular`
    before being handed to the inject layer — Audit A2 build failure
    semantics: a malformed tabular raises ``EvalError`` here instead
    of silently corrupting the PDF inside an outer ``\\begin{tabular}``
    slot.
    """
    out = {name: _sanitise_tabular(fn(metrics)) for name, fn in _TABLES.items()}
    out["table_competitors"] = _sanitise_tabular(_competitors()(metrics))
    out["table_bug_ablation"] = _sanitise_tabular(render_bug_ablation_table(metrics))
    return out
