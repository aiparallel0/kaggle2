"""Competitors-table emitter for the ADVANCED paper variant (Table tab:competitors).

Reads ``results/sroie_task3_competitors.json`` (the editable fixture of
published SROIE Task-3 numbers), substitutes ``this work`` rows from
the live ``combined_metrics.json`` at paper-stage time, and returns a
LaTeX ``tabular`` block ready to back the ``\\VAR{table_competitors}``
substitution.  Returns ``""`` (empty string) under the BASIC variant
and any other run where ``test_set_kind`` is not ``canonical_347`` —
the table is meaningful only when our own numbers are measured on the
same 347-image test set as the published competitors.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("kaggle2")

# Resolved relative to repo root via __file__ — same convention as
# stages.paper._seed_bug_timeline_fixture.  Fixtures-only file: tracked
# in git, never written at runtime (see results/README.md).
_FIXTURE = (
    Path(__file__).resolve().parents[1] / "results" / "sroie_task3_competitors.json"
)


def _is_finite_number(v: object) -> bool:
    """True iff ``v`` is a finite int/float (rejects None, NaN, inf)."""
    if not isinstance(v, int | float) or isinstance(v, bool):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def _fmt_f1(v: object) -> str:
    if _is_finite_number(v):
        return f"{float(v):.3f}"  # type: ignore[arg-type]  # narrowed by guard
    return "\\textit{n/a}"


def _fmt_params(v: object) -> str:
    if _is_finite_number(v):
        return f"{float(v):.0f}"  # type: ignore[arg-type]  # narrowed by guard
    return "\\textit{n/a}"


def _resolve_this_work(row: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    """Bind the two ``this work`` rows to live combined_metrics keys."""
    notes = str(row.get("notes", ""))
    if "donut_f1" in notes:
        v = metrics.get("donut_f1")
        return float(v) if _is_finite_number(v) else None  # type: ignore[arg-type]
    if "pipeline_f1" in notes:
        v = metrics.get("pipeline_f1")
        return float(v) if _is_finite_number(v) else None  # type: ignore[arg-type]
    return None


def render_competitors_table(metrics: dict[str, Any]) -> str:
    """Render the published-Task-3-numbers comparison tabular.

    Returns ``""`` when the run did not evaluate on the canonical 347
    test set (so the table is not emitted under the basic variant).
    """
    if metrics.get("test_set_kind") != "canonical_347":
        return ""
    if not _FIXTURE.exists():
        log.warning("competitors fixture missing at %s — emitting empty table.", _FIXTURE)
        return ""
    try:
        data = json.loads(_FIXTURE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("competitors fixture unreadable (%s) — emitting empty table.", exc)
        return ""
    rows: list[str] = []
    has_reimpl = False
    for entry in data.get("competitors", []):
        f1_val: object = entry.get("f1")
        if f1_val is None and entry.get("source") == "this work":
            f1_val = _resolve_this_work(entry, metrics)
        cite = (
            "\\textbf{this work}"
            if entry.get("source") == "this work"
            else f"\\cite{{{entry['source']}}}"
        )
        # LaTeX-escape ampersands and underscores in the system label.
        sys_lbl = (
            str(entry.get("system", "?"))
            .replace("&", "\\&").replace("_", "\\_")
        )
        # Audit C3: distinguish the in-paper DONUT re-implementation
        # from the published DONUT numbers (Kim et al., ECCV 2022) so
        # readers do not conflate the two.  The marker ``$^{\dagger}$``
        # lands on the ``this work — DONUT`` row only; an explanatory
        # note row is appended below the tabular body.
        if "this work" in str(entry.get("system", "")) and "DONUT" in sys_lbl:
            sys_lbl = sys_lbl + "$^{\\dagger}$"
            has_reimpl = True
        rows.append(
            f"{sys_lbl} & {_fmt_params(entry.get('params_m'))} & "
            f"{_fmt_f1(f1_val)} & {cite} \\\\"
        )
    if not rows:
        return ""
    note = ""
    if has_reimpl:
        note = (
            "\\midrule\n"
            "\\multicolumn{4}{p{0.92\\linewidth}}{\\footnotesize $^{\\dagger}$"
            " ``this work --- DONUT'' is our SROIE Task-3 re-implementation"
            " of the architecture from~\\cite{kim2022donut}; the numerical"
            " gap to the originally reported DONUT figure stems from"
            " differences in pre-training data scale, image-preprocessing"
            " pipeline, and the exact SROIE Task-3 train/val partition"
            " definition (Section~\\ref{sec:discussion}).} \\\\\n"
        )
    return (
        "\\begin{tabular}{lrcl}\n\\toprule\n"
        "system & params (M) & F1 & source \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n" + note + "\\bottomrule\n\\end{tabular}"
    )
