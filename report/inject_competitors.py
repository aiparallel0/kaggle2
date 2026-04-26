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


def _fmt_f1(v: object) -> str:
    if isinstance(v, int | float) and v == v:  # noqa: PLR0124 — NaN guard
        return f"{float(v):.3f}"
    return "\\textit{n/a}"


def _fmt_params(v: object) -> str:
    if isinstance(v, int | float) and v == v:  # noqa: PLR0124
        return f"{float(v):.0f}"
    return "\\textit{n/a}"


def _resolve_this_work(row: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    """Bind the two ``this work`` rows to live combined_metrics keys."""
    notes = str(row.get("notes", ""))
    if "donut_f1" in notes:
        v = metrics.get("donut_f1")
        return float(v) if isinstance(v, int | float) else None
    if "pipeline_f1" in notes:
        v = metrics.get("pipeline_f1")
        return float(v) if isinstance(v, int | float) else None
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
        rows.append(
            f"{sys_lbl} & {_fmt_params(entry.get('params_m'))} & "
            f"{_fmt_f1(f1_val)} & {cite} \\\\"
        )
    if not rows:
        return ""
    return (
        "\\begin{tabular}{lrcl}\n\\toprule\n"
        "system & params (M) & F1 & source \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )
