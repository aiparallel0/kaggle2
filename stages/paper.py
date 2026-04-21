"""Paper-generation stage: enrich metrics, inject \\VAR{}, compile PDF.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: turns the authoritative ``combined_metrics.json`` (plus the
    auxiliary ``assigner_metrics.json``, ``pipeline_metrics.json``,
    ``pipeline_meta.json``, and ``cost_*.json`` sidecars) into a single
    ``paper_filled.tex`` and — when tectonic/pdflatex is installed —
    an IEEE-conference-formatted PDF, driving every figure emitter in
    :mod:`report.figures` on the way.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from core.errors import EvalError
from core.types import ExpConfig
from report.combine import (
    merge_assigner_metrics,
    merge_cost_json,
    merge_pipeline_diagnostics,
)
from report.inject import expand_inputs, inject_results
from report.pdflatex import compile_paper_pdf

log = logging.getLogger("kaggle2")


def _render_figures(config: ExpConfig) -> None:
    """Drive every figure emitter in ``report.figures.render_all``.

    Never propagates exceptions: a missing matplotlib, missing source
    JSON, or corrupt telemetry log must not abort the paper stage.
    """
    try:
        from report.figures import render_all as _render_all
        _render_all(config.output_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("figures.render_all failed (%s) — continuing.", exc)


def stage_paper(config: ExpConfig) -> None:
    """Render figures, enrich metrics, inject \\VAR{}, compile PDF."""
    log.info("=== Stage: paper ===")
    metrics_path = os.path.join(config.output_dir, "combined_metrics.json")
    if not Path(metrics_path).exists():
        raise EvalError(f"Run eval stage first — {metrics_path} not found.")
    with open(metrics_path) as f:
        metrics: dict[str, object] = json.load(f)
    _render_figures(config)
    merge_cost_json(config, metrics)
    merge_assigner_metrics(config, metrics)
    merge_pipeline_diagnostics(config, metrics)
    with open(config.paper_template) as f:
        template = f.read()
    # Inline \input{sections/...} before \VAR{} substitution so the
    # 166-LOC cap applies per section file while tectonic still sees
    # a single flat paper_filled.tex at compile time.
    template = expand_inputs(template, Path(config.paper_template).parent)
    filled = inject_results(template, metrics)
    tex_out = Path(config.paper_output)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_out, "w") as f:
        f.write(filled)
    log.info("Paper LaTeX written to %s", tex_out)
    bib_src = Path(config.paper_template).parent / "references.bib"
    pdf = compile_paper_pdf(tex_out, bib_src)
    if pdf is not None:
        log.info("Paper PDF written to %s", pdf)
