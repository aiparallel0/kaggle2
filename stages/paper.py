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
    """Drive every figure emitter across the four ``report.figures_*`` modules.

    Never propagates exceptions: a missing matplotlib, missing source
    JSON, or corrupt telemetry log must not abort the paper stage.
    The four-module split is purely a consequence of the 166-LOC cap
    per file — each module stays under the ceiling and is independently
    unit-testable.
    """
    try:
        from report.figures import render_all as _render_all
        from report.figures_attn import render_attention_heatmap
        from report.figures_bugs import render_all_bugs_telemetry
        from report.figures_extra import render_all_extra
        _render_all(config.output_dir)
        render_attention_heatmap(config.output_dir, config.output_dir)
        render_all_extra(config.output_dir, config.output_dir)
        render_all_bugs_telemetry(config.output_dir, config.output_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("figure rendering failed (%s) — continuing.", exc)


def _seed_bug_timeline_fixture(config: ExpConfig) -> None:
    """Copy the shipped ``results/bug_timeline.json`` into ``output_dir``.

    The figure emitter in :mod:`report.figures_bugs` reads from
    ``config.output_dir``; the shipped fixture lives at the repo
    root so the paper can be rebuilt deterministically.  No-op when
    the fixture is already present in the output directory.
    """
    src = Path(__file__).resolve().parent.parent / "results" / "bug_timeline.json"
    dst = Path(config.output_dir) / "bug_timeline.json"
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text())


def stage_paper(config: ExpConfig) -> None:
    """Render figures, enrich metrics, inject \\VAR{}, compile PDF."""
    log.info("=== Stage: paper ===")
    metrics_path = os.path.join(config.output_dir, "combined_metrics.json")
    if not Path(metrics_path).exists():
        raise EvalError(f"Run eval stage first — {metrics_path} not found.")
    with open(metrics_path) as f:
        metrics: dict[str, object] = json.load(f)
    _seed_bug_timeline_fixture(config)
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
