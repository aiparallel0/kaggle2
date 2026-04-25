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
from core.manifest import write_manifest
from core.types import ExpConfig
from report.best_epoch import merge_best_epochs
from report.combine import (
    merge_assigner_metrics,
    merge_cost_json,
    merge_pipeline_diagnostics,
)
from report.combine_ext import (
    merge_ablations,
    merge_assigner_diag,
    merge_donut_diag,
    merge_env,
    merge_extended_metrics,
    merge_latency,
    merge_trocr_diagnostics,
    merge_yolo_diagnostics,
)
from report.combine_new import (
    merge_ablation_report,
    merge_foundation_metrics,
    merge_rag_metrics,
)
from report.inject import collect_unresolved, expand_inputs, inject_results
from report.pdflatex import compile_paper_pdf

log = logging.getLogger("kaggle2")


def _warn_missing_artifacts(config: ExpConfig) -> None:
    """Emit one consolidated INFO log if pipeline artifacts are absent."""
    results = Path(config.output_dir)
    missing = []
    if not (results / "training_log.json").exists():
        missing.append("training_log.json (run: train)")
    has_attn = (results / "attention_samples.npz").exists() or \
               (results / "attention_samples.json").exists()
    if not has_attn:
        missing.append("attention_samples.npz (run: eval)")
    if not (results / "pipeline_metrics.json").exists():
        missing.append("pipeline_metrics.json (run: eval)")
    if missing:
        log.info(
            "Some figures will be skipped — missing artifacts: %s",
            "; ".join(missing),
        )


def _render_figures(config: ExpConfig) -> None:
    """Drive every figure emitter across the ``report.figures_*`` modules.

    Never propagates exceptions: a missing matplotlib, missing source
    JSON, or corrupt telemetry log must not abort the paper stage.
    Section-C emitters are orchestrated from ``figures_section_c`` to
    keep this function under the 166-LOC cap.
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
    from report.figures_section_c import render_section_c
    render_section_c(Path(config.output_dir))


def _seed_bug_timeline_fixture(config: ExpConfig) -> None:
    """Copy the shipped ``results/bug_timeline.json`` fixture into ``output_dir``.

    ``./results/`` is the repo's fixtures-only directory (tracked in
    git).  The figure emitter in :mod:`report.figures_bugs` reads from
    ``config.output_dir`` (which, under the runs_root/run_id layout,
    points at ``runs/<run_id>/``), so we copy the fixture there at
    paper-stage start.  No-op when the fixture is already present in
    the output directory — a previous paper-stage run is authoritative.
    """
    src = Path(__file__).resolve().parents[1] / "results" / "bug_timeline.json"
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
    _warn_missing_artifacts(config)
    _render_figures(config)
    merge_cost_json(config, metrics)
    merge_assigner_metrics(config, metrics)
    merge_pipeline_diagnostics(config, metrics)
    merge_yolo_diagnostics(config, metrics)
    merge_trocr_diagnostics(config, metrics)
    merge_assigner_diag(config, metrics)
    merge_donut_diag(config, metrics)
    merge_latency(config, metrics)
    merge_extended_metrics(config, metrics)
    merge_env(config, metrics)
    merge_ablations(config, metrics)
    merge_ablation_report(config, metrics)
    merge_foundation_metrics(config, metrics)
    merge_rag_metrics(config, metrics)
    # v4 — surface each stage's own best epoch (YOLO, TrOCR, assigner)
    # so the training table no longer prints ``\\textit{n/a}`` for
    # those cells.  Idempotent / best-effort.
    merge_best_epochs(config, metrics)
    # Materialise auto-generated tabular blocks for the new Section-D
    # tables.  Each key is a ``table_*`` identifier the LaTeX section
    # files reference as ``\\VAR{table_headline_f1}`` etc. — so the
    # table body is sourced from the real metrics dict rather than
    # hand-coded literals.
    from report.inject_tables import inject_tables
    for key, tabular in inject_tables(metrics).items():
        metrics[key] = tabular
    with open(config.paper_template) as f:
        template = f.read()
    # Inline \input{sections/...} before \VAR{} substitution so the
    # 166-LOC cap applies per section file while tectonic still sees
    # a single flat paper_filled.tex at compile time.
    template = expand_inputs(template, Path(config.paper_template).parent)
    # Audit: enumerate every unresolved \VAR{} BEFORE inject_results
    # collapses them to "---", and write the audit side-channel so the
    # "no placeholders after a successful run" contract is verifiable.
    unresolved = collect_unresolved(template, metrics)
    metrics_dir = Path(config.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "unresolved_vars.json").open("w") as f:
        json.dump({"unresolved": unresolved, "count": len(unresolved)}, f, indent=2)
    if unresolved:
        log.warning(
            "stage_paper: %d unresolved \\VAR{} keys render as \\MissingCell{key} "
            "in the PDF. See metrics/unresolved_vars.json for the full list.",
            len(unresolved),
        )
    filled = inject_results(template, metrics)
    tex_out = Path(config.paper_output)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_out, "w") as f:
        f.write(filled)
    log.info("Paper LaTeX written to %s", tex_out)
    bib_src = Path(config.paper_template).parent / "references.bib"
    try:
        pdf = compile_paper_pdf(tex_out, bib_src)
        if pdf is not None:
            log.info("Paper PDF written to %s", pdf)
    except EvalError as exc:
        log.warning(
            "PDF compilation failed — LaTeX source preserved at %s. "
            "Run tectonic manually to diagnose. Error: %s",
            tex_out, exc,
        )
    # MANIFEST.json is the definitive "what to download" index for
    # operators pulling the run off vast.ai (see scripts/pack_run.sh).
    run_dir = Path(config.output_dir)
    manifest_path = write_manifest(run_dir, run_dir.name)
    log.info("Run manifest written to %s", manifest_path)
