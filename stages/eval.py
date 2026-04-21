"""Evaluation stages: full (DONUT + pipeline + rule-based) and CPU-only.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: runs the three evaluation arms (DONUT, YOLO+TrOCR+Attention,
    rule-based on gold OCR) over one or more seeds, aggregates
    mean/std when multi-seed, and writes the authoritative
    ``combined_metrics.json`` that the paper stage consumes.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
from pathlib import Path

from core.errors import EvalError
from core.seed import seed_everything
from core.types import DataSplit, ExpConfig, Metrics
from core.validate import validate_f1
from data.sroie import download_sroie, load_or_create_split
from models.donut_eval import eval_donut
from models.pipeline_eval import eval_pipeline
from models.rule_eval import (
    combined_from_rulebased,
    eval_rulebased_gold,
    per_field_injection,
)
from report.combine import build_combined
from stages._common import (
    assert_pipeline_beats_rulebased_gold,
    warn_below_expected,
    warn_pipeline_diagnostics,
)

log = logging.getLogger("kaggle2")


def _eval_donut_or_skip(config: ExpConfig, data: DataSplit) -> Metrics:
    """Eval DONUT iff ``skip_donut`` is False AND a checkpoint exists."""
    donut_model = os.path.join(config.output_dir, "donut")
    if config.skip_donut:
        log.info("skip_donut=True — skipping DONUT eval; reporting zeros.")
        return Metrics(
            global_f1=0.0, global_ned=0.0, global_em=0.0,
            per_field_f1={f: 0.0 for f in config.fields},
            per_field_ned={f: 0.0 for f in config.fields},
            per_field_em={f: 0.0 for f in config.fields},
        )
    if not Path(donut_model).exists():
        raise EvalError(
            f"DONUT checkpoint not found at {donut_model}. Either run train "
            "stage first, or set skip_donut=true for a Phase-1 pipeline-only run.",
        )
    dm = eval_donut(config, data.test)
    validate_f1(dm, "donut")
    warn_below_expected(dm, config, "donut")
    return dm


def stage_eval(config: ExpConfig, seeds: list[int] | None = None) -> None:
    """Run eval across one or more seeds; aggregate mean/std when multi-seed.

    Keeps the legacy single-seed keys (``donut_f1``, ``pipeline_f1``,
    ``assigner_delta``) populated for backwards compatibility with the
    paper's \\VAR{} substitution.  Multi-seed runs additionally emit
    ``*_mean`` / ``*_std`` so the paper can render bootstrap-style
    uncertainty bands without re-running eval.
    """
    log.info("=== Stage: eval ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    run_seeds = seeds if seeds else [config.seed]
    donut_f1s: list[float] = []
    pipeline_f1s: list[float] = []
    last: dict[str, object] = {}
    for seed in run_seeds:
        if len(run_seeds) > 1:
            log.info("--- Eval seed=%d ---", seed)
        seed_everything(seed)
        dm = _eval_donut_or_skip(config, data)
        log.info("DONUT F1=%.4f", dm.global_f1)
        pm = eval_pipeline(config, data.test)
        validate_f1(pm.assigner, "pipeline")
        warn_below_expected(pm.assigner, config, "pipeline")
        warn_pipeline_diagnostics(config)
        log.info("Pipeline (assigner)  F1=%.4f", pm.assigner.global_f1)
        log.info("Pipeline (rulebased) F1=%.4f", pm.rulebased.global_f1)
        # Rule-based on gold OCR isolates assignment-heuristic quality
        # from OCR noise — a legitimate ablation even in full-pipeline
        # runs, so we always compute it here too.
        rb_gold = eval_rulebased_gold(config, data.test)
        log.info("Rule-based (gold OCR) F1=%.4f", rb_gold.global_f1)
        # Regression gate: a learned model on YOLO+TrOCR features must not
        # score below a heuristic on gold OCR — otherwise something is
        # wrong upstream (bad assigner checkpoint, stale TrOCR features,
        # or an eval-fairness bug).
        assert_pipeline_beats_rulebased_gold(pm.assigner, rb_gold)
        donut_f1s.append(dm.global_f1)
        pipeline_f1s.append(pm.assigner.global_f1)
        last = build_combined(config, dm, pm, rb_gold)
    if len(run_seeds) >= 2:
        last["donut_f1_mean"] = round(statistics.fmean(donut_f1s), 4)
        last["donut_f1_std"] = round(statistics.stdev(donut_f1s), 4)
        last["pipeline_f1_mean"] = round(statistics.fmean(pipeline_f1s), 4)
        last["pipeline_f1_std"] = round(statistics.stdev(pipeline_f1s), 4)
    last["seeds_used"] = list(run_seeds)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(last, f, indent=2)


def stage_eval_rulebased_gold(config: ExpConfig) -> None:
    """Rule-based F1 over SROIE gold OCR — no HF / GPU dependency.

    Writes ``results/rulebased_gold_metrics.json`` and a
    ``combined_metrics.json`` pre-populated with zeros for the DONUT /
    pipeline-learned arms so ``stage_paper`` can still compile a paper
    whose rule-based numbers are real even when the neural components
    could not be trained in the current environment.
    """
    log.info("=== Stage: eval_rulebased_gold ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("Split: %d train / %d val / %d test",
             len(data.train), len(data.val), len(data.test))
    metrics = eval_rulebased_gold(config, data.test)
    log.info("Rule-based (gold OCR) F1=%.4f  per-field=%s",
             metrics.global_f1,
             {k: round(v, 4) for k, v in metrics.per_field_f1.items()})
    combined = combined_from_rulebased(config, metrics)
    combined.update(per_field_injection(metrics))
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)
