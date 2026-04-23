"""One-shot pipeline-eval harness on the existing checkpoint.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: loads the existing ``./results/assigner.pt`` + TrOCR + YOLO
    checkpoints, runs :func:`eval_pipeline` and :func:`eval_rulebased_gold`
    on the 63-image SROIE test split, prints the per-field F1 breakdown
    via :mod:`models.pipeline_miss_tracker`, and writes the flat
    ``./results/pipeline_metrics.json`` — **without** calling
    :func:`assert_pipeline_beats_rulebased_gold`, so the script always
    terminates and dumps metrics even when below the hard guardrail.

Usage (copy-paste ready):

    # one-shot eval of current checkpoint, no training, no hard guardrail
    python -m scripts.eval_only

    # inspect the per-field breakdown written to pipeline_metrics.json
    jq '.per_field_f1, .per_field_diagnostics' ./results/pipeline_metrics.json

    # list every per-receipt miss for a field (swap total / address / …)
    jq -r '.predictions_by_field.total[] | select(.pred != .gt) |
           [.receipt_id, .pred, .gt] | @tsv' ./results/pipeline_metrics.json
"""
from __future__ import annotations

import logging
import sys

from core.config import load_config
from data.sroie import download_sroie, load_or_create_split
from models.pipeline_eval import eval_pipeline
from models.rule_eval import eval_rulebased_gold

log = logging.getLogger("kaggle2")


def main(argv: list[str] | None = None) -> int:
    """Run pipeline + rulebased-gold eval on the current checkpoints."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = (argv or sys.argv[1:] + ["config.json"])[0]
    config = load_config(config_path)
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("eval_only: %d test receipts", len(data.test))
    pm = eval_pipeline(config, data.test)
    log.info("Pipeline (assigner)  F1=%.4f", pm.assigner.global_f1)
    log.info("Pipeline (rulebased) F1=%.4f", pm.rulebased.global_f1)
    rb_gold = eval_rulebased_gold(config, data.test)
    log.info("Rule-based (gold OCR) F1=%.4f", rb_gold.global_f1)
    log.info(
        "eval_only: skipping assert_pipeline_beats_rulebased_gold so this "
        "script always terminates and writes pipeline_metrics.json for "
        "copy-paste iteration.  Run `python main.py --stage eval` for the "
        "full hard-gated evaluation.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
