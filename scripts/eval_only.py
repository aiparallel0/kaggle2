"""One-shot pipeline-eval harness on the existing checkpoint.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: loads the existing ``./results/assigner.pt`` + TrOCR + YOLO
    checkpoints, runs :func:`eval_pipeline` and :func:`eval_gtocr_rulebased`
    on the 63-image SROIE test split, prints the per-field F1 breakdown
    via :mod:`models.pipeline_miss_tracker`, and writes the flat
    ``./results/pipeline_metrics.json`` — **without** calling
    :func:`assert_hybrid_beats_gtocr_rulebased`, so the script always
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
from models.rule_eval import eval_gtocr_rulebased

log = logging.getLogger("kaggle2")


def main(argv: list[str] | None = None) -> int:
    """Run pipeline + GT-OCR-stream-rulebased eval on the current checkpoints."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = (argv or sys.argv[1:] + ["config.json"])[0]
    config = load_config(config_path)
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("eval_only: %d test receipts", len(data.test))
    pm = eval_pipeline(config, data.test)
    log.info("Pipeline (hybrid)              F1=%.4f", pm.assigner.global_f1)
    log.info("Pipeline (TrOCR-regex)         F1=%.4f", pm.rulebased.global_f1)
    gtocr_rb = eval_gtocr_rulebased(config, data.test)
    log.info("Baseline (GT-OCR-stream regex) F1=%.4f", gtocr_rb.global_f1)
    log.info(
        "eval_only: skipping assert_hybrid_beats_gtocr_rulebased so this "
        "script always terminates and writes pipeline_metrics.json for "
        "copy-paste iteration.  Run `python main.py --stage eval` for the "
        "full hard-gated evaluation.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
