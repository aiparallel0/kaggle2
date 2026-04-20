"""``parity`` sub-command — run eval_pipeline on the full split."""
from __future__ import annotations

import argparse
import json
import logging

from core.config import load_config
from data.sroie import download_sroie, load_or_create_split
from models.pipeline_eval import eval_pipeline

log = logging.getLogger("inspect")


def _run_parity(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    receipts = getattr(data, args.split)
    log.info("Running eval_pipeline on %d %s receipts...", len(receipts), args.split)
    pm = eval_pipeline(config, receipts)
    report = {
        "split": args.split,
        "assigner_global_f1": pm.assigner.global_f1,
        "assigner_per_field_f1": pm.assigner.per_field_f1,
        "rulebased_global_f1": pm.rulebased.global_f1,
        "rulebased_per_field_f1": pm.rulebased.per_field_f1,
    }
    log.info("RESULT (via eval_pipeline — same function main.py calls):")
    print(json.dumps(report, indent=2))


def add_parity(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "parity",
        help="Run eval_pipeline on the full split and print aggregate F1.",
        description=(
            "Invoke models.pipeline_eval.eval_pipeline on the full split "
            "— the same function main.py --stage eval uses — and print "
            "the global + per-field F1 numbers for both the learned "
            "assigner and the rule-based baseline. Use to confirm a "
            "reported F1 is reproducible without running the 23-minute "
            "DONUT stage."
        ),
    )
    p.add_argument("--config", default="config.json")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which split to evaluate (default test).")
    p.set_defaults(func=_run_parity)
