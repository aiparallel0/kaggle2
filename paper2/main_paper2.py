"""Paper 2 system entry point.

Loads the Paper 2 configuration (FOCUS-disabled, rules+zone_prior on)
and dispatches train / eval / paper / all stages.  Writes all artefacts
under ``paper2/runs/<run_id>/``.

Usage:
    python paper2/main_paper2.py --stage all
    python paper2/main_paper2.py --stage eval
    python paper2/main_paper2.py --stage paper
    python paper2/main_paper2.py --stage eval_rule_gtocr   # CPU-only smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable so ``core``, ``data``, ``models`` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper2.config_paper2 import load_paper2_config

log = logging.getLogger("paper2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main_paper2() -> int:
    parser = argparse.ArgumentParser(prog="paper2")
    parser.add_argument(
        "--config", default=None,
        help="Path to a Paper 2 JSON config (default: paper2/configs/default.json).",
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["train", "eval", "paper", "all", "eval_rule_gtocr"],
    )
    args = parser.parse_args()
    cfg = load_paper2_config(args.config)
    log.info("paper2: stage=%s, paper_variant=%s, output_dir=%s",
             args.stage, cfg.paper_variant, cfg.output_dir)
    # Dispatch to the shared stage orchestrators (which read cfg flags
    # and route to the rule-based pipeline because FOCUS is off).
    from stages.eval import stage_eval, stage_eval_gtocr_rulebased
    from stages.paper import stage_paper
    from stages.train import stage_train

    if args.stage == "train":
        stage_train(cfg)
    elif args.stage == "eval":
        stage_eval(cfg)
    elif args.stage == "eval_rule_gtocr":
        stage_eval_gtocr_rulebased(cfg)
    elif args.stage == "paper":
        stage_paper(cfg)
    elif args.stage == "all":
        stage_train(cfg)
        stage_eval(cfg)
        stage_paper(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main_paper2())
