"""Paper 3 system entry point.

Loads the Paper 3 configuration (full SVKIE multi-prior framework
engaged) and dispatches train / eval / paper / all stages.  Writes
all artefacts under ``paper3/runs/<run_id>/``.

Usage:
    python paper3/main_paper3.py --stage all
    python paper3/main_paper3.py --stage eval
    python paper3/main_paper3.py --stage paper
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable so ``core``, ``data``, ``models`` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper3.config_paper3 import load_paper3_config

log = logging.getLogger("paper3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main_paper3() -> int:
    parser = argparse.ArgumentParser(prog="paper3")
    parser.add_argument(
        "--config", default=None,
        help="Path to a Paper 3 JSON config (default: paper3/configs/default.json).",
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["train", "eval", "paper", "all"],
    )
    args = parser.parse_args()
    cfg = load_paper3_config(args.config)
    log.info("paper3: stage=%s, paper_variant=%s, output_dir=%s",
             args.stage, cfg.paper_variant, cfg.output_dir)
    # Dispatch to the shared stage orchestrators (which read cfg flags
    # and route through the FOCUS pipeline because all FOCUS knobs are on).
    from stages.eval import stage_eval
    from stages.paper import stage_paper
    from stages.train import stage_train

    if args.stage == "train":
        stage_train(cfg)
    elif args.stage == "eval":
        stage_eval(cfg)
    elif args.stage == "paper":
        stage_paper(cfg)
    elif args.stage == "all":
        stage_train(cfg)
        stage_eval(cfg)
        stage_paper(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main_paper3())
