"""kaggle2 orchestrator CLI: --stage train | eval | paper | all."""
from __future__ import annotations

import argparse
import logging

from core.config import load_config
from core.seed import seed_everything
from stages import stage_eval, stage_paper, stage_train


def main() -> None:
    parser = argparse.ArgumentParser(description="kaggle2 KIE pipeline")
    parser.add_argument(
        "--stage", choices=["train", "eval", "paper", "all"], default="all",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--skip-donut",
        action="store_true",
        help="Skip DONUT training/eval (Phase 1 / vast.ai pipeline-only run). "
        "Overrides the 'skip_donut' key in config.json. Requires kd_*_weight=0.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.skip_donut:
        config.skip_donut = True
    seed_everything(config.seed)
    if args.stage in ("train", "all"):
        stage_train(config)
    if args.stage in ("eval", "all"):
        stage_eval(config)
    if args.stage in ("paper", "all"):
        stage_paper(config)


if __name__ == "__main__":
    main()
