"""kaggle2 CLI orchestrator: --stage train | eval | paper | all.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: entry point for training DONUT (~200M) and the YOLOv8n+TrOCR+Attention
    pipeline, evaluating both systems, and generating the IEEE paper PDF.
"""
from __future__ import annotations

import argparse
import logging

from core.config import load_config
from core.seed import seed_everything
from stages import stage_eval, stage_eval_rulebased_gold, stage_paper, stage_train


def _parse_seeds(value: str) -> list[int]:
    return [int(s) for s in value.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="kaggle2 KIE pipeline")
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "eval_rulebased_gold", "paper", "all"],
        default="all",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--skip-donut",
        action="store_true",
        help="Skip DONUT training/eval (Phase 1 / vast.ai pipeline-only run). "
        "Overrides the 'skip_donut' key in config.json. Requires kd_*_weight=0.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds for the eval stage multi-seed harness "
        "(e.g. '42,123,2024'). Default = single run with config.seed.",
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
    # CLI --seeds wins over config.seeds; config.seeds wins over legacy config.seed.
    # config.seeds and config.n_trials are the durable way to switch to n=5 etc.
    seeds = (
        _parse_seeds(args.seeds)
        if args.seeds
        else list(config.seeds[: config.n_trials])
    )
    seed_everything(seeds[0])
    if args.stage in ("train", "all"):
        stage_train(config)
    if args.stage in ("eval", "all"):
        stage_eval(config, seeds=seeds)
    if args.stage == "eval_rulebased_gold":
        stage_eval_rulebased_gold(config)
    if args.stage in ("paper", "all"):
        stage_paper(config)


if __name__ == "__main__":
    main()
