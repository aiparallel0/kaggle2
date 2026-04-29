"""kaggle2 CLI orchestrator: --stage train | eval | paper | all.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: entry point for training DONUT (~200M) and the YOLOv8n+TrOCR+Attention
    pipeline, evaluating both systems, and generating the IEEE paper PDF.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from core.config import load_config
from core.seed import seed_everything
from stages import stage_eval, stage_eval_gtocr_rulebased, stage_paper, stage_train


def _parse_seeds(value: str) -> list[int]:
    return [int(s) for s in value.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="kaggle2 KIE pipeline")
    parser.add_argument(
        "--stage",
        choices=["train", "train_backbone", "train_assigner",
                 "eval", "eval_rule_gtocr",
                 "ablate_bugs", "paper", "all"],
        default="all",
    )
    parser.add_argument("--config", default="configs/default.json")
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
    parser.add_argument(
        "--oracle-address",
        action="store_true",
        help="Day-1 diagnostic: emit oracle_address.json (Tier A box-file "
        "ceiling + Tier B canonical-347 heuristic ceiling) and exit "
        "before running the multi-seed eval loop.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override ``runs_root`` from config.json (e.g. '/mnt/vast/runs'). "
        "The effective output directory becomes <runs_root>/<run_id>/.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the auto-derived <UTC-timestamp>-<git-sha> run_id. "
        "Use this to resume a specific run or to pin a human-readable name.",
    )
    parser.add_argument(
        "--paper-variant",
        choices=["focus", "baseline"],
        default=None,
        help="Select the paper template: 'focus' (default, 626 train + 347 "
        "canonical test, DONUT vs YOLO+TrOCR+Assigner) or 'baseline' (500/63/63 "
        "internal split, DONUT vs YOLO+TrOCR+regex vs GT-OCR+regex baseline). "
        "Overrides config.paper_variant; flips canonical_sroie_enabled to "
        "match.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Flags apply before load_config so derive_paths sees them.
    if args.runs_root is not None:
        os.environ["KAGGLE2_RUNS_ROOT"] = args.runs_root
    if args.run_id is not None:
        os.environ["KAGGLE2_RUN_ID"] = args.run_id
    if args.paper_variant is not None:
        os.environ["KAGGLE2_PAPER_VARIANT"] = args.paper_variant
    os.environ["KAGGLE2_CONFIG_PATH"] = args.config
    # Pre-read raw config once so auto-resume and load_config share the same
    # I/O.  load_config re-parses it below; this peek only extracts runs_root.
    raw_cfg: dict[str, object] = {}
    try:
        with open(args.config) as _cfg_fh:
            raw_cfg = json.load(_cfg_fh)
    except (OSError, json.JSONDecodeError) as _exc:
        logging.getLogger("kaggle2").warning(
            "Could not pre-read %s for auto-resume: %s", args.config, _exc,
        )
    # Auto-resume: when running eval or paper without an explicit --run-id,
    # automatically target the latest existing run directory so that
    # ``python main.py --stage eval`` continues from where training left off.
    # The user may always override with --run-id or KAGGLE2_RUN_ID.
    if (
        args.stage in ("eval", "paper")
        and args.run_id is None
        and "KAGGLE2_RUN_ID" not in os.environ
    ):
        _runs_root = (
            os.environ.get("KAGGLE2_RUNS_ROOT")
            or raw_cfg.get("runs_root")
        )
        if _runs_root:
            from core.runlayout import latest_run as _latest_run
            _latest = _latest_run(str(_runs_root))
            if _latest:
                os.environ["KAGGLE2_RUN_ID"] = _latest.name
                logging.getLogger("kaggle2").info(
                    "Auto-resuming latest run: %s "
                    "(pass --run-id to target a different run)",
                    _latest.name,
                )
            else:
                logging.getLogger("kaggle2").warning(
                    "Auto-resume: no existing run directories found under %s; "
                    "a new run_id will be created.", _runs_root,
                )
    config = load_config(args.config)
    if args.skip_donut:
        config.skip_donut = True
        config.kd_attn_weight = 0.0    # no DONUT teacher → KD disabled
        config.kd_logits_weight = 0.0  # no DONUT teacher → KD disabled
        logging.getLogger("kaggle2").info(
            "--skip-donut: kd_attn_weight and kd_logits_weight forced to 0.0 "
            "(no DONUT teacher available)."
        )
    # CLI --paper-variant flips both the template choice (via env in
    # load_config) AND the canonical_sroie_enabled toggle so the basic
    # variant always evaluates on the 500/63/63 internal split.
    if args.paper_variant == "baseline":
        config.paper_variant = "baseline"
        config.canonical_sroie_enabled = False
    elif args.paper_variant == "focus":
        config.paper_variant = "focus"
        config.canonical_sroie_enabled = True
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
    if args.stage == "train_backbone":
        from stages.train_decomposed import stage_train_backbone
        stage_train_backbone(config)
    if args.stage == "train_assigner":
        from stages.train_decomposed import stage_train_assigner_only
        stage_train_assigner_only(config)
    if args.stage in ("eval", "all"):
        stage_eval(config, seeds=seeds, oracle_address=args.oracle_address)
    if args.stage == "eval_rule_gtocr":
        stage_eval_gtocr_rulebased(config)
    if args.stage == "ablate_bugs":
        from stages.ablate_bugs import stage_ablate_bugs
        stage_ablate_bugs(config)
    if args.stage in ("paper", "all"):
        stage_paper(config)


if __name__ == "__main__":
    main()
