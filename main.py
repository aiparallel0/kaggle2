"""kaggle2 orchestrator: --stage train | eval | paper | all."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from core.config import load_config
from core.errors import EvalError, TrainError
from core.seed import seed_everything
from core.types import AssignerData, ExpConfig, PipelinePaths
from data.sroie import download_sroie, extract_crops, extract_receipt_regions, split_sroie
from models.assigner_train import train_assigner
from models.donut_eval import eval_donut
from models.donut_train import train_donut
from models.pipeline_eval import eval_pipeline
from models.trocr_train import train_trocr
from models.yolo_train import train_yolo
from report.inject import inject_results

log = logging.getLogger("kaggle2")


def _validate_f1(global_f1: float, arch: str) -> None:
    """Post-eval F1 guardrails — raise if architecture-specific floor is breached."""
    if arch == "donut" and global_f1 < 0.50:
        raise TrainError(
            f"DONUT F1={global_f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), or token2json list (Bug 3).",
        )
    if arch == "pipeline" and global_f1 == 0.0:
        raise TrainError(
            "Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5) or TrOCR undertrained (Bug 6).",
        )


def _write_pipeline_meta(config: ExpConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump({"yolo_img_size": config.yolo_img_size}, f)


def _stage_train(config: ExpConfig) -> None:
    log.info("=== Stage: train ===")
    data_path = download_sroie(config)
    data = split_sroie(data_path, config.seed)
    log.info("Split: %d train / %d val / %d test",
             len(data.train), len(data.val), len(data.test))
    donut_path = train_donut(config, data)
    log.info("DONUT → %s", donut_path)
    yolo_path = train_yolo(config, data)
    log.info("YOLO  → %s", yolo_path)
    crops = extract_crops(data.train, config.fields)
    regions = extract_receipt_regions(data.train, config.fields)
    log.info("Extracted %d labeled crops / %d receipt region-groups",
             len(crops), len(regions))
    if not crops:
        raise TrainError("No labeled SROIE crops — check box/ annotations.")
    trocr_path = train_trocr(config, crops)
    log.info("TrOCR → %s", trocr_path)
    assigner_data = AssignerData(trocr_path=trocr_path, crops=crops, regions=regions)
    assigner_path = train_assigner(config, assigner_data)
    log.info("Assigner → %s", assigner_path)
    _write_pipeline_meta(config)


def _stage_eval(config: ExpConfig) -> None:
    log.info("=== Stage: eval ===")
    data_path = download_sroie(config)
    data = split_sroie(data_path, config.seed)
    donut_model = os.path.join(config.output_dir, "donut")
    dm = eval_donut(donut_model, data.test)
    _validate_f1(dm.global_f1, "donut")
    log.info("DONUT F1=%.4f", dm.global_f1)
    paths = PipelinePaths(
        yolo=os.path.join(config.output_dir, "yolo", "run", "weights", "best.pt"),
        trocr=os.path.join(config.output_dir, "trocr"),
        assigner=os.path.join(config.output_dir, "assigner.pt"),
    )
    pm = eval_pipeline(paths, data.test, config)
    _validate_f1(pm.assigner.global_f1, "pipeline")
    log.info("Pipeline (assigner)  F1=%.4f", pm.assigner.global_f1)
    log.info("Pipeline (rulebased) F1=%.4f", pm.rulebased.global_f1)
    combined: dict[str, object] = {
        "donut_f1": dm.global_f1, "donut_ned": dm.global_ned, "donut_em": dm.global_em,
        "pipeline_f1": pm.assigner.global_f1,
        "pipeline_ned": pm.assigner.global_ned,
        "pipeline_em": pm.assigner.global_em,
        "rulebased_f1": pm.rulebased.global_f1,
        "rulebased_ned": pm.rulebased.global_ned,
        "f1_gap": round(dm.global_f1 - pm.assigner.global_f1, 4),
        "assigner_delta": round(pm.assigner.global_f1 - pm.rulebased.global_f1, 4),
        "donut_f1_company": dm.per_field_f1.get("company", 0.0),
        "donut_f1_date": dm.per_field_f1.get("date", 0.0),
        "donut_f1_address": dm.per_field_f1.get("address", 0.0),
        "donut_f1_total": dm.per_field_f1.get("total", 0.0),
        "epochs_donut": config.epochs_donut, "epochs_trocr": config.epochs_trocr,
        "epochs_yolo": config.epochs_yolo, "batch_size": config.batch_size,
        "lr": config.lr, "precision": config.precision,
        "label_smoothing": config.label_smoothing,
        "yolo_img_size": config.yolo_img_size,
        "img_w": config.image_size[0], "img_h": config.image_size[1],
    }
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)


def _stage_paper(config: ExpConfig) -> None:
    log.info("=== Stage: paper ===")
    metrics_path = os.path.join(config.output_dir, "combined_metrics.json")
    if not Path(metrics_path).exists():
        raise EvalError(f"Run eval stage first — {metrics_path} not found.")
    with open(metrics_path) as f:
        metrics: dict[str, object] = json.load(f)
    with open(config.paper_template) as f:
        template = f.read()
    filled = inject_results(template, metrics)
    Path(config.paper_output).parent.mkdir(parents=True, exist_ok=True)
    with open(config.paper_output, "w") as f:
        f.write(filled)
    log.info("Paper written to %s", config.paper_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="kaggle2 KIE pipeline")
    parser.add_argument(
        "--stage", choices=["train", "eval", "paper", "all"], default="all",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    seed_everything(config.seed)
    if args.stage in ("train", "all"):
        _stage_train(config)
    if args.stage in ("eval", "all"):
        _stage_eval(config)
    if args.stage in ("paper", "all"):
        _stage_paper(config)


if __name__ == "__main__":
    main()
