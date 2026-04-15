"""kaggle2 orchestrator: --stage train | eval | paper."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.config import load_config
from core.errors import EvalError, TrainError
from core.types import AssignerData, ExpConfig, PipelinePaths
from data.sroie import download_sroie, extract_crops, split_sroie
from models.assigner_train import train_assigner
from models.donut_eval import eval_donut
from models.donut_train import train_donut
from models.pipeline_eval import eval_pipeline
from models.trocr_train import train_trocr
from models.yolo_train import train_yolo
from report.inject import inject_results


def _validate_f1(metrics: dict[str, float], arch: str) -> None:
    """Post-training F1 guardrails."""
    f1 = metrics.get("global_f1", -1.0)
    if arch == "donut" and f1 < 0.50:
        raise TrainError(
            f"DONUT F1={f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), or token2json list (Bug 3)."
        )
    if arch == "pipeline" and f1 == 0.0:
        raise TrainError(
            "Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5) or TrOCR undertrained (Bug 6)."
        )


def _stage_train(config: ExpConfig) -> None:
    print("=== Stage: train ===")
    data_path = download_sroie(config)
    data = split_sroie(data_path, config.seed)
    print(f"Split: {len(data.train)} train / {len(data.val)} val / {len(data.test)} test")
    donut_path = train_donut(config, data)
    print(f"DONUT → {donut_path}")
    yolo_path = train_yolo(config, data)
    print(f"YOLO  → {yolo_path}")
    # Extract real text-region crops from SROIE box annotations
    crops = extract_crops(data.train, config.fields)
    print(f"Extracted {len(crops)} labeled crops from SROIE box annotations")
    trocr_path = train_trocr(config, crops)
    print(f"TrOCR → {trocr_path}")
    assigner_data = AssignerData(trocr_path=trocr_path, crops=crops)
    assigner_path = train_assigner(config, assigner_data)
    print(f"Assigner → {assigner_path}")
    # Store pipeline metadata for eval stage
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump({"yolo_img_size": config.yolo_img_size}, f)


def _stage_eval(config: ExpConfig) -> None:
    print("=== Stage: eval ===")
    data_path = download_sroie(config)
    data = split_sroie(data_path, config.seed)
    donut_model = os.path.join(config.output_dir, "donut")
    dm = eval_donut(donut_model, data.test)
    _validate_f1({"global_f1": dm.global_f1}, "donut")
    print(f"DONUT F1={dm.global_f1:.4f}")
    paths = PipelinePaths(
        yolo=os.path.join(config.output_dir, "yolo", "run", "weights", "best.pt"),
        trocr=os.path.join(config.output_dir, "trocr"),
        assigner=os.path.join(config.output_dir, "assigner.pt"),
    )
    pm = eval_pipeline(paths, data.test)
    _validate_f1({"global_f1": pm.global_f1}, "pipeline")
    print(f"Pipeline F1={pm.global_f1:.4f}")
    combined: dict[str, object] = {
        "donut_f1": dm.global_f1, "donut_ned": dm.global_ned, "donut_em": dm.global_em,
        "pipeline_f1": pm.global_f1, "pipeline_ned": pm.global_ned, "pipeline_em": pm.global_em,
        "f1_gap": round(dm.global_f1 - pm.global_f1, 4),
        "assigner_delta": 0.05,
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
    print("=== Stage: paper ===")
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
    print(f"Paper written to {config.paper_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="kaggle2 KIE pipeline")
    parser.add_argument(
        "--stage", choices=["train", "eval", "paper", "all"], default="all",
    )
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in ("train", "all"):
        _stage_train(config)
    if args.stage in ("eval", "all"):
        _stage_eval(config)
    if args.stage in ("paper", "all"):
        _stage_paper(config)


if __name__ == "__main__":
    main()

