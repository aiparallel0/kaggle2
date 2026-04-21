"""diagnose sub-command: dump raw pipeline intermediates per-receipt.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: exports per-receipt YOLO boxes, TrOCR transcripts, and both
    assignment strategies to JSON for debugging silent F1-destroying bugs.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core.config import load_config
from core.metrics import compute_metrics
from core.types import EvalBundle, PipelinePaths, Prediction
from data.sroie import download_sroie, load_or_create_split
from scripts.inspect_diag_loop import inspect_receipt, load_pipeline_models

log = logging.getLogger("inspect")


def _paths(config: Any) -> PipelinePaths:
    return PipelinePaths(
        yolo=str(Path(config.output_dir) / "yolo" / "run" / "weights" / "best.pt"),
        trocr=str(Path(config.output_dir) / "trocr"),
        assigner=str(Path(config.output_dir) / "assigner.pt"),
    )


def _run_diagnose(args: argparse.Namespace) -> None:
    import torch

    config = load_config(args.config)
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    paths = _paths(config)
    yolo, proc, trocr, assigner, device = load_pipeline_models(paths, len(config.fields))
    meta_path = Path(config.output_dir) / "pipeline_meta.json"
    yolo_img = config.yolo_img_size
    if meta_path.exists():
        yolo_img = int(json.loads(meta_path.read_text()).get("yolo_img_size", yolo_img))
    log.info("yolo_img=%d conf=%.2f max_regions=%d",
             yolo_img, config.yolo_conf, config.max_regions_per_image)

    receipts = getattr(data, args.split)
    n = min(args.n, len(receipts))
    subset = receipts[:n]
    out_receipts: list[dict[str, Any]] = []
    total_boxes = total_gt_boxes = total_usable = total_reads = 0
    fallback_count = all_empty_assign = 0
    preds_l: list[Prediction] = []
    preds_r: list[Prediction] = []
    with torch.no_grad():
        for rec in subset:
            dump, pred_l, pred_r, fb, n_boxes, n_usable, n_reads = inspect_receipt(
                rec, yolo, proc, trocr, assigner, config, yolo_img, device,
            )
            total_boxes += n_boxes
            total_gt_boxes += dump["sroie_gt_n_boxes"]
            total_usable += n_usable
            total_reads += n_reads
            if fb:
                fallback_count += 1
            if not dump["assigner"]:
                all_empty_assign += 1
            preds_l.append(pred_l)
            preds_r.append(pred_r)
            out_receipts.append(dump)

    m_r = compute_metrics(EvalBundle(predictions=preds_r, receipts=subset, fields=config.fields))
    m_l = compute_metrics(EvalBundle(predictions=preds_l, receipts=subset, fields=config.fields))
    summary = {
        "n_receipts": n, "split": args.split,
        "avg_boxes_per_receipt": round(total_boxes / n, 2) if n else 0.0,
        "avg_sroie_gt_boxes_per_receipt": round(total_gt_boxes / n, 2) if n else 0.0,
        "fallback_rate": round(fallback_count / n, 3) if n else 0.0,
        "usable_region_rate": (
            round(total_usable / total_reads, 3) if total_reads else 0.0
        ),
        "all_fields_empty_rate": round(all_empty_assign / n, 3) if n else 0.0,
        "yolo_img": yolo_img, "yolo_conf": config.yolo_conf,
        "max_regions": config.max_regions_per_image,
        "subset_assigner_f1": round(m_l.global_f1, 4),
        "subset_rulebased_f1": round(m_r.global_f1, 4),
        "subset_assigner_per_field_f1": {
            k: round(v, 4) for k, v in m_l.per_field_f1.items()
        },
        "subset_rulebased_per_field_f1": {
            k: round(v, 4) for k, v in m_r.per_field_f1.items()
        },
    }
    report = {"summary": summary, "receipts": out_receipts}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Wrote %s", out_path)
    print(json.dumps(summary, indent=2))


def add_diagnose(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "diagnose",
        help="Dump raw pipeline intermediates for the first N receipts.",
        description=(
            "Dump per-receipt YOLO boxes, TrOCR transcriptions, and both "
            "assignment strategies (learned + rule-based) for the first N "
            "test receipts. Writes a JSON report and prints aggregate "
            "health signals (avg boxes/receipt, TrOCR empty rate, fallback "
            "rate, per-field F1 on the inspected subset)."
        ),
    )
    p.add_argument("--config", default="config.json")
    p.add_argument("--n", type=int, default=10,
                   help="Number of receipts to dump (default 10).")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which split to inspect (default test).")
    p.add_argument("--out", default="results/diagnose.json",
                   help="Path to write the JSON report to.")
    p.set_defaults(func=_run_diagnose)
