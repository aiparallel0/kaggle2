"""Evaluate YOLO + TrOCR + AttentionAssigner pipeline → PipelineResult."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import ExpConfig, Field, PipelinePaths, PipelineResult, Prediction, Receipt
from models.attention_assign import load_assigner
from models.pipeline_assign import assign_learned
from models.pipeline_detect import detect_and_read, fallback_full_image
from models.rule_based import rule_based_assign

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass


def _load(paths: PipelinePaths, config: ExpConfig) -> tuple[Any, Any, Any, Any, str]:
    for p, name in [
        (paths.yolo, "YOLO"), (paths.trocr, "TrOCR"), (paths.assigner, "Assigner"),
    ]:
        if not Path(p).exists():
            raise EvalError(f"{name} checkpoint not found at {p}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise EvalError("ultralytics not installed") from exc
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    yolo = YOLO(paths.yolo)
    trocr_proc = TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(paths.trocr)
    assigner = load_assigner(paths.assigner, n_fields=len(config.fields))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    assigner = assigner.to(device)
    trocr_model.eval()
    assigner.eval()
    return yolo, trocr_proc, trocr_model, assigner, device


def _resolve_yolo_img(paths: PipelinePaths, config: ExpConfig) -> int:
    meta = Path(paths.assigner).parent / "pipeline_meta.json"
    if meta.exists():
        with open(meta) as f:
            return int(json.load(f).get("yolo_img_size", config.yolo_img_size))
    return config.yolo_img_size


def eval_pipeline(
    paths: PipelinePaths, test: list[Receipt], config: ExpConfig,
) -> PipelineResult:
    """Run YOLO→TrOCR pipeline with learned + rule-based assignment."""
    yolo, trocr_proc, trocr_model, assigner, device = _load(paths, config)
    yolo_img = _resolve_yolo_img(paths, config)
    preds_l: list[Prediction] = []
    preds_r: list[Prediction] = []
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            texts, feats, bboxes = detect_and_read(
                yolo, trocr_proc, trocr_model, img, str(rec.image_path),
                config, yolo_img, device,
            )
            # Empty-detection fallback: rare on SROIE but happens on dark scans.
            # Without it, the receipt contributes 0 to all four field F1s.
            if not texts:
                texts, feats, bboxes = fallback_full_image(
                    trocr_proc, trocr_model, img, config, device,
                )
            learned = assign_learned(assigner, texts, feats, bboxes, config.fields, device)
            rule = rule_based_assign(texts, bboxes) if texts else {}
            rid = rec.image_path.stem
            preds_l.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in learned.items()],
            ))
            preds_r.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in rule.items()],
            ))
    m_l = compute_metrics(preds_l, test, config.fields)
    m_r = compute_metrics(preds_r, test, config.fields)
    out_dir = Path(paths.yolo).parent.parent
    with open(out_dir / "pipeline_metrics.json", "w") as f:
        json.dump({
            "assigner_f1": m_l.global_f1, "rulebased_f1": m_r.global_f1,
            "assigner_ned": m_l.global_ned, "assigner_em": m_l.global_em,
            "per_field_f1": m_l.per_field_f1,
            "rulebased_per_field_f1": m_r.per_field_f1,
        }, f, indent=2)
    return PipelineResult(assigner=m_l, rulebased=m_r)
