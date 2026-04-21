"""Evaluate YOLO + TrOCR + AttentionAssigner pipeline → PipelineResult."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import (
    EvalBundle,
    ExpConfig,
    Field,
    PipelinePaths,
    PipelineResult,
    Prediction,
    Receipt,
)
from models.attention_assign import _load_assigner
from models.pipeline_assign import _assign_learned_with_attn
from models.pipeline_attn import DEFAULT_SAMPLE_K, AttentionSampler
from models.pipeline_detect import _detect_and_read, _fallback_full_image
from models.rule_based import rule_based_assign

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass


def _paths_from_config(config: ExpConfig) -> PipelinePaths:
    """Derive pipeline checkpoint paths from ``config.output_dir``."""
    return PipelinePaths(
        yolo=os.path.join(config.output_dir, "yolo", "run", "weights", "best.pt"),
        trocr=os.path.join(config.output_dir, "trocr"),
        assigner=os.path.join(config.output_dir, "assigner.pt"),
    )


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
    assigner = _load_assigner(paths.assigner, n_fields=len(config.fields))
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


def eval_pipeline(config: ExpConfig, test: list[Receipt]) -> PipelineResult:
    """Run YOLO→TrOCR pipeline with learned + rule-based assignment.

    Pipeline checkpoint paths are derived from ``config.output_dir``; this
    keeps the public surface at 2-in/1-out while still reading every path
    from the single source of truth (``config.json``).
    """
    paths = _paths_from_config(config)
    yolo, trocr_proc, trocr_model, assigner, device = _load(paths, config)
    yolo_img = _resolve_yolo_img(paths, config)
    preds_l: list[Prediction] = []
    preds_r: list[Prediction] = []
    n_empty_detect = 0  # receipts where YOLO found zero boxes → full-image fallback
    n_receipt_err = 0  # receipts where per-receipt try/except caught a failure
    attn_sampler = AttentionSampler(k=DEFAULT_SAMPLE_K)
    with torch.no_grad():
        for rec in test:
            rid = rec.image_path.stem
            # Per-receipt try/except so one unreadable scan (corrupt PNG,
            # truncated JPEG, CUDA hiccup) does not abort the entire eval
            # run. We still emit an (empty-fields) Prediction for the
            # receipt so ``compute_metrics`` — which zips predictions and
            # receipts with ``strict=True`` — stays aligned; the receipt
            # then scores F1=0 for every field, which is the correct
            # accounting for "pipeline failed on this input".
            try:
                img = Image.open(rec.image_path).convert("RGB")
                texts, feats, bboxes = _detect_and_read(
                    yolo, trocr_proc, trocr_model, img, str(rec.image_path),
                    config, yolo_img, device,
                )
                # Empty-detection fallback: rare on SROIE but happens on
                # dark scans. Without it the receipt contributes 0 to all
                # four field F1s.
                if not texts:
                    n_empty_detect += 1
                    texts, feats, bboxes = _fallback_full_image(
                        trocr_proc, trocr_model, img, config, device,
                    )
                learned, attn = _assign_learned_with_attn(
                    assigner, texts, feats, bboxes, config.fields, device,
                )
                if attn is not None and not attn_sampler.full:
                    attn_sampler.capture(str(rec.image_path), bboxes, attn)
                rule = rule_based_assign(texts, bboxes) if texts else {}
            except (OSError, RuntimeError, ValueError):
                n_receipt_err += 1
                learned, rule = {}, {}
            preds_l.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in learned.items()],
            ))
            preds_r.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in rule.items()],
            ))
    m_l = compute_metrics(EvalBundle(
        predictions=preds_l, receipts=test, fields=config.fields,
    ))
    m_r = compute_metrics(EvalBundle(
        predictions=preds_r, receipts=test, fields=config.fields,
    ))
    out_dir = Path(paths.yolo).parent.parent
    n_total = max(len(test), 1)
    attn_sampler.write(out_dir)
    with open(out_dir / "pipeline_metrics.json", "w") as f:
        json.dump({
            "assigner_f1": m_l.global_f1, "rulebased_f1": m_r.global_f1,
            "assigner_ned": m_l.global_ned, "assigner_em": m_l.global_em,
            "per_field_f1": m_l.per_field_f1,
            "rulebased_per_field_f1": m_r.per_field_f1,
            "empty_detection_fraction": n_empty_detect / n_total,
            "per_receipt_error_fraction": n_receipt_err / n_total,
            "n_test_receipts": len(test),
        }, f, indent=2)
    return PipelineResult(assigner=m_l, rulebased=m_r)
