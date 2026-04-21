"""Per-receipt inspection loop for the diagnose sub-command.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: runs YOLO → TrOCR → AttentionAssigner on individual receipts and
    reports per-field diagnostics for debugging silent F1-destroying bugs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.types import ExpConfig, Field, PipelinePaths, Prediction, Receipt
from models.attention_assign import text_priors
from models.pipeline_assign import _assign_learned
from models.pipeline_detect import _is_usable_region
from models.rule_based import rule_based_assign

log = logging.getLogger("inspect")


def count_sroie_gt_boxes(image_path: Path) -> int:
    """Count SROIE gold-OCR boxes; detect if YOLO collapsed (Bug 5 symptom)."""
    box_path = image_path.parent.parent / "box" / (image_path.stem + ".txt")
    if not box_path.exists():
        return 0
    n = 0
    for line in box_path.read_text(errors="replace").splitlines():
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            [int(p) for p in parts[:8]]
        except ValueError:
            continue
        n += 1
    return n


def load_pipeline_models(
    paths: PipelinePaths, n_fields: int,
) -> tuple[Any, Any, Any, Any, str]:
    """Load YOLOv8, TrOCR, AttentionAssigner for inspection/parity."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from ultralytics import YOLO

    from models.attention_assign import _load_assigner

    yolo = YOLO(paths.yolo)
    trocr_proc = TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(paths.trocr)
    assigner = _load_assigner(paths.assigner, n_fields=n_fields)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    assigner = assigner.to(device)
    trocr_model.eval()
    assigner.eval()
    return yolo, trocr_proc, trocr_model, assigner, device


def inspect_receipt(
    rec: Receipt, yolo: Any, trocr_proc: Any, trocr_model: Any, assigner: Any,
    config: ExpConfig, yolo_img: int, device: str,
) -> tuple[dict[str, Any], Prediction, Prediction, bool, int, int, int]:
    """Run YOLO+TrOCR+AttentionAssigner on one receipt; return dump + counters."""
    from PIL import Image
    img = Image.open(rec.image_path).convert("RGB")
    results = yolo.predict(
        str(rec.image_path), imgsz=yolo_img, conf=config.yolo_conf, verbose=False,
    )
    raw_boxes = (
        [b[:4] for b in results[0].boxes.xyxyn.cpu().tolist()]
        if results[0].boxes else []
    )
    raw_boxes.sort(key=lambda b: b[1])
    regions: list[dict[str, Any]] = []
    texts: list[str] = []
    feats: list[Any] = []
    bboxes: list[list[float]] = []
    w, h = img.width, img.height
    used_fallback = False
    n_usable = 0
    n_reads = 0
    for box in raw_boxes[: config.max_regions_per_image]:
        x1, y1, x2, y2 = box
        crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        if crop.width < 1 or crop.height < 1:
            continue
        pv = trocr_proc(images=crop, return_tensors="pt").pixel_values.to(device)
        enc = trocr_model.encoder(pv).last_hidden_state
        out = trocr_model.generate(pv, max_new_tokens=config.trocr_max_new_tokens)
        text = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
        n_reads += 1
        usable = _is_usable_region(text)
        regions.append({
            "bbox": [round(c, 4) for c in [x1, y1, x2, y2]],
            "text": text, "usable": usable,
        })
        if not usable:
            continue
        n_usable += 1
        texts.append(text)
        feats.append(enc.mean(dim=1))
        bboxes.append([x1, y1, x2, y2])
    if not texts:
        used_fallback = True
        pv = trocr_proc(images=img, return_tensors="pt").pixel_values.to(device)
        enc = trocr_model.encoder(pv).last_hidden_state
        full_out = trocr_model.generate(pv, max_new_tokens=config.trocr_max_new_tokens)
        full_text = trocr_proc.batch_decode(full_out, skip_special_tokens=True)[0]
        texts = [full_text]
        feats = [enc.mean(dim=1)]
        bboxes = [[0.0, 0.0, 1.0, 1.0]]
        regions.append({"bbox": [0.0, 0.0, 1.0, 1.0], "text": full_text,
                        "_fallback": True, "usable": True})
    learned = _assign_learned(assigner, texts, feats, bboxes, config.fields, device)
    rule = rule_based_assign(texts, bboxes)
    for r in regions[:3]:
        if r.get("usable"):
            r["priors"] = [round(p, 3) for p in text_priors(r["text"])]
    rid = rec.image_path.stem
    dump = {
        "receipt_id": rid, "yolo_n_boxes": len(raw_boxes), "n_usable": len(texts),
        "sroie_gt_n_boxes": count_sroie_gt_boxes(rec.image_path),
        "fallback_used": used_fallback,
        "regions": regions, "assigner": learned, "rulebased": rule,
        "gt": {f.name.lower(): f.value for f in rec.fields},
    }
    pred_l = Prediction(
        receipt_id=rid, fields=[Field(name=k, value=v) for k, v in learned.items()],
    )
    pred_r = Prediction(
        receipt_id=rid, fields=[Field(name=k, value=v) for k, v in rule.items()],
    )
    log.info("%s: yolo=%d usable=%d gt_lines=%d learned=%s rule=%s",
             rid, len(raw_boxes), len(texts), dump["sroie_gt_n_boxes"],
             list(learned.keys()), list(rule.keys()))
    return dump, pred_l, pred_r, used_fallback, len(raw_boxes), n_usable, n_reads
