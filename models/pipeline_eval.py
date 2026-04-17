"""Evaluate YOLO + TrOCR + AttentionAssigner pipeline → PipelineResult."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import ExpConfig, Field, PipelinePaths, PipelineResult, Prediction, Receipt
from models.attention_assign import AttentionAssigner, load_assigner
from models.rule_based import rule_based_assign

try:
    import torch
except ImportError:  # lightweight CI — torch not installed
    torch = None  # type: ignore[assignment]


def _detect_and_read(
    yolo: Any, trocr_proc: Any, trocr_model: Any,
    img: Image.Image, img_path: str, cfg: ExpConfig, yolo_img: int, device: str,
) -> tuple[list[str], list[torch.Tensor], list[list[float]]]:
    results = yolo.predict(
        img_path, imgsz=yolo_img, conf=cfg.yolo_conf, verbose=False,
    )
    boxes = results[0].boxes.xyxyn.cpu().tolist() if results[0].boxes else []
    texts: list[str] = []
    feats: list[torch.Tensor] = []
    bboxes: list[list[float]] = []
    w, h = img.width, img.height
    for box in boxes[: cfg.max_regions_per_image]:
        x1, y1, x2, y2 = box[:4]
        crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        if crop.width < 1 or crop.height < 1:
            continue
        pv = trocr_proc(images=crop, return_tensors="pt").pixel_values.to(device)
        enc = trocr_model.encoder(pv).last_hidden_state
        out = trocr_model.generate(pv, max_new_tokens=cfg.trocr_max_new_tokens)
        texts.append(trocr_proc.batch_decode(out, skip_special_tokens=True)[0])
        feats.append(enc.mean(dim=1))
        bboxes.append([x1, y1, x2, y2])
    return texts, feats, bboxes


def _assign_learned(
    assigner: AttentionAssigner, texts: list[str],
    feats: list[torch.Tensor], bboxes: list[list[float]],
    fields: list[str], device: str,
) -> dict[str, str]:
    if not texts:
        return {}
    tf = torch.cat(feats, dim=0).unsqueeze(0)
    bf = torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0).to(device)
    _, attn_w = assigner(tf, bf)
    used: set[int] = set()
    out: dict[str, str] = {}
    for f_idx, name in enumerate(fields):
        if len(used) >= len(texts):
            break
        w = attn_w[0, f_idx].clone()
        for u in used:
            w[u] = -1e9
        best = int(w.argmax().item())
        used.add(best)
        out[name] = texts[best]
    return out


def eval_pipeline(
    paths: PipelinePaths, test: list[Receipt], config: ExpConfig,
) -> PipelineResult:
    """Run YOLO→TrOCR pipeline with learned + rule-based assignment."""
    for p, name in [(paths.yolo, "YOLO"), (paths.trocr, "TrOCR"), (paths.assigner, "Assigner")]:
        if not Path(p).exists():
            raise EvalError(f"{name} checkpoint not found at {p}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise EvalError("ultralytics not installed") from exc
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    yolo = YOLO(paths.yolo)
    trocr_proc: TrOCRProcessor = TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        paths.trocr,
    )
    assigner = load_assigner(paths.assigner, n_fields=len(config.fields))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    assigner = assigner.to(device)
    trocr_model.eval()
    assigner.eval()
    meta = Path(paths.assigner).parent / "pipeline_meta.json"
    yolo_img = config.yolo_img_size
    if meta.exists():
        with open(meta) as f:
            yolo_img = int(json.load(f).get("yolo_img_size", yolo_img))
    preds_l: list[Prediction] = []
    preds_r: list[Prediction] = []
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            texts, feats, bboxes = _detect_and_read(
                yolo, trocr_proc, trocr_model, img, str(rec.image_path),
                config, yolo_img, device,
            )
            learned = _assign_learned(assigner, texts, feats, bboxes, config.fields, device)
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
        }, f, indent=2)
    return PipelineResult(assigner=m_l, rulebased=m_r)
