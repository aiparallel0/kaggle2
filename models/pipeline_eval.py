"""Evaluate YOLO + TrOCR + AttentionAssigner pipeline → PipelineResult."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from PIL import Image

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import Field, PipelinePaths, PipelineResult, Prediction, Receipt
from models.attention_assign import AttentionAssigner, load_assigner

_FIELDS = ["company", "date", "address", "total"]

_DATE_RE = re.compile(r"\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}")
_MONEY_RE = re.compile(r"\$?\d+\.\d{2}$")


def _rule_based_assign(
    region_texts: list[str], bbox_list: list[list[float]],
) -> dict[str, str]:
    """Spatial + regex heuristic baseline: top→company, date→date, bottom→total."""
    assigned: dict[str, str] = {}
    used: set[int] = set()
    # Date: first region matching a date pattern
    for i, txt in enumerate(region_texts):
        if _DATE_RE.search(txt):
            assigned["date"] = txt
            used.add(i)
            break
    # Total: bottom-most region matching a money pattern
    money_idxs = [
        (i, bbox_list[i][3]) for i, txt in enumerate(region_texts)
        if i not in used and _MONEY_RE.search(txt.strip())
    ]
    if money_idxs:
        best = max(money_idxs, key=lambda x: x[1])
        assigned["total"] = region_texts[best[0]]
        used.add(best[0])
    # Company: topmost unused region
    unused_by_y = sorted(
        [(i, bbox_list[i][1]) for i in range(len(region_texts)) if i not in used],
        key=lambda x: x[1],
    )
    if unused_by_y:
        assigned["company"] = region_texts[unused_by_y[0][0]]
        used.add(unused_by_y[0][0])
    # Address: concatenate next unused regions (typically below company, above total)
    remaining = [
        region_texts[i] for i, _ in unused_by_y[1:]
        if i not in used
    ]
    if remaining:
        assigned["address"] = " ".join(remaining[:3])
    return assigned


def eval_pipeline(paths: PipelinePaths, test: list[Receipt]) -> PipelineResult:
    """Run YOLO→TrOCR pipeline on test with learned + rule-based assignment."""
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
        paths.trocr
    )
    assigner: AttentionAssigner = load_assigner(paths.assigner)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    assigner = assigner.to(device)
    trocr_model.eval()
    assigner.eval()
    meta_path = Path(paths.assigner).parent / "pipeline_meta.json"
    yolo_img_size = 512
    if meta_path.exists():
        with open(meta_path) as f:
            yolo_img_size = json.load(f).get("yolo_img_size", 512)
    preds_learned: list[Prediction] = []
    preds_rule: list[Prediction] = []
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            det_results = yolo.predict(
                str(rec.image_path), imgsz=yolo_img_size, conf=0.25, verbose=False,
            )
            boxes = det_results[0].boxes.xyxyn.cpu().tolist() if det_results[0].boxes else []
            region_texts: list[str] = []
            text_feats_list: list[torch.Tensor] = []
            bbox_list: list[list[float]] = []
            for box in boxes[:16]:
                x1, y1, x2, y2 = box[:4]
                w, h = img.width, img.height
                crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
                if crop.width < 1 or crop.height < 1:
                    continue
                pv = trocr_proc(images=crop, return_tensors="pt").pixel_values.to(device)
                out = trocr_model.generate(pv, max_new_tokens=64)
                text = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
                feat = trocr_model.encoder(pv).last_hidden_state.mean(dim=1)
                region_texts.append(text)
                text_feats_list.append(feat)
                bbox_list.append([x1, y1, x2, y2])
            # Learned assignment (attention assigner)
            learned: dict[str, str] = {}
            if region_texts:
                tf = torch.cat(text_feats_list, dim=0).unsqueeze(0)
                bf = torch.tensor(bbox_list, dtype=torch.float32).unsqueeze(0).to(device)
                _, attn_w = assigner(tf, bf)
                n_regions = len(region_texts)
                used: set[int] = set()
                for f_idx, field_name in enumerate(_FIELDS):
                    if len(used) >= n_regions:
                        break
                    weights = attn_w[0, f_idx].clone()
                    for u in used:
                        weights[u] = -1e9
                    best_r = int(weights.argmax().item())
                    used.add(best_r)
                    learned[field_name] = region_texts[best_r]
            # Rule-based assignment (spatial + regex baseline)
            rule = _rule_based_assign(region_texts, bbox_list) if region_texts else {}
            rid = rec.image_path.stem
            preds_learned.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in learned.items()],
            ))
            preds_rule.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in rule.items()],
            ))
    m_learned = compute_metrics(preds_learned, test, _FIELDS)
    m_rule = compute_metrics(preds_rule, test, _FIELDS)
    out_dir = Path(paths.yolo).parent.parent
    with open(out_dir / "pipeline_metrics.json", "w") as f:
        json.dump({
            "assigner_f1": m_learned.global_f1, "rulebased_f1": m_rule.global_f1,
            "assigner_ned": m_learned.global_ned, "assigner_em": m_learned.global_em,
            "per_field_f1": m_learned.per_field_f1,
        }, f, indent=2)
    return PipelineResult(assigner=m_learned, rulebased=m_rule)
