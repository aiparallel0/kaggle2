"""Evaluate YOLO + TrOCR + AttentionAssigner pipeline → Metrics."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from core.errors import EvalError
from core.types import Field, Metrics, PipelinePaths, Prediction, Receipt
from models.attention_assign import AttentionAssigner, load_assigner

_FIELDS = ["company", "date", "address", "total"]


def _ned(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = _edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _token_f1(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = ta & tb
    p = len(common) / len(tb)
    r = len(common) / len(ta)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _compute_metrics(predictions: list[Prediction], receipts: list[Receipt]) -> Metrics:
    pf1: dict[str, list[float]] = {f: [] for f in _FIELDS}
    pned: dict[str, list[float]] = {f: [] for f in _FIELDS}
    pem: dict[str, list[float]] = {f: [] for f in _FIELDS}
    for pred, rec in zip(predictions, receipts):
        gt = {fld.name.lower(): fld.value.lower() for fld in rec.fields}
        pr = {fld.name.lower(): fld.value.lower() for fld in pred.fields}
        for f in _FIELDS:
            g, p = gt.get(f, ""), pr.get(f, "")
            pem[f].append(1.0 if g == p else 0.0)
            pned[f].append(_ned(g, p))
            pf1[f].append(_token_f1(g, p))
    per_f1 = {f: sum(v) / len(v) for f, v in pf1.items() if v}
    per_ned = {f: sum(v) / len(v) for f, v in pned.items() if v}
    per_em = {f: sum(v) / len(v) for f, v in pem.items() if v}
    return Metrics(
        global_f1=sum(per_f1.values()) / len(per_f1) if per_f1 else 0.0,
        global_ned=sum(per_ned.values()) / len(per_ned) if per_ned else 0.0,
        global_em=sum(per_em.values()) / len(per_em) if per_em else 0.0,
        per_field_f1=per_f1, per_field_ned=per_ned, per_field_em=per_em,
    )


def eval_pipeline(paths: PipelinePaths, test: list[Receipt]) -> Metrics:
    """Run YOLO→TrOCR→Assigner pipeline on test; return Metrics."""
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
    # Discover yolo_img_size from assigner dir metadata if stored, else 512
    meta_path = Path(paths.assigner).parent / "pipeline_meta.json"
    yolo_img_size = 512
    if meta_path.exists():
        with open(meta_path) as f:
            yolo_img_size = json.load(f).get("yolo_img_size", 512)
    predictions: list[Prediction] = []
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            # Bug 5: always pass imgsz explicitly
            det_results = yolo.predict(str(rec.image_path), imgsz=yolo_img_size, verbose=False)
            boxes = det_results[0].boxes.xyxyn.cpu().tolist() if det_results[0].boxes else []
            region_texts: list[str] = []
            text_feats_list: list[object] = []
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
                enc_out = trocr_model.encoder(pv)
                feat = enc_out.last_hidden_state.mean(dim=1)  # (1, 768)
                region_texts.append(text)
                text_feats_list.append(feat)
                bbox_list.append([x1, y1, x2, y2])
            assigned: dict[str, str] = {}
            if region_texts:
                tf = torch.cat(list(text_feats_list), dim=0).unsqueeze(0)
                bf = torch.tensor(bbox_list, dtype=torch.float32).unsqueeze(0).to(device)
                _, attn_w = assigner(tf, bf)  # attn_w: (1, n_fields, N_regions)
                for f_idx, field_name in enumerate(_FIELDS):
                    weights = attn_w[0, f_idx]  # (N_regions,)
                    best_r = int(weights.argmax().item())
                    assigned[field_name] = region_texts[best_r]
            pred_fields = [Field(name=k, value=v) for k, v in assigned.items()]
            predictions.append(Prediction(receipt_id=rec.image_path.stem, fields=pred_fields))
    metrics = _compute_metrics(predictions, test)
    out_dir = Path(paths.yolo).parent.parent
    with open(out_dir / "pipeline_metrics.json", "w") as f:
        json.dump(
            {"global_f1": metrics.global_f1, "global_ned": metrics.global_ned,
             "global_em": metrics.global_em, "per_field_f1": metrics.per_field_f1},
            f, indent=2,
        )
    return metrics
