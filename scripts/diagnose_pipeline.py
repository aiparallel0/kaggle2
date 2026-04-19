"""Dump raw pipeline intermediates for the first N test receipts.

Run this on the vast.ai box *after* a completed eval when pipeline F1 is
unexpectedly low but DONUT F1 / TrOCR eval_f1 look fine.  It loads the
same four checkpoints ``eval_pipeline`` loads, walks the test split, and
writes a JSON report summarising, per receipt:

  * how many YOLO boxes were returned (``yolo_n_boxes``)
  * whether the empty-box fallback fired (``fallback_used``)
  * each region's (bbox, TrOCR transcription)
  * the rule-based field assignment (``rulebased``)
  * the learned assigner's field assignment (``assigner``)
  * the ground-truth field values (``gt``)

The ``summary`` block at the top gives aggregate counts that usually
pinpoint the bottleneck in one glance:

  * ``avg_boxes_per_receipt`` — should be ≥ 10 on SROIE.  If it's ~0 or 1,
    YOLO is broken and every receipt routes through the full-image
    fallback — pipeline F1 is guaranteed to collapse.
  * ``empty_trocr_rate`` — fraction of TrOCR reads that decoded to the
    empty string.  If > 0.3, TrOCR is failing at inference despite high
    eval_f1 on labelled crops.
  * ``all_fields_empty_rate`` — fraction of receipts where the assigner
    produced zero fields.

Usage (same CWD as main.py):

    python scripts/diagnose_pipeline.py --n 5
    python scripts/diagnose_pipeline.py --n 20 --out results/diag.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.types import PipelinePaths  # noqa: E402
from data.sroie import download_sroie, load_or_create_split  # noqa: E402
from models.rule_based import rule_based_assign  # noqa: E402

log = logging.getLogger("diagnose")


def _load_models(paths: PipelinePaths, n_fields: int) -> tuple[Any, Any, Any, Any, str]:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from ultralytics import YOLO

    from models.attention_assign import load_assigner

    yolo = YOLO(paths.yolo)
    trocr_proc = TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(paths.trocr)
    assigner = load_assigner(paths.assigner, n_fields=n_fields)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    assigner = assigner.to(device)
    trocr_model.eval()
    assigner.eval()
    return yolo, trocr_proc, trocr_model, assigner, device


def _detect(yolo: Any, img_path: str, yolo_img: int, conf: float) -> list[list[float]]:
    results = yolo.predict(img_path, imgsz=yolo_img, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    return [b[:4] for b in boxes.xyxyn.cpu().tolist()]


def _read_one(trocr_proc: Any, trocr_model: Any, img: Any, device: str, max_new: int) -> str:
    pv = trocr_proc(images=img, return_tensors="pt").pixel_values.to(device)
    out = trocr_model.generate(pv, max_new_tokens=max_new)
    return trocr_proc.batch_decode(out, skip_special_tokens=True)[0]


def _diagnose(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image

    from models.pipeline_eval import _assign_learned  # reuse exact inference code

    config = load_config(args.config)
    data_path = download_sroie(config)
    split_cache = Path(config.output_dir) / "split.json"
    data = load_or_create_split(data_path, config.seed, split_cache)
    paths = PipelinePaths(
        yolo=str(Path(config.output_dir) / "yolo" / "run" / "weights" / "best.pt"),
        trocr=str(Path(config.output_dir) / "trocr"),
        assigner=str(Path(config.output_dir) / "assigner.pt"),
    )
    yolo, trocr_proc, trocr_model, assigner, device = _load_models(paths, len(config.fields))

    meta_path = Path(config.output_dir) / "pipeline_meta.json"
    yolo_img = config.yolo_img_size
    if meta_path.exists():
        yolo_img = int(json.loads(meta_path.read_text()).get("yolo_img_size", yolo_img))
    log.info("yolo_img=%d conf=%.2f max_regions=%d",
             yolo_img, config.yolo_conf, config.max_regions_per_image)

    n = min(args.n, len(data.test))
    receipts: list[dict[str, Any]] = []
    total_boxes = 0
    empty_reads = 0
    total_reads = 0
    fallback_count = 0
    all_empty_assign = 0

    with torch.no_grad():
        for rec in data.test[:n]:
            img = Image.open(rec.image_path).convert("RGB")
            boxes = _detect(yolo, str(rec.image_path), yolo_img, config.yolo_conf)
            total_boxes += len(boxes)
            used_fallback = False
            regions: list[dict[str, Any]] = []
            texts: list[str] = []
            feats: list[Any] = []
            bboxes: list[list[float]] = []
            w, h = img.width, img.height
            for box in boxes[: config.max_regions_per_image]:
                x1, y1, x2, y2 = box
                crop = img.crop(
                    (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                )
                if crop.width < 1 or crop.height < 1:
                    continue
                pv = trocr_proc(images=crop, return_tensors="pt").pixel_values.to(device)
                enc = trocr_model.encoder(pv).last_hidden_state
                out = trocr_model.generate(pv, max_new_tokens=config.trocr_max_new_tokens)
                text = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
                texts.append(text)
                feats.append(enc.mean(dim=1))
                bboxes.append([x1, y1, x2, y2])
                regions.append({"bbox": [round(c, 4) for c in [x1, y1, x2, y2]],
                                "text": text})
                total_reads += 1
                if not text.strip():
                    empty_reads += 1
            if not texts:
                fallback_count += 1
                used_fallback = True
                full_text = _read_one(trocr_proc, trocr_model, img, device,
                                      config.trocr_max_new_tokens)
                pv = trocr_proc(images=img, return_tensors="pt").pixel_values.to(device)
                enc = trocr_model.encoder(pv).last_hidden_state
                texts = [full_text]
                feats = [enc.mean(dim=1)]
                bboxes = [[0.0, 0.0, 1.0, 1.0]]
                regions.append({"bbox": [0.0, 0.0, 1.0, 1.0], "text": full_text,
                                "_fallback": True})
            learned = _assign_learned(assigner, texts, feats, bboxes,
                                      config.fields, device)
            rule = rule_based_assign(texts, bboxes)
            if not learned:
                all_empty_assign += 1
            gt = {f.name.lower(): f.value for f in rec.fields}
            receipts.append({
                "receipt_id": rec.image_path.stem,
                "yolo_n_boxes": len(boxes),
                "fallback_used": used_fallback,
                "regions": regions,
                "assigner": learned,
                "rulebased": rule,
                "gt": gt,
            })
            log.info("%s: %d boxes, learned=%s",
                     rec.image_path.stem, len(boxes), list(learned.keys()))

    summary = {
        "n_receipts": n,
        "avg_boxes_per_receipt": round(total_boxes / n, 2) if n else 0.0,
        "fallback_rate": round(fallback_count / n, 3) if n else 0.0,
        "empty_trocr_rate": round(empty_reads / total_reads, 3) if total_reads else 0.0,
        "all_fields_empty_rate": round(all_empty_assign / n, 3) if n else 0.0,
        "yolo_img": yolo_img,
        "yolo_conf": config.yolo_conf,
    }
    return {"summary": summary, "receipts": receipts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of test receipts to dump (default 5).")
    parser.add_argument("--out", default="results/diag.json",
                        help="Path to write the JSON report to.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = _diagnose(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Wrote %s", out_path)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
