"""YOLOv8 detection + TrOCR recognition per receipt region.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: runs YOLOv8n to detect text-line bboxes, then TrOCR-small-printed
    on each crop.  Filters label-only/noise regions and falls back to
    full-image OCR when YOLO detects zero boxes (Bug 5 guard).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from PIL import Image

from core.types import ExpConfig

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    import torch

# Lines that are just field labels with no value content — the assigner
# sometimes picks them when the actual value line lost a box to YOLO drop-out
# or fell below the confidence threshold.
_LABEL_ONLY_RE = re.compile(
    r"^\s*(?:date|total|sub\s*-?\s*total|subtotal|amount|"
    r"grand\s*total|cashier|receipt|invoice|address|"
    r"company|name|description|item|qty|quantity|price)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Non-printable / punctuation-only noise occasionally produced by TrOCR on
# blurry receipt lines.
_NOISE_ONLY_RE = re.compile(r"^[\s\W_]+$")


def _is_usable_region(text: str) -> bool:
    """False for empty, label-only, or punctuation-only TrOCR output."""
    t = text.strip()
    if not t:
        return False
    if _NOISE_ONLY_RE.match(t):
        return False
    return not _LABEL_ONLY_RE.match(t)


def _detect_and_read(
    yolo: Any, trocr_proc: Any, trocr_model: Any,
    img: Image.Image, img_path: str, cfg: ExpConfig, yolo_img: int, device: str,
) -> tuple[list[str], list[torch.Tensor], list[list[float]]]:
    """YOLO → TrOCR on each crop; return usable (texts, feats, bboxes)."""
    results = yolo.predict(
        img_path, imgsz=yolo_img, conf=cfg.yolo_conf, verbose=False,
    )
    # ``results`` is typically a one-element list for a single image, but
    # ultralytics has shipped builds that return an empty list on total
    # detector failure. Guard both cases explicitly.
    first = results[0] if results else None
    # ``first.boxes`` is a ``Boxes`` container that is *always* truthy even
    # when empty; use an explicit length check so a detection-miss scan
    # falls through to the full-image fallback instead of iterating an
    # empty tensor-view and producing junk coordinates.
    has_boxes = (
        first is not None
        and getattr(first, "boxes", None) is not None
        and len(first.boxes) > 0
    )
    boxes = first.boxes.xyxyn.cpu().tolist() if has_boxes else []
    boxes.sort(key=lambda b: b[1])
    texts: list[str] = []
    feats: list[torch.Tensor] = []
    bboxes: list[list[float]] = []
    w, h = img.width, img.height
    for box in boxes[: cfg.max_regions_per_image]:
        x1, y1, x2, y2 = box[:4]
        crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        if crop.width < 1 or crop.height < 1:
            continue
        try:
            pv = trocr_proc(images=crop, return_tensors="pt").pixel_values.to(device)
            enc = trocr_model.encoder(pv).last_hidden_state
            out = trocr_model.generate(
                pv,
                max_new_tokens=cfg.trocr_max_new_tokens,
                decoder_start_token_id=trocr_proc.tokenizer.cls_token_id,
            )
            txt = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
        except (RuntimeError, ValueError):
            # CUDA OOM, assertion-tripped generate, or preprocessor reject
            # on a degenerate crop — drop this region, keep the receipt.
            continue
        if not _is_usable_region(txt):
            continue
        texts.append(txt)
        feats.append(enc.mean(dim=1))
        bboxes.append([x1, y1, x2, y2])
    return texts, feats, bboxes


def _fallback_full_image(
    trocr_proc: Any, trocr_model: Any, img: Image.Image, cfg: ExpConfig, device: str,
) -> tuple[list[str], list[torch.Tensor], list[list[float]]]:
    """TrOCR on full image as single region (empty-detection fallback)."""
    pv = trocr_proc(images=img, return_tensors="pt").pixel_values.to(device)
    enc = trocr_model.encoder(pv).last_hidden_state
    out = trocr_model.generate(
        pv,
        max_new_tokens=cfg.trocr_max_new_tokens,
        decoder_start_token_id=trocr_proc.tokenizer.cls_token_id,
    )
    txt = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
    return [txt], [enc.mean(dim=1)], [[0.0, 0.0, 1.0, 1.0]]
