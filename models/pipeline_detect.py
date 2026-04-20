"""YOLO detection + TrOCR recognition per receipt region (usability filters)."""
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
    """False for empty, label-only, or pure-punctuation TrOCR transcriptions."""
    t = text.strip()
    if not t:
        return False
    if _NOISE_ONLY_RE.match(t):
        return False
    return not _LABEL_ONLY_RE.match(t)


def detect_and_read(
    yolo: Any, trocr_proc: Any, trocr_model: Any,
    img: Image.Image, img_path: str, cfg: ExpConfig, yolo_img: int, device: str,
) -> tuple[list[str], list[torch.Tensor], list[list[float]]]:
    """Run YOLO to box the receipt, TrOCR on each crop; return usable regions.

    Boxes are sorted top→bottom so downstream heuristics (rule_based,
    multi-line address concatenation) can assume reading order. Regions
    failing :func:`_is_usable_region` are silently dropped.
    """
    results = yolo.predict(
        img_path, imgsz=yolo_img, conf=cfg.yolo_conf, verbose=False,
    )
    boxes = results[0].boxes.xyxyn.cpu().tolist() if results[0].boxes else []
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
        pv = trocr_proc(images=crop, return_tensors="pt").pixel_values.to(device)
        enc = trocr_model.encoder(pv).last_hidden_state
        out = trocr_model.generate(pv, max_new_tokens=cfg.trocr_max_new_tokens)
        txt = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
        if not _is_usable_region(txt):
            continue
        texts.append(txt)
        feats.append(enc.mean(dim=1))
        bboxes.append([x1, y1, x2, y2])
    return texts, feats, bboxes


def fallback_full_image(
    trocr_proc: Any, trocr_model: Any, img: Image.Image, cfg: ExpConfig, device: str,
) -> tuple[list[str], list[torch.Tensor], list[list[float]]]:
    """Run TrOCR on the full image as a single region (empty-detection fallback)."""
    pv = trocr_proc(images=img, return_tensors="pt").pixel_values.to(device)
    enc = trocr_model.encoder(pv).last_hidden_state
    out = trocr_model.generate(pv, max_new_tokens=cfg.trocr_max_new_tokens)
    txt = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
    return [txt], [enc.mean(dim=1)], [[0.0, 0.0, 1.0, 1.0]]
