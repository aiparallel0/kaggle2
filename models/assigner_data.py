"""Prepare per-receipt region groups for AttentionAssigner training.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: encodes TrOCR hidden states for every region in a receipt and bundles
    them with enriched 8-d bboxes and 6-d text priors into training groups.
    The 90/10 train/val split is seeded deterministically.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

from core.errors import TrainError
from core.types import AssignerData, Crop
from models.attention_assign import text_priors, text_priors_v2
from models.attention_priors import _MONEY_RE

# Fraction of prepared receipts reserved for validation. 10 % is standard and
# leaves enough training signal for a ~50k-param model on O(500) receipts.
_VAL_FRACTION = 0.1

try:
    import torch
    from torch import Tensor
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    from torch import Tensor

Group = tuple["Tensor", "Tensor", "Tensor", dict[int, list[int]]]


def _encode_regions(
    proc: Any, trocr: Any, regions: list[Crop], device: str, feat_dim: int,
) -> Tensor:
    feats: list[Tensor] = []
    for crop in regions:
        img = Image.open(crop.image_path).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = crop.bbox
        region = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        if region.width < 1 or region.height < 1:
            region = img
        pv = proc(images=region, return_tensors="pt").pixel_values.to(device)
        feat = trocr.encoder(pv).last_hidden_state.mean(dim=1)
        feats.append(feat.cpu())
    return torch.cat(feats, dim=0) if feats else torch.zeros(0, feat_dim)


def _prepare_groups(
    data: AssignerData, field_to_idx: dict[str, int], device: str,
    priors_v2: bool = True,
) -> tuple[list[Group], int]:
    """Encode per-receipt regions via TrOCR encoder → training groups."""
    if not data.regions:
        raise TrainError("AssignerData.regions is empty — cannot train assigner.")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(data.trocr_path)
    trocr = VisionEncoderDecoderModel.from_pretrained(data.trocr_path).to(device)
    trocr.eval()
    feat_dim: int = trocr.config.encoder.hidden_size
    prepared: list[Group] = []
    with torch.no_grad():
        for regions in data.regions:
            if not regions:
                continue
            feats = _encode_regions(proc, trocr, regions, device, feat_dim)
            if feats.shape[0] == 0:
                continue
            bboxes = torch.tensor([list(r.bbox) for r in regions], dtype=torch.float32)
            if priors_v2:
                money_mask = [bool(_MONEY_RE.search(r.text)) for r in regions]
                money_idxs = [i for i, m in enumerate(money_mask) if m]
                last_money = max(money_idxs) if money_idxs else -1
                y_vals = [r.bbox[3] for r in regions]
                max_y = max(y_vals) if y_vals else 1.0
                prior_list = [
                    text_priors_v2(
                        r.text, r.bbox[3] / max(max_y, 1e-6), i == last_money,
                    )
                    for i, r in enumerate(regions)
                ]
            else:
                prior_list = [text_priors(r.text) for r in regions]
            priors = torch.tensor(prior_list, dtype=torch.float32)
            targets: dict[int, list[int]] = {}
            for i, r in enumerate(regions):
                fi = field_to_idx.get(r.field_label)
                if fi is not None:
                    targets.setdefault(fi, []).append(i)
            if targets:
                prepared.append((feats, bboxes, priors, targets))
    if not prepared:
        raise TrainError("No valid labeled receipts for assigner training.")
    return prepared, feat_dim


def split_train_val(prepared: list[Group], seed: int) -> tuple[list[Group], list[Group]]:
    """Deterministic 90/10 split for AttentionAssigner val-loss tracking."""
    n = len(prepared)
    if n <= 1:
        return list(prepared), list(prepared)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_val = max(1, int(round(n * _VAL_FRACTION)))
    val_idxs = set(perm[:n_val])
    train = [prepared[i] for i in range(n) if i not in val_idxs]
    val = [prepared[i] for i in range(n) if i in val_idxs]
    if not train:
        train = list(prepared)
    if not val:
        val = list(prepared)
    return train, val
