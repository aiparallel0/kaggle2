"""Prepare per-receipt region groups for AttentionAssigner training.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: encodes TrOCR hidden states for every region in a receipt and bundles
    them with enriched 8-d bboxes, text priors (v1/v2/v3), and the raw
    region texts so the trainer can derive hard-negative region sets
    (strategy B) and rule-based teacher distributions (strategy C) on
    every augmented batch.  The 90/10 train/val split is seeded
    deterministically.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from models.focus_inference import (
    N_TEXT_PRIORS,
    N_TEXT_PRIORS_V2,
    N_TEXT_PRIORS_V3,
    N_TEXT_PRIORS_V4,
    arithmetic_witnesses_v4,
    text_priors,
    text_priors_v2,
    text_priors_v3,
    text_priors_v4,
)
from models.focus_priors import _MONEY_RE, _parse_money
from PIL import Image

from core.errors import TrainError
from core.types import AssignerData, Crop

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

# Group now carries the per-region text strings as a fifth element so the
# trainer can rebuild hard-negatives / teacher distributions after each
# region-order shuffle without re-decoding TrOCR embeddings.
Group = tuple["Tensor", "Tensor", "Tensor", dict[int, list[int]], list[str]]


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


def _build_prior_vectors(
    regions: list[Crop], n_priors: int,
) -> list[list[float]]:
    """Dispatch to v1/v2/v3/v4 prior builders — mirrors ``_build_priors``."""
    if n_priors == N_TEXT_PRIORS:
        return [text_priors(r.text) for r in regions]
    money_mask = [bool(_MONEY_RE.search(r.text)) for r in regions]
    money_idxs = [i for i, m in enumerate(money_mask) if m]
    last_money = max(money_idxs) if money_idxs else -1
    y_vals = [r.bbox[3] for r in regions]
    max_y = max(y_vals) if y_vals else 1.0
    denom = max(max_y, 1e-6)
    if n_priors == N_TEXT_PRIORS_V2:
        return [
            text_priors_v2(r.text, r.bbox[3] / denom, i == last_money)
            for i, r in enumerate(regions)
        ]
    if n_priors == N_TEXT_PRIORS_V3:
        return [
            text_priors_v3(r.text, r.bbox[3] / denom, i == last_money)
            for i, r in enumerate(regions)
        ]
    if n_priors == N_TEXT_PRIORS_V4:
        texts = [r.text for r in regions]
        # Receipt-level FOCUS-T arithmetic witness column (O(N²) once).
        witnesses = arithmetic_witnesses_v4(texts)
        # Receipt-level money_value_normalised: parse money on each line,
        # divide by max(money) on the receipt; lines without money → 0.
        monies = [_parse_money(t) for t in texts]
        max_money = max((m for m in monies if m is not None), default=0.0)
        denom_money = max(max_money, 1e-6)
        money_norm = [
            (m / denom_money) if m is not None else 0.0 for m in monies
        ]
        return [
            text_priors_v4(
                r.text, r.bbox[3] / denom, i == last_money,
                money_norm[i], witnesses[i],
            )
            for i, r in enumerate(regions)
        ]
    raise ValueError(f"Unsupported n_text_priors={n_priors}")


def _prepare_groups(
    data: AssignerData, field_to_idx: dict[str, int], device: str,
    priors_v2: bool = True, priors_v3: bool = False, priors_v4: bool = False,
) -> tuple[list[Group], int]:
    """Encode per-receipt regions via TrOCR encoder → training groups.

    ``priors_v4=True`` overrides ``priors_v3`` which overrides
    ``priors_v2`` and produces 20-d FOCUS framework priors.  ``priors_v3``
    alone produces 14-d distractor-aware priors; ``priors_v2`` alone is
    the legacy 9-d builder; all-False is the original 6-d baseline.
    """
    if not data.regions:
        raise TrainError("AssignerData.regions is empty — cannot train assigner.")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(data.trocr_path)
    trocr = VisionEncoderDecoderModel.from_pretrained(data.trocr_path).to(device)
    trocr.eval()
    feat_dim: int = trocr.config.encoder.hidden_size
    if priors_v4:
        n_priors = N_TEXT_PRIORS_V4
    elif priors_v3:
        n_priors = N_TEXT_PRIORS_V3
    elif priors_v2:
        n_priors = N_TEXT_PRIORS_V2
    else:
        n_priors = N_TEXT_PRIORS
    prepared: list[Group] = []
    with torch.no_grad():
        for regions in data.regions:
            if not regions:
                continue
            feats = _encode_regions(proc, trocr, regions, device, feat_dim)
            if feats.shape[0] == 0:
                continue
            bboxes = torch.tensor([list(r.bbox) for r in regions], dtype=torch.float32)
            prior_list = _build_prior_vectors(regions, n_priors)
            priors = torch.tensor(prior_list, dtype=torch.float32)
            texts = [r.text for r in regions]
            targets: dict[int, list[int]] = {}
            for i, r in enumerate(regions):
                fi = field_to_idx.get(r.field_label)
                if fi is not None:
                    targets.setdefault(fi, []).append(i)
            if targets:
                prepared.append((feats, bboxes, priors, targets, texts))
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
