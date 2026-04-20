"""Prepare per-receipt assigner training groups and deterministic train/val split."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

from core.errors import TrainError
from core.types import AssignerData, Crop
from models.attention_assign import text_priors

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
) -> tuple[list[Group], int]:
    """Encode every region once per receipt → list of training groups.

    Each group is ``(feats (N, D), bboxes (N, 4), priors (N, n_text_priors),
    targets {field_idx: [positive_region_idxs]})`` where D is the TrOCR
    encoder hidden size (384 for trocr-small, 768 for trocr-base). The
    assigner enriches the bbox to 8-d at forward time, so we store the
    4-d form here. Returns ``(groups, feat_dim)`` so the caller can
    construct ``AttentionAssigner(text_feat_dim=feat_dim)``.
    """
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
            priors = torch.tensor(
                [text_priors(r.text) for r in regions], dtype=torch.float32,
            )
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
    """Deterministic 90/10 split by receipt index via ``torch.Generator(seed)``.

    On a pathological tiny dataset (<=1 group) both sides degenerate to the
    full set so training can still run — val-loss collapses to train loss in
    that case, which is communicated in the log rather than hidden.
    """
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
