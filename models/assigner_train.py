"""Train AttentionAssigner on per-receipt multi-region batches.

The assigner's job at inference time is to point each of N_FIELDS queries at
one region out of many detected text lines. Training therefore mirrors that
setup: each receipt contributes one variable-length batch of regions, and
the loss is per-field cross-entropy over the attention distribution.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from core.errors import TrainError
from core.types import AssignerData, Crop, ExpConfig
from models.attention_assign import AttentionAssigner, save_assigner

try:
    import torch
    from torch import Tensor
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    from torch import Tensor


def _encode_regions(
    proc: Any, trocr: Any, regions: list[Crop], device: str,
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
    return torch.cat(feats, dim=0) if feats else torch.zeros(0, 768)


def _prepare_groups(
    data: AssignerData, field_to_idx: dict[str, int], device: str,
) -> list[tuple[Tensor, Tensor, dict[int, list[int]]]]:
    """Encode every region once; return (feats, bboxes, field→region-idx) per receipt."""
    if not data.regions:
        raise TrainError("AssignerData.regions is empty — cannot train assigner.")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(data.trocr_path)
    trocr = VisionEncoderDecoderModel.from_pretrained(data.trocr_path)
    trocr = trocr.to(device)
    trocr.eval()
    prepared: list[tuple[Tensor, Tensor, dict[int, list[int]]]] = []
    with torch.no_grad():
        for regions in data.regions:
            if not regions:
                continue
            feats = _encode_regions(proc, trocr, regions, device)
            if feats.shape[0] == 0:
                continue
            bboxes = torch.tensor([list(r.bbox) for r in regions], dtype=torch.float32)
            targets: dict[int, list[int]] = {}
            for i, r in enumerate(regions):
                fi = field_to_idx.get(r.field_label)
                if fi is not None:
                    targets.setdefault(fi, []).append(i)
            if targets:
                prepared.append((feats, bboxes, targets))
    if not prepared:
        raise TrainError("No valid labeled receipts for assigner training.")
    return prepared


def _group_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor,
    targets: dict[int, list[int]], device: str,
) -> Tensor:
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    _, attn_w = assigner(tf, bf)  # (1, n_fields, N)
    loss = torch.zeros((), device=device)
    for f_idx, region_idxs in targets.items():
        probs = attn_w[0, f_idx]
        pos_mass = probs[region_idxs].sum().clamp(min=1e-8)
        loss = loss + -torch.log(pos_mass)
    return loss / len(targets)


def train_assigner(config: ExpConfig, data: AssignerData) -> str:
    """Train AttentionAssigner on per-receipt multi-region batches."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field_to_idx = {f.lower(): i for i, f in enumerate(config.fields)}
    prepared = _prepare_groups(data, field_to_idx, device)
    assigner = AttentionAssigner(hidden_dim=64, n_fields=len(config.fields)).to(device)
    opt = torch.optim.Adam(assigner.parameters(), lr=1e-3)
    assigner.train()
    perm = torch.arange(len(prepared))
    for epoch in range(config.epochs_assigner):
        perm = perm[torch.randperm(len(prepared))]
        total, steps = 0.0, 0
        for idx in perm.tolist():
            feats, bboxes, targets = prepared[idx]
            opt.zero_grad()
            loss = _group_loss(assigner, feats, bboxes, targets, device)
            cast(Any, loss).backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        print(
            f"  Assigner epoch {epoch + 1}/{config.epochs_assigner} "
            f"loss={total / max(steps, 1):.3f}"
        )
    out_path = os.path.join(config.output_dir, "assigner.pt")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    save_assigner(assigner, out_path)
    return out_path
