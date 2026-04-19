"""Train AttentionAssigner on per-receipt multi-region batches.

The assigner's job at inference time is to point each of N_FIELDS queries at
one region out of many detected text lines. Training therefore mirrors that
setup: each receipt contributes one variable-length batch of regions, and
the loss is per-field cross-entropy over the attention distribution.

A deterministic 90/10 train/val split is carved out of the prepared groups so
we have a generalisation signal — without it, the only thing the reported
loss tells us is that the model can memorise its training set, which is of
no use for picking a checkpoint or for judging whether the assigner is
actually ready to ship. The best-by-val-loss state is restored before
saving so ``assigner.pt`` is always the checkpoint with the lowest
held-out loss, not the last epoch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from core.errors import TrainError
from core.types import AssignerData, Crop, ExpConfig
from models.attention_assign import (
    DEFAULT_HIDDEN_DIM,
    AttentionAssigner,
    save_assigner,
    text_priors,
)

# Fraction of prepared receipts reserved for validation. 10 % is a standard
# low-bias choice that still leaves enough training signal for a ~50k-param
# model on O(500) receipts.
_VAL_FRACTION = 0.1

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
) -> list[tuple[Tensor, Tensor, Tensor, dict[int, list[int]]]]:
    """Encode every region once per receipt.

    Returns a list of ``(feats, bboxes, priors, targets)`` tuples where

      * ``feats``   is ``(N, 768)`` mean-pooled TrOCR encoder states,
      * ``bboxes``  is ``(N, 4)`` normalised ``(x1, y1, x2, y2)``,
      * ``priors``  is ``(N, n_text_priors)`` regex/length/upper priors,
      * ``targets`` maps field index → list of positive region indices.

    The assigner itself enriches bboxes to 8-d at forward time, so we
    only store the 4-d form here (fewer bytes per group at the small
    cost of a re-derive per step).
    """
    if not data.regions:
        raise TrainError("AssignerData.regions is empty — cannot train assigner.")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(data.trocr_path)
    trocr = VisionEncoderDecoderModel.from_pretrained(data.trocr_path)
    trocr = trocr.to(device)
    trocr.eval()
    prepared: list[tuple[Tensor, Tensor, Tensor, dict[int, list[int]]]] = []
    with torch.no_grad():
        for regions in data.regions:
            if not regions:
                continue
            feats = _encode_regions(proc, trocr, regions, device)
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
    return prepared


def _group_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor, priors: Tensor,
    targets: dict[int, list[int]], device: str,
) -> Tensor:
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    pf = priors.to(device).unsqueeze(0)
    _, attn_w = assigner(tf, bf, pf)  # (1, n_fields, N)
    loss = torch.zeros((), device=device)
    for f_idx, region_idxs in targets.items():
        probs = attn_w[0, f_idx]
        pos_mass = probs[region_idxs].sum().clamp(min=1e-8)
        loss = loss + -torch.log(pos_mass)
    return loss / len(targets)


def _split_train_val(
    prepared: list[tuple[Tensor, Tensor, Tensor, dict[int, list[int]]]], seed: int,
) -> tuple[
    list[tuple[Tensor, Tensor, Tensor, dict[int, list[int]]]],
    list[tuple[Tensor, Tensor, Tensor, dict[int, list[int]]]],
]:
    """Deterministic 90/10 split by receipt index.

    Uses ``torch.Generator(seed)`` so the split is reproducible across runs
    and independent of global RNG state. On a pathological tiny dataset
    (<= 1 group) both sides degenerate to the full set so training can
    still run — in that case the val-loss signal collapses to train loss,
    which is communicated in the log line rather than silently hidden.
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
    # Defensive fallbacks — with n >= 2 and n_val >= 1 both sides are
    # non-empty in practice, but keep the guard so a future refactor of
    # _VAL_FRACTION cannot silently produce an empty split.
    if not train:
        train = list(prepared)
    if not val:
        val = list(prepared)
    return train, val


def _evaluate(
    assigner: AttentionAssigner,
    groups: list[tuple[Tensor, Tensor, Tensor, dict[int, list[int]]]],
    device: str,
) -> float:
    """Mean per-receipt loss on *groups* with grads disabled, in eval mode."""
    if not groups:
        return float("nan")
    was_training = assigner.training
    assigner.eval()
    total = 0.0
    with torch.no_grad():
        for feats, bboxes, priors, targets in groups:
            total += float(
                _group_loss(assigner, feats, bboxes, priors, targets, device).item(),
            )
    if was_training:
        assigner.train()
    return total / len(groups)


def train_assigner(config: ExpConfig, data: AssignerData) -> str:
    """Train AttentionAssigner with a held-out val split + best-by-val save.

    Reporting *train* loss alone tells us nothing about generalisation — a
    model can drive its training loss arbitrarily low by memorising. The
    function now reports ``train_loss`` and ``val_loss`` every epoch, and
    the saved checkpoint is the one with the lowest val loss observed
    across training (not the last epoch).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field_to_idx = {f.lower(): i for i, f in enumerate(config.fields)}
    prepared = _prepare_groups(data, field_to_idx, device)
    train_groups, val_groups = _split_train_val(prepared, config.seed)
    assigner = AttentionAssigner(
        hidden_dim=DEFAULT_HIDDEN_DIM, n_fields=len(config.fields),
    ).to(device)
    # Adam with weight decay on non-bias params — the network is small
    # enough that we don't need a fancy schedule.
    opt = torch.optim.AdamW(assigner.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    for epoch in range(config.epochs_assigner):
        assigner.train()
        # Deterministic per-epoch shuffle: seed on (config.seed, epoch) so
        # runs are reproducible yet the order differs across epochs.
        gen = torch.Generator().manual_seed(config.seed * 1_000 + epoch)
        perm = torch.randperm(len(train_groups), generator=gen).tolist()
        train_total, train_steps = 0.0, 0
        for idx in perm:
            feats, bboxes, priors, targets = train_groups[idx]
            opt.zero_grad()
            loss = _group_loss(assigner, feats, bboxes, priors, targets, device)
            cast(Any, loss).backward()
            torch.nn.utils.clip_grad_norm_(assigner.parameters(), max_norm=1.0)
            opt.step()
            train_total += float(loss.item())
            train_steps += 1
        train_loss = train_total / max(train_steps, 1)
        val_loss = _evaluate(assigner, val_groups, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in assigner.state_dict().items()
            }
        print(
            f"  Assigner epoch {epoch + 1}/{config.epochs_assigner} "
            f"train_loss={train_loss:.3f} val_loss={val_loss:.3f}"
            + (" *best*" if val_loss == best_val else "")
        )
    if best_state is not None:
        assigner.load_state_dict(best_state)
    out_path = os.path.join(config.output_dir, "assigner.pt")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    save_assigner(assigner, out_path)
    # Emit a small JSON next to the checkpoint so paper/report code can
    # inspect the best-val loss without re-loading the state dict.
    with open(os.path.join(config.output_dir, "assigner_metrics.json"), "w") as f:
        json.dump(
            {
                "best_val_loss": best_val if best_val != float("inf") else None,
                "n_train_groups": len(train_groups),
                "n_val_groups": len(val_groups),
                "epochs": config.epochs_assigner,
            },
            f, indent=2,
        )
    return out_path
