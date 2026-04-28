"""Composite per-field assigner loss — pos-mass NLL + CTKR + soft-IoU (Bug 18).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: replaces the one-sided ``-log Σ_{i∈T_f} A_{f,i}`` MIL loss with
    ``L_pos + λ_ctkr·L_ctkr + λ_iou·L_iou``.  L_ctkr is contrastive top-K
    repulsion against the *weakest* gold line (sparse + adaptive
    margin); L_iou is the differentiable analogue of token-F1 at line-
    mask granularity.  Distractor sets from
    :mod:`models.assigner_distractors` (priors_v3 bits) act only as a
    tie-breaker inside top-K, NOT as a stacked penalty.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch
    from torch import Tensor  # noqa: F401
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    from torch import Tensor

from models.assigner_distractors import (
    address_distractor_mask,
    field_distractor_mask,
    total_distractor_mask,
)


def pos_mass_nll(probs: Tensor, pos_idxs: list[int]) -> Tensor:
    """``L_pos = -log Σ_{i∈T_f} A_{f,i}`` with the legacy 1e-8 floor."""
    pos_mass = probs[pos_idxs].sum().clamp(min=1e-8)
    return -torch.log(pos_mass)


def ctkr_loss(
    probs: Tensor, pos_idxs: list[int], distractor_mask: list[bool],
    k: int, margin: float,
) -> Tensor:
    """Contrastive Top-K Repulsion — ``ReLU(A_{f,j} - a_min + margin)`` on top-K negatives.

    ``a_min = min_{i∈T_f} A_{f,i}`` is the *weakest* gold line (so long
    addresses with thin per-line mass still get a usable margin); top-K
    negatives are sorted by descending probability with the distractor
    flag as a stable tie-breaker (priors_v3 bits act as a hint, not a
    hard restriction).  Returns the mean over the K negatives.
    """
    n = int(probs.shape[0])
    pos_set = set(pos_idxs)
    neg_idxs = [j for j in range(n) if j not in pos_set]
    if not pos_idxs or not neg_idxs:
        return probs.new_zeros(())
    a_min = probs[pos_idxs].min()
    flags = [bool(distractor_mask[j]) if j < len(distractor_mask) else False
             for j in neg_idxs]
    probs_neg = probs[neg_idxs].detach().cpu().tolist()
    ranking = sorted(
        range(len(neg_idxs)),
        key=lambda r: (probs_neg[r], 1.0 if flags[r] else 0.0),
        reverse=True,
    )[: max(int(k), 1)]
    topk_idxs = [neg_idxs[r] for r in ranking]
    return (probs[topk_idxs] - a_min + float(margin)).clamp(min=0.0).mean()


def soft_iou_attention(
    probs: Tensor, pos_idxs: list[int], eps: float = 1e-8,
) -> Tensor:
    """``1 − Σ min(Â, m) / Σ max(Â, m)`` with row-max-normalised ``Â``."""
    if not pos_idxs:
        return probs.new_zeros(())
    n = int(probs.shape[0])
    p_norm = probs / (probs.max() + eps)
    mask = probs.new_zeros((n,))
    for i in pos_idxs:
        if 0 <= i < n:
            mask[i] = 1.0
    inter = torch.minimum(p_norm, mask).sum()
    union = torch.maximum(p_norm, mask).sum()
    return 1.0 - inter / (union + eps)


def composite_field_loss(
    probs: Tensor, pos_idxs: list[int], distractor_mask: list[bool],
    ctkr_k: int, ctkr_margin: float,
    ctkr_weight: float, iou_weight: float,
) -> tuple[Tensor, float, float]:
    """Per-field composite loss; returns ``(loss, ctkr_active, iou_value)``.

    ``ctkr_active`` is 1.0 iff ``max_{j∉T_f} A_{f,j} ≥ a_min − margin``
    (i.e. CTKR is still pushing); ``iou_value`` is ``1 − L_iou`` in
    [0, 1].  Both are float scalars driving the
    ``ctkr_active_fraction`` / ``iou_per_field`` traces in
    ``metrics/focus_diagnostics.json``.
    """
    loss = pos_mass_nll(probs, pos_idxs)
    ctkr_active = 0.0
    if ctkr_weight > 0 and pos_idxs:
        loss = loss + ctkr_weight * ctkr_loss(
            probs, pos_idxs, distractor_mask, ctkr_k, ctkr_margin,
        )
        with torch.no_grad():
            pos_set = set(pos_idxs)
            neg_idxs = [j for j in range(int(probs.shape[0])) if j not in pos_set]
            if neg_idxs:
                a_min_v = probs[pos_idxs].min()
                a_max_neg = probs[neg_idxs].max()
                ctkr_active = float(
                    (a_max_neg >= a_min_v - float(ctkr_margin)).item(),
                )
    iou_value = 1.0
    if iou_weight > 0 and pos_idxs:
        l_iou = soft_iou_attention(probs, pos_idxs)
        loss = loss + iou_weight * l_iou
        iou_value = float((1.0 - l_iou).detach().item())
    return loss, ctkr_active, iou_value


__all__ = [
    "address_distractor_mask",
    "composite_field_loss",
    "ctkr_loss",
    "field_distractor_mask",
    "pos_mass_nll",
    "soft_iou_attention",
    "total_distractor_mask",
]
