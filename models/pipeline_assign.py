"""Field assignment via the learned AttentionAssigner's cross-attention.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: performs inference-time field assignment by picking regions whose
    attention exceeds half of max for multi-line fields (address), then
    postprocessing date/total through regex to match SROIE GT format.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from models.attention_assign import AttentionAssigner, text_priors
from models.rule_based import DATE_RE, MONEY_RE

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    import torch

# Fields whose ground-truth value spans multiple OCR regions on a typical
# SROIE receipt. ``address`` is the canonical case (street / city / postcode
# on three text lines). The assigner is *trained* with sum-mass loss, which
# spreads attention across all positive regions; argmax at inference would
# cap address F1 at ~0.3. We instead pick every region whose attention
# exceeds ``_MULTI_LINE_FRACTION * max(attn)`` and concatenate in spatial
# (top→bottom) order.
_MULTI_LINE_FIELDS = frozenset({"address"})
_MULTI_LINE_FRACTION = 0.5

_FIELD_REGEX = {"date": DATE_RE, "total": MONEY_RE}


def postprocess_value(name: str, value: str) -> str:
    """Strip region text to SROIE GT format (date/total regex extraction)."""
    pattern = _FIELD_REGEX.get(name)
    if pattern is None:
        return value
    m = pattern.search(value)
    if not m:
        return value
    out = m.group(0).strip()
    if name == "total":
        out = re.sub(r"^(RM|USD|SGD|MYR|\$)\s*", "", out, flags=re.IGNORECASE)
    return out


def _assign_learned(
    assigner: AttentionAssigner, texts: list[str],
    feats: list[torch.Tensor], bboxes: list[list[float]],
    fields: list[str], device: str,
) -> dict[str, str]:
    """Use AttentionAssigner cross-attention to pick regions per field."""
    values, _attn = _assign_learned_with_attn(
        assigner, texts, feats, bboxes, fields, device,
    )
    return values


def _assign_learned_with_attn(
    assigner: AttentionAssigner, texts: list[str],
    feats: list[torch.Tensor], bboxes: list[list[float]],
    fields: list[str], device: str,
) -> tuple[dict[str, str], torch.Tensor | None]:
    """Field-assign and return (F, N) cross-attention for fig_attn_heatmap.

    Multi-line fields (address) pick all regions with attn ≥ 0.5 × max
    because the pos-mass loss spreads attention across positive regions
    at train time.  Returns (values, None) on empty text.
    """
    if not texts:
        return {}, None
    tf = torch.cat(feats, dim=0).unsqueeze(0)
    bf = torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0).to(device)
    priors = torch.tensor(
        [text_priors(t) for t in texts], dtype=torch.float32,
    ).unsqueeze(0).to(device)
    _logits, attn_w = assigner(tf, bf, priors)
    attn_sample = attn_w[0].detach().cpu()  # (F, N), kept for the sampler
    used: set[int] = set()
    out: dict[str, str] = {}
    for f_idx, name in enumerate(fields):
        if len(used) >= len(texts):
            break
        w = attn_w[0, f_idx].clone()
        for u in used:
            w[u] = -1e9
        if name in _MULTI_LINE_FIELDS:
            max_w = float(w.max().item())
            if max_w <= 0:
                continue
            picks = [
                i for i in range(w.shape[0])
                if i not in used and float(w[i].item()) >= _MULTI_LINE_FRACTION * max_w
            ]
            if not picks:
                continue
            picks.sort(key=lambda i: bboxes[i][1])  # spatial top→bottom
            for i in picks:
                used.add(i)
            value = " ".join(texts[i].strip() for i in picks if texts[i].strip())
        else:
            best = int(w.argmax().item())
            used.add(best)
            value = texts[best]
        out[name] = postprocess_value(name, value)
    return out, attn_sample
