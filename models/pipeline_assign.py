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

from models.attention_assign import (
    N_TEXT_PRIORS,
    N_TEXT_PRIORS_V2,
    AttentionAssigner,
    text_priors,
    text_priors_v2,
)
from models.attention_priors import _MONEY_RE as _PRIORS_MONEY_RE
from models.rule_based import DATE_RE, MONEY_RE
from models.rule_regex import repair_money_ocr

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    import torch

# Fields whose GT value spans multiple OCR regions (address = street/city/
# postcode).  Pick every region with ``attn >= _MULTI_LINE_FRACTION * max``
# and concatenate top→bottom; pos-mass loss trains for this.
_MULTI_LINE_FIELDS = frozenset({"address"})
_MULTI_LINE_FRACTION = 0.5

_FIELD_REGEX = {"date": DATE_RE, "total": MONEY_RE}


def _build_priors(
    texts: list[str], bboxes: list[list[float]], n_priors: int,
) -> list[list[float]]:
    """Per-region text priors matching the assigner's expected dim (6 or 9).

    v2 mirrors :mod:`models.assigner_data`: ``y_norm = bbox[3] / max_y`` and
    ``is_last_money_line = (i == argmax_i(_MONEY_RE.search(texts[i])))``.
    Unknown ``n_priors`` raise ``ValueError`` (no silent zero-padding).
    """
    if n_priors == N_TEXT_PRIORS:
        return [text_priors(t) for t in texts]
    if n_priors == N_TEXT_PRIORS_V2:
        money_idxs = [i for i, t in enumerate(texts) if _PRIORS_MONEY_RE.search(t)]
        last_money = max(money_idxs) if money_idxs else -1
        y_vals = [bb[3] for bb in bboxes]
        denom = max(max(y_vals) if y_vals else 1.0, 1e-6)
        return [
            text_priors_v2(texts[i], bboxes[i][3] / denom, i == last_money)
            for i in range(len(texts))
        ]
    raise ValueError(
        f"Unsupported n_text_priors={n_priors}; "
        f"expected {N_TEXT_PRIORS} or {N_TEXT_PRIORS_V2}.",
    )


def postprocess_value(name: str, value: str) -> str:
    """Strip region text to SROIE GT format; for ``total`` retry after OCR-repair."""
    pattern = _FIELD_REGEX.get(name)
    if pattern is None:
        return value
    m = pattern.search(value) or (
        pattern.search(repair_money_ocr(value)) if name == "total" else None
    )
    if not m:
        return value
    out = m.group(0).strip()
    if name == "total":
        out = re.sub(r"^(RM|USD|SGD|MYR|\$)\s*", "", out, flags=re.IGNORECASE)
    return out


def _has_regex_value(name: str, text: str) -> bool:
    """True iff ``text`` contains a valid regex value for field ``name``."""
    pattern = _FIELD_REGEX.get(name)
    if pattern is None:
        return True
    return bool(pattern.search(text) or (
        name == "total" and pattern.search(repair_money_ocr(text))))


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
    tf = torch.cat(feats, dim=0).unsqueeze(0).to(device)
    bf = torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0).to(device)
    prior_list = _build_priors(texts, bboxes, assigner.n_text_priors)
    priors = torch.tensor(
        prior_list, dtype=torch.float32,
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
            # Regex fields (total/date): argmax with runner-up fallback so
            # a label-only pick (``"TOTAL:"``) doesn't score F1=0.
            if name in _FIELD_REGEX:
                order = [int(i) for i in torch.argsort(w, descending=True).tolist()]
                best = next(
                    (i for i in order
                     if i not in used and _has_regex_value(name, texts[i])),
                    order[0],
                )
            else:
                best = int(w.argmax().item())
            used.add(best)
            value = texts[best]
        out[name] = postprocess_value(name, value)
    return out, attn_sample
