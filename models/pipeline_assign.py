"""Field-assignment head: AttentionAssigner → ``{field: value}`` dict."""
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
    """Strip the picked region's text down to the SROIE GT substring.

    SROIE's ``date`` and ``total`` ground truth contain only the matched
    pattern (e.g. ``"01/01/2024"``, ``"12.30"``), but TrOCR returns the full
    region text — typically ``"DATE: 01/01/2024"`` or ``"TOTAL    12.30"``.
    Without stripping, every correct prediction scores token-F1 ≈ 0.5.
    The regex prior is non-destructive: on no match, return the raw text.

    For ``total`` we additionally strip currency prefixes (``"RM12.30"`` →
    ``"12.30"``) because SROIE ground truth omits them.
    """
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
    """Use the learned assigner's attention to pick one region per field."""
    if not texts:
        return {}
    tf = torch.cat(feats, dim=0).unsqueeze(0)
    bf = torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0).to(device)
    priors = torch.tensor(
        [text_priors(t) for t in texts], dtype=torch.float32,
    ).unsqueeze(0).to(device)
    _logits, attn_w = assigner(tf, bf, priors)
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
    return out
