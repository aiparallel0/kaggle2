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
    N_TEXT_PRIORS_V3,
    AttentionAssigner,
    text_priors,
    text_priors_v2,
    text_priors_v3,
)
from models.attention_priors import _MONEY_RE as _PRIORS_MONEY_RE
from models.pipeline_consensus import enforce_address_contiguity
from models.rule_based import DATE_RE, MONEY_RE
from models.rule_fields import (
    _SUBTOTAL_KW_RE,
    _TOTAL_KW_RE,
    extract_date,
    extract_total,
)
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
# and concatenate top→bottom; pos-mass loss trains for this.  The
# fraction was ``0.5`` originally, which drops the 3rd/4th line of a
# long address whenever attention is spread even slightly — 38/63
# address misses in the live miss table are pure prefix-of-GT.  ``0.25``
# widens the accept band; ``refine_assignments`` then filters junk
# post-hoc via ``_is_addr_boundary`` so the lower threshold never
# pulls in phone/tax-id/header lines.
_MULTI_LINE_FIELDS = frozenset({"address"})
# Change B — default restored from 0.25 to 0.5.  The 0.25 band pulled
# phone/tax/GST lines into the address whenever the attention head was
# mildly diffuse (the common failure mode of a 2-layer encoder trained
# on 500 SROIE receipts); the new ``enforce_address_contiguity`` gate
# below now excises such lines spatially, so we no longer need the
# wider band to compensate for missed prefix-of-GT lines.  Callers that
# want a different band pass ``address_accept_fraction`` via
# :class:`ExpConfig` (Change G).
_MULTI_LINE_FRACTION = 0.5

_FIELD_REGEX = {"date": DATE_RE, "total": MONEY_RE}


def _build_priors(
    texts: list[str], bboxes: list[list[float]], n_priors: int,
) -> list[list[float]]:
    """Per-region text priors matching the assigner's expected dim (6/9/14).

    v2 mirrors :mod:`models.assigner_data`: ``y_norm = bbox[3] / max_y`` and
    ``is_last_money_line = (i == argmax_i(_MONEY_RE.search(texts[i])))``.
    v3 extends v2 with five distractor-aware bits (SUBTOTAL / CASH /
    CHANGE / TAX / ROUNDING) — strategy E of the assigner plan.
    Unknown ``n_priors`` raise ``ValueError`` (no silent zero-padding).
    """
    if n_priors == N_TEXT_PRIORS:
        return [text_priors(t) for t in texts]
    if n_priors in (N_TEXT_PRIORS_V2, N_TEXT_PRIORS_V3):
        money_idxs = [i for i, t in enumerate(texts) if _PRIORS_MONEY_RE.search(t)]
        last_money = max(money_idxs) if money_idxs else -1
        y_vals = [bb[3] for bb in bboxes]
        denom = max(max(y_vals) if y_vals else 1.0, 1e-6)
        if n_priors == N_TEXT_PRIORS_V2:
            return [
                text_priors_v2(texts[i], bboxes[i][3] / denom, i == last_money)
                for i in range(len(texts))
            ]
        return [
            text_priors_v3(texts[i], bboxes[i][3] / denom, i == last_money)
            for i in range(len(texts))
        ]
    raise ValueError(
        f"Unsupported n_text_priors={n_priors}; "
        f"expected {N_TEXT_PRIORS}, {N_TEXT_PRIORS_V2}, or {N_TEXT_PRIORS_V3}.",
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


def _route_regex_field(
    name: str, texts: list[str], bboxes: list[list[float]], used: set[int],
) -> tuple[int, str] | None:
    """Change A — regex-oracle router for deterministic fields (date/total).

    Invert the legacy "attention picks region, regex filters the string"
    pipeline: ask the rule-based extractor first.  Only return a pick
    when it is *unambiguous* — otherwise fall through so the attention
    argmax handles the genuinely hard case.

    * ``date``  — first DATE_RE match in reading order (already unique
                  on well-formed SROIE receipts).
    * ``total`` — confident iff the rule extractor's pick has a TOTAL
                  keyword in its ±1-line neighbourhood; ambiguous
                  (multiple money lines with *no* TOTAL keyword anywhere)
                  falls through to attention.

    Returns ``None`` on ambiguity or when the pick was already consumed
    by another field.
    """
    if name == "date":
        pick = extract_date(texts)
    elif name == "total":
        pick = extract_total(texts, bboxes)
    else:
        return None
    if pick is None or pick[0] in used:
        return None
    if name == "total" and not _is_confident_total(texts, pick[0]):
        return None
    return pick


def _is_confident_total(texts: list[str], idx: int) -> bool:
    """A ``total`` pick is confident iff its ±1-line window contains a
    TOTAL keyword and the picked line itself is not a SUBTOTAL line.

    Returns ``False`` for out-of-range ``idx`` as a defensive guard.
    Co-occurrence of SUBTOTAL in a *neighbour* line (the canonical
    SROIE layout is ``SUBTOTAL … / TOTAL …`` on consecutive rows) is
    NOT a disqualifier; we only disqualify when the picked line itself
    is labelled SUBTOTAL or when no TOTAL keyword appears at all in
    the neighbourhood (the "multiple money lines, no TOTAL keyword"
    ambiguity case from the Change A spec).
    """
    if idx >= len(texts):
        return False
    if _SUBTOTAL_KW_RE.search(texts[idx]):
        return False
    lo, hi = max(0, idx - 1), min(len(texts), idx + 2)
    window = " ".join(texts[lo:hi])
    return bool(_TOTAL_KW_RE.search(window))


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
    address_accept_fraction: float | None = None,
    regex_router: bool = True,
) -> tuple[dict[str, str], torch.Tensor | None]:
    """Field-assign and return (F, N) cross-attention for fig_attn_heatmap.

    Multi-line fields (address) pick all regions with attn ≥
    ``address_accept_fraction`` × max (Change B, default 0.5 via
    ``_MULTI_LINE_FRACTION``) then run through ``enforce_address_contiguity``
    so diffuse-head picks cannot glue phone/GST/tax lines to the
    address.  Regex-deterministic fields (date/total) are resolved via
    :func:`_route_regex_field` first when ``regex_router`` is True
    (Change A); attention is the fallback.  Returns ``(values, None)``
    on empty text.
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
    addr_frac = (
        address_accept_fraction if address_accept_fraction is not None
        else _MULTI_LINE_FRACTION
    )
    for f_idx, name in enumerate(fields):
        if len(used) >= len(texts):
            break
        # Change A — regex-oracle router for date/total.
        if regex_router and name in _FIELD_REGEX:
            routed = _route_regex_field(name, texts, bboxes, used)
            if routed is not None:
                best_idx, value = routed
                used.add(best_idx)
                out[name] = postprocess_value(name, value)
                continue
        w = attn_w[0, f_idx].clone()
        for u in used:
            w[u] = -1e9
        if name in _MULTI_LINE_FIELDS:
            max_w = float(w.max().item())
            if max_w <= 0:
                continue
            picks = [
                i for i in range(w.shape[0])
                if i not in used and float(w[i].item()) >= addr_frac * max_w
            ]
            if not picks:
                continue
            picks.sort(key=lambda i: bboxes[i][1])  # spatial top→bottom
            # Change B — spatial-contiguity gate excises tax/phone/GST
            # lines that the diffuse head sometimes drags in when its
            # attention mass is > addr_frac of max.
            picks = enforce_address_contiguity(picks, bboxes)
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
