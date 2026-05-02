"""Train the AttentionAssigner with pos-mass NLL + hard-neg + KD losses.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: implements the multi-instance negative-log positive-mass loss that
    lets the learned cross-attention assigner handle multi-line fields
    (e.g. address), augmented with the plan's strategies B (hard-negative
    margin term) and C (KL-divergence KD from the rule-based teacher).
    Persists train/val loss trajectories for fig_assigner_loss_curve.

    Strategy knobs (attached to :class:`ExpConfig` via ``extra``):
      * ``focus_hardneg_weight`` — λ in the margin term (default 0.5)
      * ``assigner_kd_weight``      — λ in the KD term     (default 0.1)
      * ``focus_synth_subtotal`` — prob. of injecting a fake SUBTOTAL
                                      line per receipt per epoch (I)
      * ``assigner_ocr_noise``      — prob. of OCR-noise prior aug (F-lite)
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from models.assigner_loss import (
    composite_field_loss,
    field_distractor_mask,
)
from models.focus_data import Group, _build_prior_vectors, _prepare_groups, split_train_val
from models.focus_inference import (
    N_TEXT_PRIORS_V2,
    N_TEXT_PRIORS_V3,
    N_TEXT_PRIORS_V4,
    AttentionAssigner,
    save_assigner,
)
from models.focus_priors import (
    V4_IS_COMPANY_BOILERPLATE_IDX,
    V4_WITNESS_IDX,
    V4_Y_NORM_IDX,
)
from models.focus_teacher import (
    KD_TEMPERATURE,
    hard_negatives,
    teacher_distribution,
)

from core.types import AssignerData, ExpConfig
from models.oracle import _best_span

_import_error: ImportError | None = None
try:
    import torch
    from torch import Tensor
except ImportError as _exc:  # lightweight CI — torch not installed
    _import_error = _exc

if TYPE_CHECKING:
    from torch import Tensor

# Strategy B — margin used in the listwise hard-negative hinge.  Small
# because attn probabilities are in [0, 1]; hinge triggers when any
# negative's attn mass is within 0.10 of the positive mean.
_HARDNEG_MARGIN = 0.10

# Strategy I — a handful of SROIE-plausible SUBTOTAL values to inject.
_SYNTH_SUBTOTAL_TEMPLATES = (
    "SUBTOTAL RM {val}", "SUB TOTAL {val}", "SUB-TOTAL: {val}",
)

# Change E — per-field loss weights.  The live miss table shows the
# three fields that regex handles best (company, address, total) are
# exactly the three that regress when the assigner underfits; ``date``
# is already at F1≈0.95 on the rule-based arm so a smaller weight
# still keeps that gradient present without dominating.  Weights are
# applied multiplicatively to each field's per-group loss term.
FIELD_LOSS_WEIGHTS: dict[str, float] = {
    "company": 1.5, "address": 1.3, "total": 1.2, "date": 0.8,
}


def _group_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor, priors: Tensor,
    targets: dict[int, list[int]], device: str,
    negatives: dict[int, list[int]] | None = None,
    teacher: dict[int, list[float]] | None = None,
    hardneg_weight: float = 0.0, kd_weight: float = 0.0,
    idx_to_field: dict[int, str] | None = None,
    texts: list[str] | None = None,
    ctkr_k: int = 0, ctkr_margin: float = 0.0,
    ctkr_weight: float = 0.0, iou_weight: float = 0.0,
    diagnostics: dict[str, list[float]] | None = None,
) -> Tensor:
    """Composite per-field loss: pos-mass NLL + CTKR + soft-IoU + KD (Bug 18).

    The default-zero CTKR + IoU weights preserve bit-exact behaviour
    with pre-Bug-18 runs; the Bug-18 default config flips both to 1.0.
    Legacy ``hardneg_weight`` (PR-A margin hinge) and ``kd_weight``
    (KD KL) terms are retained as opt-in for back-compat with older
    sweeps but are NOT stacked with CTKR by default — see the parent
    PR description for why two ``stop-here'' terms double-penalise.
    When ``diagnostics`` is supplied, ``ctkr_active`` and ``iou`` per-
    field traces are appended in-place for the
    ``metrics/focus_diagnostics.json`` rollup.
    """
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    pf = priors.to(device).unsqueeze(0)
    _, attn_w = assigner(tf, bf, pf)  # (1, n_fields, N)
    loss = torch.zeros((), device=device)
    n_fields = 0
    for f_idx, region_idxs in targets.items():
        probs = attn_w[0, f_idx]
        field_name = idx_to_field.get(f_idx, "") if idx_to_field is not None else ""
        if ctkr_weight > 0 or iou_weight > 0:
            distractor_mask = (
                field_distractor_mask(field_name, texts) if texts is not None
                else [False] * int(probs.shape[0])
            )
            term, ctkr_active, iou_v = composite_field_loss(
                probs, list(region_idxs), distractor_mask,
                ctkr_k=ctkr_k, ctkr_margin=ctkr_margin,
                ctkr_weight=ctkr_weight, iou_weight=iou_weight,
            )
            if diagnostics is not None and field_name:
                diagnostics.setdefault(f"ctkr_active::{field_name}", []).append(
                    ctkr_active,
                )
                diagnostics.setdefault(f"iou::{field_name}", []).append(iou_v)
        else:
            pos_mass = probs[region_idxs].sum().clamp(min=1e-8)
            term = -torch.log(pos_mass)
        if hardneg_weight > 0 and negatives is not None:
            neg_idxs = negatives.get(f_idx, [])
            if neg_idxs:
                neg = probs[neg_idxs]
                pos_mean = probs[region_idxs].mean()
                # Listwise hinge: penalise negatives that come within
                # ``_HARDNEG_MARGIN`` of the positive mean.
                hinge = (neg - pos_mean + _HARDNEG_MARGIN).clamp(min=0.0)
                # log1p of the sum keeps it bounded even with many
                # distractors; matches the ``log(1+Σexp)`` form in the plan.
                term = term + hardneg_weight * torch.log1p(hinge.sum())
        if kd_weight > 0 and teacher is not None:
            t_probs = teacher.get(f_idx)
            if t_probs is not None and len(t_probs) == probs.shape[0]:
                tt = torch.tensor(t_probs, dtype=probs.dtype, device=device)
                if tt.sum() > 0:
                    # KL(teacher ‖ student) with a temperature-softened
                    # student — classic distillation form.
                    log_student = torch.log(
                        torch.softmax(probs / KD_TEMPERATURE, dim=-1).clamp(min=1e-8),
                    )
                    term = term + kd_weight * -(tt * log_student).sum()
        if idx_to_field is not None:
            w = FIELD_LOSS_WEIGHTS.get(field_name, 1.0)
            term = term * w
        loss = loss + term
        n_fields += 1
    return loss / max(n_fields, 1)


def span_iou_boundary_ce(
    start: Tensor, end: Tensor, gold: tuple[int, int], max_span: int,
) -> tuple[Tensor, Tensor]:
    """FOCUS loss — span-IoU + boundary cross-entropy.

    ``start`` / ``end`` are per-token logits ``(N,)`` from the
    ``_AddressSpanHead`` projections.  ``gold`` is the gold ``(gi, gj)``
    pair (inclusive) the trainer aligned via ``oracle._best_span``.
    ``max_span`` mirrors :attr:`ExpConfig.focus_max_span`.

    Computes:

    * ``L_iou = 1 - (p_pair * iou).sum()`` where ``p_pair`` is the
      outer product ``softmax(start) ⊗ softmax(end)``, masked to
      ``j >= i`` AND ``j - i + 1 <= max_span`` and renormalised; and
      ``iou[i, j]`` is the standard 1-D IoU between ``[i, j]`` and the
      gold span ``[gi, gj]``.
    * ``L_bce = CE(start, gi) + CE(end, gj)`` — boundary cross-entropy.
    """
    n = int(start.shape[0])
    gi, gj = int(gold[0]), int(gold[1])
    device = start.device
    p_s = torch.softmax(start, dim=0)
    p_e = torch.softmax(end, dim=0)
    p_pair = p_s.unsqueeze(1) * p_e.unsqueeze(0)  # (N, N)
    i_idx = torch.arange(n, device=device).view(n, 1)
    j_idx = torch.arange(n, device=device).view(1, n)
    mask = (j_idx >= i_idx) & ((j_idx - i_idx + 1) <= max_span)
    p_pair = p_pair * mask.to(p_pair.dtype)
    p_pair = p_pair / p_pair.sum().clamp(min=1e-12)
    inter = (
        torch.minimum(j_idx, j_idx.new_full((1, 1), gj))
        - torch.maximum(i_idx, i_idx.new_full((1, 1), gi)) + 1
    ).clamp(min=0).to(p_pair.dtype)
    union = (
        torch.maximum(j_idx, j_idx.new_full((1, 1), gj))
        - torch.minimum(i_idx, i_idx.new_full((1, 1), gi)) + 1
    ).clamp(min=1).to(p_pair.dtype)
    iou = inter / union
    l_iou = 1.0 - (p_pair * iou).sum()
    gi_t = torch.tensor([gi], device=device, dtype=torch.long)
    gj_t = torch.tensor([gj], device=device, dtype=torch.long)
    l_bce = (
        torch.nn.functional.cross_entropy(start.unsqueeze(0), gi_t)
        + torch.nn.functional.cross_entropy(end.unsqueeze(0), gj_t)
    )
    return l_iou, l_bce


def _focus_gold_span(
    texts: list[str], targets: dict[int, list[int]], address_idx: int,
) -> tuple[int, int] | None:
    """Align gold address text → contiguous ``(gi, gj)`` via token-set F1.

    Returns ``None`` when the receipt has no address-labeled regions or
    when ``_best_span`` reports no overlap (best F1 == 0).  Reuses
    :func:`models.oracle._best_span` so the training-time alignment
    matches the inference-side oracle helper bit-for-bit.
    """
    pos = targets.get(address_idx, [])
    if not pos:
        return None
    gold = " ".join(texts[k] for k in sorted(pos) if 0 <= k < len(texts))
    if not gold.strip():
        return None
    gi, gj, f1 = _best_span(texts, gold)
    if f1 <= 0.0 or gj < gi:
        return None
    return gi, gj


def _focus_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor, priors: Tensor,
    gold: tuple[int, int], device: str, iou_w: float, boundary_w: float,
) -> Tensor:
    """Compute the FOCUS span-cohesion loss on one address-field receipt.

    Encodes ``kv`` via ``AttentionAssigner._encode_kv``, projects to
    ``start`` / ``end`` logits via ``_span_head``, then weights and
    sums :func:`span_iou_boundary_ce`.  Caller is responsible for
    gating on ``cfg.focus_enabled`` and the address-field guard.
    """
    head = assigner._span_head
    if head is None:
        return torch.zeros((), device=device)
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    pf = priors.to(device).unsqueeze(0)
    kv = assigner._encode_kv(tf, bf, pf)[0]  # (N, d)
    start, end = head.start_end(kv)
    l_iou, l_bce = span_iou_boundary_ce(start, end, gold, assigner.focus_max_span)
    return iou_w * l_iou + boundary_w * l_bce


def _focus_company_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor, priors: Tensor,
    gold: tuple[int, int], device: str, iou_w: float, boundary_w: float,
) -> Tensor:
    """Compute the FOCUS-C span-cohesion loss on one company-field receipt.

    Mirrors :func:`_focus_loss` but uses ``_company_span_head`` and
    ``focus_company_span_max_span``.
    """
    head = assigner._company_span_head
    if head is None:
        return torch.zeros((), device=device)
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    pf = priors.to(device).unsqueeze(0)
    kv = assigner._encode_kv(tf, bf, pf)[0]  # (N, d)
    start, end = head.start_end(kv)
    l_iou, l_bce = span_iou_boundary_ce(
        start, end, gold, assigner.focus_company_span_max_span,
    )
    return iou_w * l_iou + boundary_w * l_bce


def _focus_total_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor, priors: Tensor,
    gold_idx: int, device: str, weight: float,
) -> Tensor:
    """Dedicated FOCUS-T auxiliary loss: cross-entropy on the total line index.

    Calls _TotalHead directly on the post-encoder kv so its parameters
    receive a clean gradient decoupled from the shared cross-attention.
    Only fires when assigner._total_head is not None (focus_total_enabled=True)
    and weight > 0.  Total is always a single line, so CE == pos-mass NLL.
    """
    head = assigner._total_head
    if head is None or weight <= 0:
        return torch.zeros((), device=device)
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    pf = priors.to(device).unsqueeze(0)
    kv = assigner._encode_kv(tf, bf, pf)[0]  # (N, d)
    witness_col = pf[0, :, V4_WITNESS_IDX]   # (N,)
    logits = head(kv, witness_col, assigner.focus_total_witness_weight)  # (N,)
    target = torch.tensor(gold_idx, dtype=torch.long, device=device)
    return weight * torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), target.unsqueeze(0),
    )


def _focus_company_pos_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor, priors: Tensor,
    gold_idxs: list[int], device: str, weight: float,
) -> Tensor:
    """Dedicated FOCUS-C positional head auxiliary loss: pos-mass NLL.

    Calls _CompanyHead directly on the post-encoder kv.  Uses pos-mass NLL
    (consistent with the main training objective) rather than cross-entropy
    so multi-line company targets (trade name + SDN BHD suffix) are handled
    gracefully — the head learns to spread mass across all gold company lines.
    Only fires when assigner._company_head is not None and weight > 0.
    """
    head = assigner._company_head
    if head is None or weight <= 0 or not gold_idxs:
        return torch.zeros((), device=device)
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    pf = priors.to(device).unsqueeze(0)
    kv = assigner._encode_kv(tf, bf, pf)[0]  # (N, d)
    y_col = pf[0, :, V4_Y_NORM_IDX]
    boil_col = pf[0, :, V4_IS_COMPANY_BOILERPLATE_IDX]
    logits = head(
        kv, y_col, boil_col,
        assigner.focus_company_y_weight,
        assigner.focus_company_boilerplate_weight,
    )  # (N,)
    probs = torch.softmax(logits, dim=-1)
    pos_mass = probs[gold_idxs].sum().clamp(min=1e-8)
    return weight * (-torch.log(pos_mass))


def _evaluate(assigner: AttentionAssigner, groups: list[Group], device: str) -> float:
    """Mean per-receipt pos-mass NLL on validation set (grads disabled)."""
    if not groups:
        return float("nan")
    was_training = assigner.training
    assigner.eval()
    total = 0.0
    with torch.no_grad():
        for feats, bboxes, priors, targets, _texts in groups:
            total += float(
                _group_loss(assigner, feats, bboxes, priors, targets, device).item(),
            )
    if was_training:
        assigner.train()
    return total / len(groups)


_MONEY_TOKEN_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b")


def _ocr_noise_money(text: str, rng: random.Random) -> str:
    """Strategy F-lite — perturb money tokens to match TrOCR's error
    distribution: split decimal (``12.50`` → ``12 50``), O↔0 swap
    (``12.30`` → ``I2.30`` / ``12.3O``), or trailing-zero drop that
    preserves a well-formed one-decimal-place suffix (``12.50`` →
    ``12.5``).  Returns the original text unchanged when no money
    token is present."""
    if not _MONEY_TOKEN_RE.search(text):
        return text

    def _jitter(m: re.Match[str]) -> str:
        tok = m.group(0)
        choice = rng.randrange(3)
        if choice == 0:
            return tok.replace(".", " ")
        if choice == 1:
            # Swap the first '0' for 'O' or insert 'I' in place of '1'.
            return tok.replace("0", "O", 1) if "0" in tok else tok.replace("1", "I", 1)
        # Trailing-zero drop that keeps at least one fractional digit
        # (``12.50`` → ``12.5`` but never ``12.00`` → ``12``, which
        # would produce an unparseable money token).  Fallback: the
        # unmodified original.
        if "." in tok:
            head, frac = tok.rsplit(".", 1)
            stripped = frac.rstrip("0")
            if stripped:
                return f"{head}.{stripped}"
        return tok
    return _MONEY_TOKEN_RE.sub(_jitter, text)


def _maybe_ocr_noise_priors(
    texts: list[str], bboxes: Tensor, n_priors: int, rng: random.Random,
    prob: float,
) -> Tensor | None:
    """Re-derive priors from OCR-noised texts with probability ``prob``.

    Returns None when no noise applied (caller keeps the original
    ``priors`` tensor unchanged — saves a tensor build per receipt).
    """
    if prob <= 0 or rng.random() >= prob:
        return None
    noised = [_ocr_noise_money(t, rng) for t in texts]
    if noised == texts:
        return None
    # Build a synthetic ``Crop``-shaped sequence for ``_build_prior_vectors``.
    from core.types import Crop
    bbox_list = bboxes.tolist()
    regions = [
        Crop(image_path=Path("."), bbox=tuple(bbox_list[i][:4]), text=noised[i])
        for i in range(len(noised))
    ]
    prior_list = _build_prior_vectors(regions, n_priors)
    return torch.tensor(prior_list, dtype=torch.float32)


def _maybe_inject_subtotal(
    feats: Tensor, bboxes: Tensor, priors: Tensor, targets: dict[int, list[int]],
    texts: list[str], field_to_idx: dict[str, int], n_priors: int,
    rng: random.Random, prob: float,
) -> tuple[Tensor, Tensor, Tensor, dict[int, list[int]], list[str]]:
    """Strategy I — inject a synthetic ``SUBTOTAL`` line just before the
    true TOTAL positive on a fraction of receipts.  New region gets a
    cloned feature vector (forces the model to rely on priors, not
    TrOCR embeddings, to disambiguate), a bbox nudged 2 % up, and NO
    positive label — so the hard-neg loss will treat it as a distractor.
    """
    t_idx = field_to_idx.get("total")
    if t_idx is None or prob <= 0 or rng.random() >= prob:
        return feats, bboxes, priors, targets, texts
    pos = targets.get(t_idx, [])
    if not pos:
        return feats, bboxes, priors, targets, texts
    anchor = pos[0]
    if anchor >= feats.shape[0] or anchor >= len(texts):
        return feats, bboxes, priors, targets, texts
    # Build the synthetic region.  Text value mirrors the anchor money
    # value when parseable, otherwise falls back to a plausible default.
    m = re.search(r"\d{1,3}(?:,\d{3})*\.\d{2}", texts[anchor])
    val = m.group(0) if m else f"{rng.randint(10, 300)}.{rng.randint(0, 99):02d}"
    template = _SYNTH_SUBTOTAL_TEMPLATES[rng.randrange(len(_SYNTH_SUBTOTAL_TEMPLATES))]
    new_text = template.format(val=val)
    # Geometry: place just above the anchor line at 98 % of its y.
    new_bbox = bboxes[anchor].clone()
    dy = (new_bbox[3] - new_bbox[1]).clamp(min=0.01)
    new_bbox[1] = (new_bbox[1] - dy).clamp(min=0.0)
    new_bbox[3] = (new_bbox[3] - dy).clamp(min=0.0)
    # Features: reuse the anchor's so the distractor is visually
    # indistinguishable — the model MUST use priors/bbox to reject it.
    new_feat = feats[anchor].unsqueeze(0).clone()
    # Priors: derive from the synthetic text at the new y.
    from core.types import Crop
    new_regions = [Crop(
        image_path=Path("."), bbox=tuple(new_bbox.tolist()[:4]), text=new_text,
    )]
    new_prior = torch.tensor(
        _build_prior_vectors(new_regions, n_priors), dtype=torch.float32,
    )
    # Splice just before anchor in region order.
    def _splice(t: Tensor, new: Tensor, pos: int) -> Tensor:
        return torch.cat([t[:pos], new, t[pos:]], dim=0)

    feats2 = _splice(feats, new_feat, anchor)
    bboxes2 = _splice(bboxes, new_bbox.unsqueeze(0), anchor)
    priors2 = _splice(priors, new_prior, anchor)
    texts2 = texts[:anchor] + [new_text] + texts[anchor:]
    # Shift every positive index at or after ``anchor`` by +1.
    targets2 = {f: [i + 1 if i >= anchor else i for i in v] for f, v in targets.items()}
    return feats2, bboxes2, priors2, targets2, texts2


def _augment(
    f: Tensor, b: Tensor, p: Tensor, t: dict[int, list[int]], texts: list[str],
    gen: Any,
) -> tuple[Tensor, Tensor, Tensor, dict[int, list[int]], list[str]]:
    """Bbox jitter ±2 % and region-order shuffle for train-time augmentation."""
    n = f.shape[0]
    if n > 1:
        pi = torch.randperm(n, generator=gen).tolist()
        inv = {o: i for i, o in enumerate(pi)}
        f, b, p = f[pi], b[pi], p[pi]
        t = {k: [inv[x] for x in v] for k, v in t.items()}
        texts = [texts[i] for i in pi]
    jitter = (torch.rand(b.shape, generator=gen) * 2 - 1) * 0.02
    return f, (b + jitter).clamp(0.0, 1.0), p, t, texts


def _loss_knob(config: ExpConfig, name: str, default: float) -> float:
    """Fetch a float knob from ``config.extra`` with a type-safe default."""
    raw = config.extra.get(name, default)
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _train_epoch(
    assigner: AttentionAssigner, opt: Any, groups: list[Group], seed: int, epoch: int,
    device: str, field_to_idx: dict[str, int], n_priors: int,
    hardneg_weight: float, kd_weight: float,
    synth_subtotal: float, ocr_noise: float,
    focus_enabled: bool = False, focus_iou_w: float = 0.0,
    focus_boundary_w: float = 0.0,
    ctkr_k: int = 0, ctkr_margin: float = 0.0,
    ctkr_weight: float = 0.0, attn_iou_weight: float = 0.0,
    diagnostics: dict[str, list[float]] | None = None,
    focus_company_span_enabled: bool = False,
    focus_company_span_iou_w: float = 0.0,
    focus_company_span_boundary_w: float = 0.0,
    focus_total_aux_w: float = 0.0,
    focus_company_pos_aux_w: float = 0.0,
) -> float:
    assigner.train()
    gen = torch.Generator().manual_seed(seed * 1_000 + epoch)
    py_rng = random.Random(seed * 1_000 + epoch)
    perm = torch.randperm(len(groups), generator=gen).tolist()
    total, steps = 0.0, 0
    idx_to_field = {v: k for k, v in field_to_idx.items()}
    address_idx = field_to_idx.get("address", -1)
    company_idx = field_to_idx.get("company", -1)
    for idx in perm:
        feats, bboxes, priors, targets, texts = groups[idx]
        feats, bboxes, priors, targets, texts = _augment(
            feats, bboxes, priors, targets, texts, gen,
        )
        feats, bboxes, priors, targets, texts = _maybe_inject_subtotal(
            feats, bboxes, priors, targets, texts, field_to_idx, n_priors,
            py_rng, synth_subtotal,
        )
        noisy_priors = _maybe_ocr_noise_priors(
            texts, bboxes, n_priors, py_rng, ocr_noise,
        )
        priors_eff = noisy_priors if noisy_priors is not None else priors
        negs = hard_negatives(texts, targets, field_to_idx) if hardneg_weight > 0 else None
        teacher = teacher_distribution(texts, field_to_idx) if kd_weight > 0 else None
        opt.zero_grad()
        loss = _group_loss(
            assigner, feats, bboxes, priors_eff, targets, device,
            negatives=negs, teacher=teacher,
            hardneg_weight=hardneg_weight, kd_weight=kd_weight,
            idx_to_field=idx_to_field,
            texts=texts,
            ctkr_k=ctkr_k, ctkr_margin=ctkr_margin,
            ctkr_weight=ctkr_weight, iou_weight=attn_iou_weight,
            diagnostics=diagnostics,
        )
        if focus_enabled and address_idx >= 0:
            gold = _focus_gold_span(texts, targets, address_idx)
            if gold is not None:
                loss = loss + _focus_loss(
                    assigner, feats, bboxes, priors_eff, gold, device,
                    focus_iou_w, focus_boundary_w,
                )
        if focus_company_span_enabled and company_idx >= 0:
            gold_c = _focus_gold_span(texts, targets, company_idx)
            if gold_c is not None:
                loss = loss + _focus_company_loss(
                    assigner, feats, bboxes, priors_eff, gold_c, device,
                    focus_company_span_iou_w, focus_company_span_boundary_w,
                )
        # FOCUS-T aux loss — dedicated CE directly on _TotalHead logits,
        # decoupled from the shared cross-attention so total doesn't compete
        # with the stronger address gradient.  Only fires for single-line
        # total targets (always the case on SROIE).
        if focus_total_aux_w > 0 and getattr(assigner, "focus_total_enabled", False):
            total_idx = field_to_idx.get("total", -1)
            if total_idx >= 0:
                gt_total = targets.get(total_idx, [])
                if len(gt_total) == 1:
                    loss = loss + _focus_total_loss(
                        assigner, feats, bboxes, priors_eff,
                        gt_total[0], device, focus_total_aux_w,
                    )
        # FOCUS-C positional head aux loss — pos-mass NLL directly on
        # _CompanyHead logits, handles multi-line company targets naturally.
        if (
            focus_company_pos_aux_w > 0
            and getattr(assigner, "focus_company_enabled", False)
            and company_idx >= 0
        ):
            gt_company = targets.get(company_idx, [])
            if gt_company:
                loss = loss + _focus_company_pos_loss(
                    assigner, feats, bboxes, priors_eff,
                    list(gt_company), device, focus_company_pos_aux_w,
                )
        cast(Any, loss).backward()
        torch.nn.utils.clip_grad_norm_(assigner.parameters(), max_norm=1.0)
        opt.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def train_assigner(config: ExpConfig, data: AssignerData) -> str:
    """Train AttentionAssigner with pos-mass NLL; return checkpoint path.

    The base loss spreads attention across all positive regions at train
    time, enabling multi-line field handling (address).  When
    ``config.extra`` enables ``focus_hardneg_weight`` / ``assigner_kd_weight``
    the loss additionally penalises attention on SROIE distractor
    regions (SUBTOTAL/CASH/CHANGE/TAX/etc.) and distils from the
    rule-based teacher — strategies B and C of the assigner plan.
    Synthetic SUBTOTAL distractors (I) and OCR-noise prior augmentation
    (F-lite) are enabled through ``focus_synth_subtotal`` and
    ``assigner_ocr_noise`` respectively.  Early-stopping on val-loss
    with patience=config.focus_patience.  Metrics written to
    assigner_metrics.json for fig_assigner_loss_curve.
    """
    if _import_error is not None:
        raise ImportError(
            "torch is required for AssignTrainer training. "
            "Run: pip install -r requirements.txt"
        ) from _import_error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field_to_idx = {f.lower(): i for i, f in enumerate(config.fields)}
    priors_v3 = bool(config.priors_v3) or bool(
        config.extra.get("priors_v3", False),
    )
    priors_v4 = bool(config.priors_v4)
    prepared, text_feat_dim = _prepare_groups(
        data, field_to_idx, device,
        priors_v2=config.priors_v2, priors_v3=priors_v3, priors_v4=priors_v4,
    )
    train_groups, val_groups = split_train_val(prepared, config.seed)
    n_priors = (
        N_TEXT_PRIORS_V4 if priors_v4
        else N_TEXT_PRIORS_V3 if priors_v3
        else (N_TEXT_PRIORS_V2 if config.priors_v2 else 6)
    )
    # Wire the ``assigner_hidden`` / ``assigner_n_layers_level2`` knobs
    # that were previously declared in ExpConfig but silently ignored.
    assigner = AttentionAssigner(
        hidden_dim=config.focus_hidden_dim, n_fields=len(config.fields),
        n_layers=config.focus_n_layers_level2,
        text_feat_dim=text_feat_dim, dropout=config.dropout_focus,
        n_text_priors=n_priors,
        text_pool_learned=config.text_pool_learned,
        focus_enabled=config.focus_enabled,
        focus_max_span=config.focus_max_span,
        focus_total_enabled=config.focus_total_enabled,
        focus_total_witness_weight=config.focus_total_witness_weight,
        focus_company_enabled=config.focus_company_enabled,
        focus_company_y_weight=config.focus_company_y_weight,
        focus_company_boilerplate_weight=config.focus_company_boilerplate_weight,
        field_names=[f.lower() for f in config.fields],
        focus_company_span_enabled=config.focus_company_span_enabled,
        focus_company_span_max_span=config.focus_company_span_max_span,
    ).to(device)
    hardneg_weight = _loss_knob(config, "focus_hardneg_weight", 0.0)
    # P6 — prefer typed ExpConfig KD knobs over legacy ``config.extra``.
    # ``kd_logits_weight`` feeds the KL term (was ``assigner_kd_weight``);
    # ``kd_attn_weight`` adds attention-row KD against the teacher softmax.
    # Both default to 0.0 so legacy runs stay bit-compatible.
    kd_weight = config.kd_logits_weight or _loss_knob(config, "assigner_kd_weight", 0.0)
    synth_subtotal = _loss_knob(config, "focus_synth_subtotal", 0.0)
    ocr_noise = _loss_knob(config, "assigner_ocr_noise", 0.0)
    # Bug 18 — composite-loss knobs (CTKR + soft-IoU on attention).
    # Default-zero means a bit-exact reproduction of pre-Bug-18 runs;
    # the Bug-18 default config flips both to 1.0.  ``attn_iou_weight``
    # reuses ``focus_iou_weight`` per the new-requirement's λ_iou
    # specification (single knob shared with the FOCUS span-IoU term).
    ctkr_k = int(_loss_knob(config, "focus_ctkr_k", 0.0))
    ctkr_margin = _loss_knob(config, "focus_ctkr_margin", 0.0)
    ctkr_weight = _loss_knob(config, "focus_ctkr_weight", 0.0)
    attn_iou_weight = float(config.focus_iou_weight) if config.focus_enabled else 0.0
    # FOCUS — write the gold ``(gi, gj)`` per-receipt label sidecar to
    # ``runs/`` (config.output_dir) so the trainer's gold-span alignment
    # is auditable from the run archive.  Persisted once at startup over
    # the un-shuffled training groups; the in-loop alignment recomputes
    # against the shuffled regions on every step.
    if config.focus_enabled:
        addr_idx = field_to_idx.get("address", -1)
        if addr_idx >= 0:
            labels: list[dict[str, int | None]] = []
            for gi_, (_f, _b, _p, t_, txts_) in enumerate(train_groups):
                span = _focus_gold_span(txts_, t_, addr_idx)
                labels.append({
                    "group_id": gi_,
                    "gi": span[0] if span is not None else None,
                    "gj": span[1] if span is not None else None,
                    "n_regions": len(txts_),
                })
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(config.output_dir, "focus_labels.json"), "w") as fh:
                json.dump({"labels": labels}, fh, indent=2)
    opt = torch.optim.AdamW(
        assigner.parameters(), lr=config.lr_focus,
        weight_decay=config.weight_decay_focus,
    )
    # Fix 4 — optional linear warmup followed by cosine decay.  When
    # ``warmup_ratio_assigner == 0`` (legacy default) we keep the
    # previous bare cosine schedule for bit-compat with older configs.
    warmup_steps = int(config.epochs_focus * config.warmup_ratio_focus)
    sched: torch.optim.lr_scheduler.LRScheduler
    if warmup_steps > 0:
        def _lr_lambda(step: int) -> float:
            """Linear warmup to 1.0, then cosine decay to 0.0."""
            if step < warmup_steps:
                return float(step + 1) / float(max(warmup_steps, 1))
            progress = float(step - warmup_steps) / float(
                max(config.epochs_focus - warmup_steps, 1),
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config.epochs_focus,
        )
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    patience = config.focus_patience
    min_delta = config.focus_min_delta
    no_improve = 0
    stopped_at = config.epochs_focus
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    diagnostics_per_epoch: list[dict[str, float]] = []
    for epoch in range(config.epochs_focus):
        epoch_diag: dict[str, list[float]] = {}
        train_loss = _train_epoch(
            assigner, opt, train_groups, config.seed, epoch, device,
            field_to_idx, n_priors,
            hardneg_weight, kd_weight, synth_subtotal, ocr_noise,
            focus_enabled=config.focus_enabled,
            focus_iou_w=config.focus_iou_weight,
            focus_boundary_w=config.focus_boundary_weight,
            ctkr_k=ctkr_k, ctkr_margin=ctkr_margin,
            ctkr_weight=ctkr_weight, attn_iou_weight=attn_iou_weight,
            diagnostics=epoch_diag,
            focus_company_span_enabled=config.focus_company_span_enabled,
            focus_company_span_iou_w=config.focus_company_span_iou_w,
            focus_company_span_boundary_w=config.focus_company_span_boundary_w,
            focus_total_aux_w=config.focus_total_aux_w,
            focus_company_pos_aux_w=config.focus_company_pos_aux_w,
        )
        # Bug 18 — collapse per-receipt-per-field traces to per-epoch
        # scalars for ``metrics/focus_diagnostics.json``.  The
        # ``ctkr_active_fraction`` for each field is the mean over
        # receipts of the 0/1 indicator, and ``iou_per_field`` is the
        # mean soft-IoU at the field's gold mask granularity.
        diag_scalars = _summarise_diagnostics(epoch_diag)
        diagnostics_per_epoch.append(diag_scalars)
        val_loss = _evaluate(assigner, val_groups, device)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        improved = val_loss < best_val - min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in assigner.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"  Assigner epoch {epoch + 1}/{config.epochs_focus} "
            f"train_loss={train_loss:.3f} val_loss={val_loss:.3f}"
            + (" *best*" if improved else "")
        )
        if no_improve >= patience:
            stopped_at = epoch + 1
            print(
                f"  Assigner early-stopped at epoch {stopped_at}, "
                f"best val_loss={best_val:.3f} @ epoch {best_epoch + 1}"
            )
            break
        sched.step()
    if best_state is not None:
        assigner.load_state_dict(best_state)
        # Round-trip guard: re-evaluate after loading best_state and
        # assert the reproduced val_loss matches the tracked best_val
        # (follow-up Fix B3 — noisy val curves made early-stop fire on
        # epochs where the *last* checkpoint was worse than ``best``;
        # this guarantees the returned ``assigner.pt`` is the best-val
        # checkpoint, not whatever happened to live in memory last).
        reloaded_val = _evaluate(assigner, val_groups, device)
        assert abs(reloaded_val - best_val) < 1e-6, (
            f"Assigner best-checkpoint round-trip failed: "
            f"reloaded val_loss={reloaded_val:.6f} != best_val={best_val:.6f}"
        )
    out_path = os.path.join(config.output_dir, "assigner.pt")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    save_assigner(assigner, out_path)
    n_params = int(sum(p.numel() for p in assigner.parameters()))
    with open(os.path.join(config.output_dir, "assigner_metrics.json"), "w") as f:
        json.dump(
            {
                "best_val_loss": best_val if best_val != float("inf") else None,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
                "stopped_at_epoch": stopped_at,
                "n_train_groups": len(train_groups),
                "n_val_groups": len(val_groups),
                "epochs": config.epochs_focus,
                "patience": patience,
                "min_delta": min_delta,
                "weight_decay": config.weight_decay_focus,
                "dropout": config.dropout_focus,
                "scheduler": "cosine",
                "priors_v2": config.priors_v2,
                "priors_v3": priors_v3,
                "priors_v4": priors_v4,
                "n_priors": n_priors,
                "hardneg_weight": hardneg_weight,
                "kd_weight": kd_weight,
                "synth_subtotal": synth_subtotal,
                "ocr_noise": ocr_noise,
                "lr_assigner": config.lr_focus,
                "warmup_ratio_assigner": config.warmup_ratio_focus,
                "n_params": n_params,
                "focus_enabled": bool(config.focus_enabled),
                "focus_total_enabled": bool(config.focus_total_enabled),
                "focus_company_enabled": bool(config.focus_company_enabled),
                # Bug-18 merge-gate inputs: assigner_metrics.json must
                # carry focus_enabled=true, priors_v4=true, n_priors>=20.
                "ctkr_k": ctkr_k,
                "ctkr_margin": ctkr_margin,
                "ctkr_weight": ctkr_weight,
                "attn_iou_weight": attn_iou_weight,
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
            },
            f, indent=2,
        )
    # Bug 18 — separate diagnostics file mirrors the per-epoch CTKR /
    # IoU trace so the merge gate can verify the FOCUS-A boundary-CE
    # and IoU losses converged (boundary < 0.3, IoU > 0.6 on val) and
    # the CTKR active-fraction collapsed below 0.05 by training end.
    metrics_dir = os.path.join(config.output_dir, "metrics")
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(metrics_dir, "focus_diagnostics.json"), "w") as fdiag:
        json.dump(
            {
                "schema_version": 1,
                "n_epochs": len(diagnostics_per_epoch),
                "fields": [f.lower() for f in config.fields],
                "ctkr_k": ctkr_k,
                "ctkr_margin": ctkr_margin,
                "ctkr_weight": ctkr_weight,
                "attn_iou_weight": attn_iou_weight,
                "per_epoch": diagnostics_per_epoch,
            },
            fdiag, indent=2,
        )
    return out_path


def _summarise_diagnostics(
    epoch_diag: dict[str, list[float]],
) -> dict[str, float]:
    """Collapse per-receipt CTKR / IoU traces to per-field scalar means."""
    out: dict[str, float] = {}
    for key, values in epoch_diag.items():
        if not values:
            continue
        out[key.replace("ctkr_active::", "ctkr_active_fraction::")
            .replace("iou::", "iou_per_field::")] = (
            sum(values) / len(values)
        )
    return out
