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
      * ``assigner_hardneg_weight`` — λ in the margin term (default 0.5)
      * ``assigner_kd_weight``      — λ in the KD term     (default 0.1)
      * ``assigner_synth_subtotal`` — prob. of injecting a fake SUBTOTAL
                                      line per receipt per epoch (I)
      * ``assigner_ocr_noise``      — prob. of OCR-noise prior aug (F-lite)
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core.types import AssignerData, ExpConfig
from models.assigner_data import Group, _build_prior_vectors, _prepare_groups, split_train_val
from models.assigner_teacher import (
    KD_TEMPERATURE,
    hard_negatives,
    teacher_distribution,
)
from models.attention_assign import (
    N_TEXT_PRIORS_V2,
    N_TEXT_PRIORS_V3,
    AttentionAssigner,
    save_assigner,
)

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
) -> Tensor:
    """Augmented loss: pos-mass NLL + hard-neg hinge (B) + KD KL (C).

    Each extra term is zero-weighted by default so existing training
    runs reproduce bit-for-bit without ``hardneg_weight`` / ``kd_weight``
    set.  When enabled, the hinge penalises any negative region whose
    attn mass exceeds (mean-pos-attn − margin), and KD pulls the
    full attn row toward the rule-based teacher softmax.  When
    ``idx_to_field`` is supplied each field's term is additionally
    multiplied by :data:`FIELD_LOSS_WEIGHTS` (Change E).
    """
    tf = feats.to(device).unsqueeze(0)
    bf = bboxes.to(device).unsqueeze(0)
    pf = priors.to(device).unsqueeze(0)
    _, attn_w = assigner(tf, bf, pf)  # (1, n_fields, N)
    loss = torch.zeros((), device=device)
    n_fields = 0
    for f_idx, region_idxs in targets.items():
        probs = attn_w[0, f_idx]
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
            w = FIELD_LOSS_WEIGHTS.get(idx_to_field.get(f_idx, ""), 1.0)
            term = term * w
        loss = loss + term
        n_fields += 1
    return loss / max(n_fields, 1)


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
) -> float:
    assigner.train()
    gen = torch.Generator().manual_seed(seed * 1_000 + epoch)
    py_rng = random.Random(seed * 1_000 + epoch)
    perm = torch.randperm(len(groups), generator=gen).tolist()
    total, steps = 0.0, 0
    idx_to_field = {v: k for k, v in field_to_idx.items()}
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
    ``config.extra`` enables ``assigner_hardneg_weight`` / ``assigner_kd_weight``
    the loss additionally penalises attention on SROIE distractor
    regions (SUBTOTAL/CASH/CHANGE/TAX/etc.) and distils from the
    rule-based teacher — strategies B and C of the assigner plan.
    Synthetic SUBTOTAL distractors (I) and OCR-noise prior augmentation
    (F-lite) are enabled through ``assigner_synth_subtotal`` and
    ``assigner_ocr_noise`` respectively.  Early-stopping on val-loss
    with patience=config.assigner_patience.  Metrics written to
    assigner_metrics.json for fig_assigner_loss_curve.
    """
    if _import_error is not None:
        raise ImportError(
            "torch is required for AssignTrainer training. "
            "Run: pip install -r requirements.txt"
        ) from _import_error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field_to_idx = {f.lower(): i for i, f in enumerate(config.fields)}
    priors_v3 = bool(config.extra.get("priors_v3", False))
    prepared, text_feat_dim = _prepare_groups(
        data, field_to_idx, device,
        priors_v2=config.priors_v2, priors_v3=priors_v3,
    )
    train_groups, val_groups = split_train_val(prepared, config.seed)
    n_priors = (
        N_TEXT_PRIORS_V3 if priors_v3
        else (N_TEXT_PRIORS_V2 if config.priors_v2 else 6)
    )
    # Wire the ``assigner_hidden`` / ``assigner_n_layers_level2`` knobs
    # that were previously declared in ExpConfig but silently ignored.
    assigner = AttentionAssigner(
        hidden_dim=config.assigner_hidden, n_fields=len(config.fields),
        n_layers=config.assigner_n_layers_level2,
        text_feat_dim=text_feat_dim, dropout=config.dropout_assigner,
        n_text_priors=n_priors,
        text_pool_learned=config.text_pool_learned,
    ).to(device)
    hardneg_weight = _loss_knob(config, "assigner_hardneg_weight", 0.0)
    kd_weight = _loss_knob(config, "assigner_kd_weight", 0.0)
    synth_subtotal = _loss_knob(config, "assigner_synth_subtotal", 0.0)
    ocr_noise = _loss_knob(config, "assigner_ocr_noise", 0.0)
    opt = torch.optim.AdamW(
        assigner.parameters(), lr=1e-3, weight_decay=config.weight_decay_assigner,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.epochs_assigner,
    )
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    patience = config.assigner_patience
    min_delta = config.assigner_min_delta
    no_improve = 0
    stopped_at = config.epochs_assigner
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    for epoch in range(config.epochs_assigner):
        train_loss = _train_epoch(
            assigner, opt, train_groups, config.seed, epoch, device,
            field_to_idx, n_priors,
            hardneg_weight, kd_weight, synth_subtotal, ocr_noise,
        )
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
            f"  Assigner epoch {epoch + 1}/{config.epochs_assigner} "
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
                "epochs": config.epochs_assigner,
                "patience": patience,
                "min_delta": min_delta,
                "weight_decay": config.weight_decay_assigner,
                "dropout": config.dropout_assigner,
                "scheduler": "cosine",
                "priors_v2": config.priors_v2,
                "priors_v3": priors_v3,
                "n_priors": n_priors,
                "hardneg_weight": hardneg_weight,
                "kd_weight": kd_weight,
                "synth_subtotal": synth_subtotal,
                "ocr_noise": ocr_noise,
                "n_params": n_params,
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
            },
            f, indent=2,
        )
    return out_path
