"""Train the AttentionAssigner with pos-mass NLL loss and best-by-val saving.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: implements the multi-instance negative-log positive-mass loss that
    lets the learned cross-attention assigner handle multi-line fields
    (e.g. address).  Persists train/val loss trajectories for
    fig_assigner_loss_curve.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core.types import AssignerData, ExpConfig
from models.assigner_data import Group, _prepare_groups, split_train_val
from models.attention_assign import (
    DEFAULT_HIDDEN_DIM,
    N_TEXT_PRIORS_V2,
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


def _group_loss(
    assigner: AttentionAssigner, feats: Tensor, bboxes: Tensor, priors: Tensor,
    targets: dict[int, list[int]], device: str,
) -> Tensor:
    """Per-receipt pos-mass NLL loss: −log(Σ attn over positive regions)."""
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


def _evaluate(assigner: AttentionAssigner, groups: list[Group], device: str) -> float:
    """Mean per-receipt pos-mass NLL on validation set (grads disabled)."""
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


def _augment(
    f: Tensor, b: Tensor, p: Tensor, t: dict[int, list[int]], gen: Any,
) -> tuple[Tensor, Tensor, Tensor, dict[int, list[int]]]:
    """Bbox jitter ±2 % and region-order shuffle for train-time augmentation."""
    n = f.shape[0]
    if n > 1:
        pi = torch.randperm(n, generator=gen).tolist()
        inv = {o: i for i, o in enumerate(pi)}
        f, b, p = f[pi], b[pi], p[pi]
        t = {k: [inv[x] for x in v] for k, v in t.items()}
    jitter = (torch.rand(b.shape, generator=gen) * 2 - 1) * 0.02
    return f, (b + jitter).clamp(0.0, 1.0), p, t


def _train_epoch(
    assigner: AttentionAssigner, opt: Any, groups: list[Group], seed: int, epoch: int,
    device: str,
) -> float:
    assigner.train()
    gen = torch.Generator().manual_seed(seed * 1_000 + epoch)
    perm = torch.randperm(len(groups), generator=gen).tolist()
    total, steps = 0.0, 0
    for idx in perm:
        feats, bboxes, priors, targets = groups[idx]
        feats, bboxes, priors, targets = _augment(feats, bboxes, priors, targets, gen)
        opt.zero_grad()
        loss = _group_loss(assigner, feats, bboxes, priors, targets, device)
        cast(Any, loss).backward()
        torch.nn.utils.clip_grad_norm_(assigner.parameters(), max_norm=1.0)
        opt.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def train_assigner(config: ExpConfig, data: AssignerData) -> str:
    """Train AttentionAssigner with pos-mass NLL; return checkpoint path.

    The loss spreads attention across all positive regions at train time,
    enabling multi-line field handling (address).  Early-stopping on
    val-loss with patience=config.assigner_patience.  Metrics written to
    assigner_metrics.json for fig_assigner_loss_curve.
    """
    if _import_error is not None:
        raise ImportError(
            "torch is required for AssignTrainer training. "
            "Run: pip install -r requirements.txt"
        ) from _import_error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field_to_idx = {f.lower(): i for i, f in enumerate(config.fields)}
    prepared, text_feat_dim = _prepare_groups(
        data, field_to_idx, device, priors_v2=config.priors_v2,
    )
    train_groups, val_groups = split_train_val(prepared, config.seed)
    n_priors = N_TEXT_PRIORS_V2 if config.priors_v2 else 6
    assigner = AttentionAssigner(
        hidden_dim=DEFAULT_HIDDEN_DIM, n_fields=len(config.fields),
        text_feat_dim=text_feat_dim, dropout=config.dropout_assigner,
        n_text_priors=n_priors,
    ).to(device)
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
        train_loss = _train_epoch(assigner, opt, train_groups, config.seed, epoch, device)
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
                "n_params": n_params,
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
            },
            f, indent=2,
        )
    return out_path
