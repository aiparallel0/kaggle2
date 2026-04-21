"""Train AttentionAssigner on per-receipt multi-region batches (best-by-val save)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core.types import AssignerData, ExpConfig
from models.assigner_data import Group, _prepare_groups, split_train_val
from models.attention_assign import DEFAULT_HIDDEN_DIM, AttentionAssigner, save_assigner

try:
    import torch
    from torch import Tensor
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    from torch import Tensor


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


def _evaluate(assigner: AttentionAssigner, groups: list[Group], device: str) -> float:
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
        opt.zero_grad()
        loss = _group_loss(assigner, feats, bboxes, priors, targets, device)
        cast(Any, loss).backward()
        torch.nn.utils.clip_grad_norm_(assigner.parameters(), max_norm=1.0)
        opt.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def train_assigner(config: ExpConfig, data: AssignerData) -> str:
    """Train :class:`AttentionAssigner` with a held-out validation split.

    The optimisation objective is the multi-instance *positive-mass*
    loss defined in Section III of the paper: for each receipt and each
    of the four field queries, the negative log of the attention mass
    the cross-attention places on the line(s) whose ground-truth label
    is that field.  This formulation --- rather than plain
    cross-entropy --- is what lets the assigner tackle multi-line
    fields such as ``address``.

    Per-epoch ``train_loss`` / ``val_loss`` trajectories are persisted
    to ``assigner_metrics.json`` alongside ``best_epoch`` and
    ``best_val_loss``, so the paper's assigner-loss-curve figure
    (Fig.~\\ref{fig:assigner_loss}) can be emitted from real data.
    The saved checkpoint is the lowest-val-loss one, not the last.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field_to_idx = {f.lower(): i for i, f in enumerate(config.fields)}
    prepared, text_feat_dim = _prepare_groups(data, field_to_idx, device)
    train_groups, val_groups = split_train_val(prepared, config.seed)
    assigner = AttentionAssigner(
        hidden_dim=DEFAULT_HIDDEN_DIM, n_fields=len(config.fields),
        text_feat_dim=text_feat_dim,
    ).to(device)
    opt = torch.optim.AdamW(assigner.parameters(), lr=1e-3, weight_decay=1e-4)
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
                "n_params": n_params,
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
            },
            f, indent=2,
        )
    return out_path
