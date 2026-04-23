"""AttentionAssigner save/load and public API surface.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: re-exports AttentionAssigner and text_priors so callers need not
    import from attention_model/attention_priors directly.  The private
    _load_assigner handles legacy checkpoint formats.
"""
from __future__ import annotations

import logging

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

from models.attention_model import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    AttentionAssigner,
)
from models.attention_priors import N_TEXT_PRIORS, N_TEXT_PRIORS_V2, text_priors, text_priors_v2

__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_N_HEADS",
    "DEFAULT_N_LAYERS",
    "N_TEXT_PRIORS",
    "N_TEXT_PRIORS_V2",
    "AttentionAssigner",
    "_load_assigner",
    "save_assigner",
    "text_priors",
    "text_priors_v2",
]

_log = logging.getLogger("kaggle2")


def _architecture_config(model: AttentionAssigner) -> dict[str, int]:
    """Persist architecture params so mismatched checkpoints fail loudly."""
    return {
        "hidden_dim": model.hidden_dim,
        "n_fields": model.n_fields,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "n_text_priors": model.n_text_priors,
        "text_feat_dim": model.text_feat_dim,
    }


def save_assigner(model: AttentionAssigner, path: str) -> None:
    """Save AttentionAssigner state_dict + architecture config."""
    torch.save(
        {"state_dict": model.state_dict(), "config": _architecture_config(model)},
        path,
    )


def _load_assigner(
    path: str, n_fields: int | None = None, hidden_dim: int | None = None,
    text_feat_dim: int | None = None,
) -> AttentionAssigner:
    """Load AttentionAssigner (bundle or legacy format); internal use."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg: dict[str, int]
    if isinstance(blob, dict) and "state_dict" in blob and "config" in blob:
        cfg = dict(blob["config"])
        sd = blob["state_dict"]
    else:
        sd = blob
        fq = sd.get("field_queries")
        if fq is None:
            raise ValueError("Checkpoint missing 'field_queries' — cannot infer dims.")
        cfg = {
            "hidden_dim": int(fq.shape[1]), "n_fields": int(fq.shape[0]),
            "n_heads": DEFAULT_N_HEADS, "n_layers": DEFAULT_N_LAYERS,
            "n_text_priors": N_TEXT_PRIORS, "text_feat_dim": 768,
        }
    if n_fields is not None:
        cfg["n_fields"] = n_fields
    if hidden_dim is not None:
        cfg["hidden_dim"] = hidden_dim
    if text_feat_dim is not None:
        cfg["text_feat_dim"] = text_feat_dim
    m = AttentionAssigner(
        hidden_dim=cfg["hidden_dim"], n_fields=cfg["n_fields"],
        n_heads=cfg.get("n_heads", DEFAULT_N_HEADS),
        n_layers=cfg.get("n_layers", DEFAULT_N_LAYERS),
        n_text_priors=cfg.get("n_text_priors", N_TEXT_PRIORS),
        text_feat_dim=cfg.get("text_feat_dim", 768),
    )
    m.load_state_dict(sd)
    if m.n_text_priors not in (N_TEXT_PRIORS, N_TEXT_PRIORS_V2):
        raise ValueError(
            f"Loaded assigner has unsupported n_text_priors={m.n_text_priors}; "
            f"expected {N_TEXT_PRIORS} or {N_TEXT_PRIORS_V2}. "
            f"Inference priors builder cannot match this checkpoint.",
        )
    _log.info(
        "Loaded AttentionAssigner from %s (n_text_priors=%d, n_fields=%d, "
        "hidden_dim=%d)",
        path, m.n_text_priors, m.n_fields, m.hidden_dim,
    )
    return m
