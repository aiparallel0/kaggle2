"""Save/load helpers + public surface for the AttentionAssigner.

Re-exports ``AttentionAssigner`` and ``text_priors`` so existing callers keep
``from models.attention_assign import ...`` without caring about the split.
"""
from __future__ import annotations

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
from models.attention_priors import N_TEXT_PRIORS, text_priors

__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_N_HEADS",
    "DEFAULT_N_LAYERS",
    "N_TEXT_PRIORS",
    "AttentionAssigner",
    "_load_assigner",
    "save_assigner",
    "text_priors",
]


def _architecture_config(model: AttentionAssigner) -> dict[str, int]:
    """Hyperparameters ``load_assigner`` needs to rebuild an equivalent module.

    Saved alongside the state dict so mismatched checkpoints fail loudly at
    load time rather than silently producing junk attention weights.
    """
    return {
        "hidden_dim": model.hidden_dim,
        "n_fields": model.n_fields,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "n_text_priors": model.n_text_priors,
        "text_feat_dim": model.text_feat_dim,
    }


def save_assigner(model: AttentionAssigner, path: str) -> None:
    """Save state_dict + architecture config in a single torch file."""
    torch.save(
        {"state_dict": model.state_dict(), "config": _architecture_config(model)},
        path,
    )


def _load_assigner(
    path: str, n_fields: int | None = None, hidden_dim: int | None = None,
    text_feat_dim: int | None = None,
) -> AttentionAssigner:
    """Load an ``AttentionAssigner`` from ``path``.

    Accepts both the new bundle format (``{state_dict, config}``) and the
    legacy bare-state-dict format. For the legacy format we infer
    ``hidden_dim`` / ``n_fields`` from the ``field_queries`` shape; all other
    architectural knobs fall back to the new defaults.

    This helper is intentionally private (leading underscore): it is
    consumed only by :mod:`models.pipeline_eval` and carries the legacy
    shape-inference overrides that do not fit a 2-in/1-out contract.
    """
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
    return m
