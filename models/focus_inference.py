"""AttentionAssigner save/load and public API surface.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: re-exports AttentionAssigner and text_priors so callers need not
    import from focus_attention/focus_priors directly.  The private
    _load_assigner handles legacy checkpoint formats.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

from models.focus_attention import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    LEGACY_HIDDEN_DIM,
    LEGACY_N_HEADS,
    LEGACY_N_LAYERS,
    SHIPPED_HIDDEN_DIM,
    SHIPPED_N_HEADS,
    SHIPPED_N_LAYERS,
    AttentionAssigner,
)
from models.focus_priors import (
    N_TEXT_PRIORS,
    N_TEXT_PRIORS_V2,
    N_TEXT_PRIORS_V3,
    N_TEXT_PRIORS_V4,
    arithmetic_witnesses_v4,
    text_priors,
    text_priors_v2,
    text_priors_v3,
    text_priors_v4,
)

__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_N_HEADS",
    "DEFAULT_N_LAYERS",
    "LEGACY_HIDDEN_DIM",
    "LEGACY_N_HEADS",
    "LEGACY_N_LAYERS",
    "N_TEXT_PRIORS",
    "N_TEXT_PRIORS_V2",
    "N_TEXT_PRIORS_V3",
    "N_TEXT_PRIORS_V4",
    "SHIPPED_HIDDEN_DIM",
    "SHIPPED_N_HEADS",
    "SHIPPED_N_LAYERS",
    "AttentionAssigner",
    "_load_assigner",
    "arithmetic_witnesses_v4",
    "load_assigner",
    "migrate_v2_checkpoint",
    "save_assigner",
    "text_priors",
    "text_priors_v2",
    "text_priors_v3",
    "text_priors_v4",
]

_log = logging.getLogger("kaggle2")


def _architecture_config(model: AttentionAssigner) -> dict[str, Any]:
    """Persist architecture params so mismatched checkpoints fail loudly."""
    return {
        "hidden_dim": model.hidden_dim,
        "n_fields": model.n_fields,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "n_text_priors": model.n_text_priors,
        "text_feat_dim": model.text_feat_dim,
        "text_pool_learned": bool(model.text_pool_learned),
        "focus_enabled": bool(model.focus_enabled),
        "focus_max_span": int(model.focus_max_span),
        "focus_total_enabled": bool(model.focus_total_enabled),
        "focus_total_witness_weight": float(model.focus_total_witness_weight),
        "focus_company_enabled": bool(model.focus_company_enabled),
        "focus_company_y_weight": float(model.focus_company_y_weight),
        "focus_company_boilerplate_weight": float(
            model.focus_company_boilerplate_weight,
        ),
        "focus_company_span_enabled": bool(
            getattr(model, "focus_company_span_enabled", False),
        ),
        "focus_company_span_max_span": int(
            getattr(model, "focus_company_span_max_span", 4),
        ),
        "field_names": list(model.field_names) if model.field_names else None,
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
    cfg: dict[str, Any]
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
    fn = cfg.get("field_names")
    field_names: list[str] | None = (
        [str(x) for x in fn] if isinstance(fn, list) else None
    )
    m = AttentionAssigner(
        hidden_dim=int(cfg["hidden_dim"]), n_fields=int(cfg["n_fields"]),
        n_heads=int(cfg.get("n_heads", DEFAULT_N_HEADS)),
        n_layers=int(cfg.get("n_layers", DEFAULT_N_LAYERS)),
        n_text_priors=int(cfg.get("n_text_priors", N_TEXT_PRIORS)),
        text_feat_dim=int(cfg.get("text_feat_dim", 768)),
        text_pool_learned=bool(cfg.get("text_pool_learned", False)),
        focus_enabled=bool(cfg.get("focus_enabled", False)),
        focus_max_span=int(cfg.get("focus_max_span", 8)),
        focus_total_enabled=bool(cfg.get("focus_total_enabled", False)),
        focus_total_witness_weight=float(
            cfg.get("focus_total_witness_weight", 1.0),
        ),
        focus_company_enabled=bool(cfg.get("focus_company_enabled", False)),
        focus_company_y_weight=float(cfg.get("focus_company_y_weight", 1.0)),
        focus_company_boilerplate_weight=float(
            cfg.get("focus_company_boilerplate_weight", 1.0),
        ),
        focus_company_span_enabled=bool(
            cfg.get("focus_company_span_enabled", False),
        ),
        focus_company_span_max_span=int(
            cfg.get("focus_company_span_max_span", 4),
        ),
        field_names=field_names,
    )
    m.load_state_dict(sd)
    if m.n_text_priors not in (
        N_TEXT_PRIORS, N_TEXT_PRIORS_V2, N_TEXT_PRIORS_V3, N_TEXT_PRIORS_V4,
    ):
        raise ValueError(
            f"Loaded assigner has unsupported n_text_priors={m.n_text_priors}; "
            f"expected {N_TEXT_PRIORS}, {N_TEXT_PRIORS_V2}, "
            f"{N_TEXT_PRIORS_V3}, or {N_TEXT_PRIORS_V4}. "
            f"Inference priors builder cannot match this checkpoint.",
        )
    _log.info(
        "Loaded AttentionAssigner from %s (n_text_priors=%d, n_fields=%d, "
        "hidden_dim=%d)",
        path, m.n_text_priors, m.n_fields, m.hidden_dim,
    )
    return m


def load_assigner(path: str) -> AttentionAssigner:
    """Public alias for :func:`_load_assigner` (PR-A / T-A1).

    Used by ``report.combine_ext.merge_assigner_arch`` to introspect the
    saved checkpoint without depending on a private name.
    """
    return _load_assigner(path)


def migrate_v2_checkpoint(path: str) -> AttentionAssigner:
    """PR-A / T-B — pad a legacy V2 (9-d) checkpoint to V3 (14-d) priors.

    Loads the V2 checkpoint, builds a V3-shaped :class:`AttentionAssigner`
    and copies every shared weight; the five extra prior projection
    columns are zero-initialised (so the V3 model behaves identically
    to the V2 model on receipts where none of the new distractor bits
    fire).  Saves nothing — caller decides whether to persist.

    Returns the migrated model.  Raises ``ValueError`` when the input
    is not a V2 checkpoint (n_text_priors must be 9).
    """
    src = _load_assigner(path)
    if src.n_text_priors != N_TEXT_PRIORS_V2:
        raise ValueError(
            f"migrate_v2_checkpoint: expected n_text_priors={N_TEXT_PRIORS_V2}, "
            f"got {src.n_text_priors} (path={path}).",
        )
    dst = AttentionAssigner(
        hidden_dim=src.hidden_dim, n_fields=src.n_fields,
        n_heads=src.n_heads, n_layers=src.n_layers,
        n_text_priors=N_TEXT_PRIORS_V3,
        text_feat_dim=src.text_feat_dim,
        text_pool_learned=bool(src.text_pool_learned),
    )
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for k in dst_sd:
        if k in src_sd and dst_sd[k].shape == src_sd[k].shape:
            dst_sd[k] = src_sd[k]
        elif k == "prior_proj.weight" and "prior_proj.weight" in src_sd:
            sw = src_sd["prior_proj.weight"]
            dw = dst_sd[k].clone()
            dw[:, : sw.shape[1]] = sw
            dst_sd[k] = dw
        elif k == "prior_proj.bias" and "prior_proj.bias" in src_sd:
            dst_sd[k] = src_sd["prior_proj.bias"].clone()
    dst.load_state_dict(dst_sd)
    return dst
