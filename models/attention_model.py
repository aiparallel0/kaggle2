"""Learned cross-attention assigner (~380K params) for field assignment.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: the AttentionAssigner takes TrOCR encoder features (768-d), enriched
    8-d bboxes, and 6-d handcrafted text priors, then applies a 2-layer
    Transformer encoder followed by 4-query cross-attention to produce
    per-receipt field assignments.  Trained with pos-mass NLL loss.

Architecture (trains in <10 min on RTX 4090):
  region inputs  ──[text_proj]──┐
                  [bbox_proj]   ├─ + → LayerNorm → TransformerEncoder ──┐
                  [prior_proj]──┘                                       │
                                                                        ▼
  field queries ──[cross-attn]─────────────────────────────────────► (B, F, H)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from models.attention_priors import N_TEXT_PRIORS

try:
    import torch
    import torch.nn as nn

    _NN_BASE: type = nn.Module
except ImportError:  # lightweight CI — torch not installed
    _NN_BASE = object

if TYPE_CHECKING:
    from torch import Tensor

# Architecture defaults.  Exposed as constants so save/load stay consistent.
# 384-d / 6-layer yields ~7–8M parameters — the "capacity compensates for
# TrOCR/YOLO mistakes" hypothesis.  The live miss table (total F1≈0.62,
# rule-based F1≈0.73) falsified that hypothesis: on O(500) SROIE receipts
# the big backbone overfits label noise and is beaten by the regex arm.
# The assigner plan therefore recommends a shrunk fresh-train config
# (``assigner_hidden=192``, ``assigner_n_layers_level2=3`` → ~1.4M params)
# when strategies B/C/E are enabled, to be re-grown only if they plateau.
# Defaults below are kept at the legacy 384/6 so existing checkpoints
# load bit-exact; fresh trains should override via ``ExpConfig``.
DEFAULT_HIDDEN_DIM = 384
DEFAULT_N_HEADS = 12
DEFAULT_N_LAYERS = 6
DEFAULT_DROPOUT = 0.1
DEFAULT_FF_MULT = 2  # FFN hidden = hidden_dim * DEFAULT_FF_MULT

# Recommended shrunk config for fresh trains with strategies B + C + E
# enabled (see :mod:`models.assigner_train`).  Not used directly by this
# module — callers (``ExpConfig``) opt in by setting the corresponding
# ``assigner_hidden`` / ``assigner_n_layers_level2`` fields.
MINI_HIDDEN_DIM = 192
MINI_N_LAYERS = 3


def _pick_n_heads(hidden_dim: int, requested: int) -> int:
    """Largest divisor of hidden_dim ≤ requested (MultiheadAttention requires it)."""
    if hidden_dim <= 0:
        return 1
    for h in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % h == 0:
            return h
    return 1


class AttentionAssigner(_NN_BASE):  # type: ignore[misc]
    """Transformer + 4-query cross-attention field assigner (~380K params)."""

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_fields: int = 4,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        n_text_priors: int = N_TEXT_PRIORS,
        text_feat_dim: int = 768,
        text_pool_learned: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fields = n_fields
        self.n_text_priors = n_text_priors
        self.n_layers = n_layers
        self.text_feat_dim = text_feat_dim
        self.text_pool_learned = text_pool_learned
        heads = _pick_n_heads(hidden_dim, n_heads)
        self.n_heads = heads

        self.field_queries = nn.Parameter(torch.randn(n_fields, hidden_dim) * 0.02)
        self.text_proj = nn.Linear(text_feat_dim, hidden_dim)
        self.bbox_proj = nn.Linear(8, hidden_dim)
        self.prior_proj: nn.Module | None
        if n_text_priors > 0:
            self.prior_proj = nn.Linear(n_text_priors, hidden_dim)
        else:
            self.prior_proj = None
        # Change D — opt-in learned attention pool over TrOCR encoder
        # tokens.  ``None`` when disabled so the state_dict is bit-compatible
        # with legacy mean-pool checkpoints.  When enabled, :meth:`forward`
        # accepts a 4-D ``text_feats`` of shape ``(B, N, T, text_feat_dim)``
        # and attention-pools along the ``T`` (token) axis using
        # ``softmax(text_pool_proj(feats))`` weights — ~``text_feat_dim``+1
        # extra params, which preserves ``SUBTOTAL`` / ``CASH TENDERED``
        # sub-word signals that the mean-pool erased.
        self.text_pool_proj: nn.Module | None
        if text_pool_learned:
            self.text_pool_proj = nn.Linear(text_feat_dim, 1)
        else:
            self.text_pool_proj = None
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * DEFAULT_FF_MULT,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _enrich_bbox(bbox: Tensor) -> Tensor:
        """Expand 4-d bbox to 8-d by adding centre (cx, cy) and size (w, h).

        Change C — bboxes are documented as *normalised* ``[0, 1]``
        throughout the pipeline (``Crop.bbox`` at train time, YOLO
        ``xyxyn`` at eval time).  Clamp to ``[0, 1]`` here so a
        regression that re-introduces raw-pixel coordinates drifts the
        ``bbox_proj`` input into noise-land silently; with the clamp the
        projection stays inside the distribution the encoder was
        trained on even if the caller accidentally feeds pixel space.
        The derived ``cx/cy/w/h`` are computed after the clamp and are
        themselves in ``[0, 1]``.
        """
        if bbox.shape[-1] == 8:
            # 8-d inputs must already have consistent x1/y1/x2/y2 and
            # cx/cy/w/h relations; we only clamp to the normalised range
            # and trust the caller's geometry (the training-time 4-d
            # path is what writes the derived features).
            return bbox.clamp(0.0, 1.0)
        if bbox.shape[-1] != 4:
            raise ValueError(
                f"bbox must be 4-d or 8-d per region, got {bbox.shape[-1]}",
            )
        bbox = bbox.clamp(0.0, 1.0)
        x1, y1, x2, y2 = bbox.unbind(-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        return torch.stack([x1, y1, x2, y2, cx, cy, w, h], dim=-1)

    def _maybe_attn_pool(self, text_feats: Tensor) -> Tensor:
        """Reduce 4-D ``(B, N, T, D)`` token features to ``(B, N, D)`` via
        learned attention pool; 3-D inputs pass through unchanged.

        Raises when ``text_pool_proj`` is not configured but 4-D input
        is supplied (caller contract mismatch).
        """
        if text_feats.dim() == 3:
            return text_feats
        if text_feats.dim() != 4:
            raise ValueError(
                f"text_feats must be 3-D (B,N,D) or 4-D (B,N,T,D); got {text_feats.dim()}-D.",
            )
        if self.text_pool_proj is None:
            raise ValueError(
                "4-D text_feats supplied but text_pool_learned=False; "
                "pre-pool (mean or otherwise) before calling forward.",
            )
        scores = self.text_pool_proj(text_feats).squeeze(-1)  # (B, N, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, N, T, 1)
        return (text_feats * weights).sum(dim=-2)  # (B, N, D)

    def forward(
        self, text_feats: Tensor, bbox_feats: Tensor,
        text_priors: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute field-assignment logits and per-receipt cross-attention.

        ``text_feats`` may be 3-D ``(B, N, text_feat_dim)`` — the legacy
        pre-pooled path — or, when ``text_pool_learned=True``, 4-D
        ``(B, N, T, text_feat_dim)`` so the model pools over the TrOCR
        encoder's ``T`` tokens itself (Change D).  Returns
        ``(logits (B, n_fields), attn_w (B, n_fields, N))`` where
        ``attn_w`` is the per-field soft assignment over regions used
        for inference and rendered as Fig.~\\ref{fig:attn_heatmap}.
        """
        text_feats = self._maybe_attn_pool(text_feats)
        bbox_feats = self._enrich_bbox(bbox_feats)
        kv = self.text_proj(text_feats) + self.bbox_proj(bbox_feats)
        if self.prior_proj is not None:
            if text_priors is None:
                text_priors = torch.zeros(
                    kv.size(0), kv.size(1), self.n_text_priors,
                    device=kv.device, dtype=kv.dtype,
                )
            kv = kv + self.prior_proj(text_priors)
        kv = self.input_norm(kv)
        kv = self.encoder(kv)
        q = self.field_queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        attn_out, attn_w = self.cross_attn(q, kv, kv)
        attn_out = self.cross_norm(attn_out + q)
        logits = self.classifier(attn_out).squeeze(-1)
        return logits, attn_w
