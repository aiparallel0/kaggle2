"""AttentionAssigner neural module (Transformer encoder + cross-attention).

Architecture (default ≈ 380 K params, trains in < 10 min on an RTX 4090):

  region inputs  ──[text_proj]──┐
                  [bbox_proj]   ├─ + → LayerNorm → TransformerEncoder ──┐
                  [prior_proj]──┘                                       │
                                                                        ▼
  field queries ──[cross-attn]─────────────────────────────────────► (B, F, H)
                                                                        │
                                                    LayerNorm + residual│
                                                                        ▼
                                               MLP classifier → logits (B, F)
                                               attn_w (B, F, N) ← field→region
                                                                 soft assignment
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

# Architecture defaults. Exposed as constants so save/load stay consistent.
DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_HEADS = 8
DEFAULT_N_LAYERS = 2
DEFAULT_DROPOUT = 0.1


def _pick_n_heads(hidden_dim: int, requested: int) -> int:
    """Largest divisor of ``hidden_dim`` that is ≤ ``requested``.

    ``nn.MultiheadAttention`` requires ``hidden_dim % n_heads == 0``; rather
    than failing loudly on pathological small hidden dims (e.g. unit tests
    passing ``hidden_dim=4``), degrade to fewer heads.
    """
    if hidden_dim <= 0:
        return 1
    for h in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % h == 0:
            return h
    return 1


class AttentionAssigner(_NN_BASE):  # type: ignore[misc]
    """Transformer + cross-attention field assigner (see module docstring)."""

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_fields: int = 4,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        n_text_priors: int = N_TEXT_PRIORS,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fields = n_fields
        self.n_text_priors = n_text_priors
        self.n_layers = n_layers
        heads = _pick_n_heads(hidden_dim, n_heads)
        self.n_heads = heads

        self.field_queries = nn.Parameter(torch.randn(n_fields, hidden_dim) * 0.02)
        self.text_proj = nn.Linear(768, hidden_dim)
        self.bbox_proj = nn.Linear(8, hidden_dim)
        self.prior_proj: nn.Module | None
        if n_text_priors > 0:
            self.prior_proj = nn.Linear(n_text_priors, hidden_dim)
        else:
            self.prior_proj = None
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _enrich_bbox(bbox: Tensor) -> Tensor:
        """``(B, N, 4)`` → ``(B, N, 8)`` by adding centre (cx, cy) + size (w, h)."""
        if bbox.shape[-1] == 8:
            return bbox
        if bbox.shape[-1] != 4:
            raise ValueError(
                f"bbox must be 4-d or 8-d per region, got {bbox.shape[-1]}",
            )
        x1, y1, x2, y2 = bbox.unbind(-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        return torch.stack([x1, y1, x2, y2, cx, cy, w, h], dim=-1)

    def forward(
        self, text_feats: Tensor, bbox_feats: Tensor,
        text_priors: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute field-assignment logits and cross-attention weights.

        Args:
            text_feats: ``(B, N, 768)`` TrOCR encoder hidden states.
            bbox_feats: ``(B, N, 4)`` or ``(B, N, 8)`` normalised boxes.
            text_priors: ``(B, N, n_text_priors)``. If ``None`` and the
                module has a prior head, a zero tensor is substituted.

        Returns:
            ``(logits (B, n_fields), attn_w (B, n_fields, N))``.
        """
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
