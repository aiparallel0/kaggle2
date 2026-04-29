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

from typing import TYPE_CHECKING, cast

from models.focus_addr_penalty import boundary_prior_vec
from models.focus_priors import (
    N_TEXT_PRIORS,
    V4_IS_COMPANY_BOILERPLATE_IDX,
    V4_WITNESS_IDX,
    V4_Y_NORM_IDX,
)

try:
    import torch
    import torch.nn as nn

    _NN_BASE: type = nn.Module
except ImportError:  # lightweight CI — torch not installed
    _NN_BASE = object

if TYPE_CHECKING:
    from torch import Tensor

from core.types import AddrPred, CompanyPred, CompanySpanPred, TotalPred

# PR-ADDR-PREC-2 — additive penalty applied at inference-time only to
# the FOCUS-A ``score`` matrix before argmax.  ``-FOCUS_ADDR_BOUNDARY_PENALTY``
# is added to row ``i`` and column ``j`` whenever the corresponding text
# carries a boundary signal (company-boilerplate / money / date / phone
# / GST / receipt-metadata via :func:`models.consensus._is_addr_boundary`).
# Empirically the head's logits sit in roughly ``[-3, +3]`` after warm-up
# so a penalty of ``5.0`` is large enough to repel a header / footer cell
# without saturating the argmax on long, valid addresses (whose interior
# regions all carry ``prior=0``).  Kept as a module constant so callers
# don't have to plumb it through the inference path.
FOCUS_ADDR_BOUNDARY_PENALTY = 5.0

# Architecture defaults — single-source-of-truth (PR-A / T-A2 / L1).
#
# The repo historically carried THREE competing "truths" for the
# AttentionAssigner architecture:
#
#   * legacy 384-d / 6-layer (~7–8M params; ``DEFAULT_*`` constants
#     here, retained as aliases for back-compat),
#   * mini    192-d / 3-layer (~1.16M params; previously ``MINI_*``,
#     now ``SHIPPED_*`` because this is what the published paper
#     checkpoint uses),
#   * runtime ``ExpConfig`` 384/6 (matching the legacy defaults).
#
# A reviewer who clones the repo and instantiates ``AttentionAssigner()``
# with no arguments would historically have got the 384/6 model — which
# does NOT match the shipped checkpoint and therefore does NOT reproduce
# the paper's numbers.  The single-source-of-truth rule is now: the
# defaults below match the SHIPPED checkpoint, and the legacy 384/6
# constants are aliases preserved for code that explicitly opts into the
# bigger backbone (training scripts, bug-replay).  Existing checkpoints
# load bit-exact via the saved ``hyperparams`` blob in
# ``models/attention_assign.py::_load_assigner``.
#
# Verified by introspection at commit fd9d7b0 (see
# ``report/combine_ext.py::merge_assigner_arch``).
SHIPPED_HIDDEN_DIM = 192
SHIPPED_N_LAYERS = 3
SHIPPED_N_HEADS = 8
LEGACY_HIDDEN_DIM = 384
LEGACY_N_LAYERS = 6
LEGACY_N_HEADS = 12
DEFAULT_DROPOUT = 0.1
DEFAULT_FF_MULT = 2  # FFN hidden = hidden_dim * DEFAULT_FF_MULT

# Back-compat aliases — used by callers that explicitly want the
# big-backbone variant (e.g. legacy training scripts, bug-replay).
# Kept here so external code that imports ``DEFAULT_HIDDEN_DIM`` /
# ``MINI_HIDDEN_DIM`` does not break.  Prefer the ``SHIPPED_*`` /
# ``LEGACY_*`` names in new code.
DEFAULT_HIDDEN_DIM = LEGACY_HIDDEN_DIM
DEFAULT_N_HEADS = LEGACY_N_HEADS
DEFAULT_N_LAYERS = LEGACY_N_LAYERS
MINI_HIDDEN_DIM = SHIPPED_HIDDEN_DIM
MINI_N_LAYERS = SHIPPED_N_LAYERS


def _pick_n_heads(hidden_dim: int, requested: int) -> int:
    """Largest divisor of hidden_dim ≤ requested (MultiheadAttention requires it)."""
    if hidden_dim <= 0:
        return 1
    for h in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % h == 0:
            return h
    return 1


class _AddressSpanHead(_NN_BASE):  # type: ignore[misc]
    """FOCUS span-cohesion head — 3 ``Linear(d, 1)`` projections.

    Operates on the post-encoder pre-cross-attn ``kv`` tensor ``H``
    (shape ``(N, d)``).  Produces per-token ``start`` / ``end`` logits
    plus a ``cohesion[i, j]`` matrix derived from the cumulative-sum
    span-mean trick — no quadratic materialisation of full token
    sequences, only of the span-mean ``(N, N, d)`` tensor (which is
    bounded by ``focus_max_span * N * d`` after the mask is applied).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.start_proj = nn.Linear(hidden_dim, 1)
        self.end_proj = nn.Linear(hidden_dim, 1)
        self.cohesion_proj = nn.Linear(hidden_dim, 1)

    def start_end(self, kv: Tensor) -> tuple[Tensor, Tensor]:
        """Per-token start / end logits ``(N,)`` for one receipt's ``kv``."""
        return self.start_proj(kv).squeeze(-1), self.end_proj(kv).squeeze(-1)

    def score_matrix(
        self, kv: Tensor, max_span: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute ``(start, end, score, mask)`` for one receipt's ``kv``.

        ``score[i, j] = start[i] + end[j] + cohesion[i, j]`` with
        ``cohesion[i, j] = cohesion_proj(span_mean[i, j])`` and
        ``span_mean[i, j] = (cumH[j+1] - cumH[i]) / (j - i + 1)``.
        ``mask`` is ``True`` for valid cells (``j >= i`` AND
        ``j - i + 1 <= max_span``).
        """
        n, d = kv.shape
        zeros = kv.new_zeros(1, d)
        cum = torch.cat([zeros, kv], dim=0).cumsum(0)  # (N+1, d)
        i_idx = torch.arange(n, device=kv.device).view(n, 1)
        j_idx = torch.arange(n, device=kv.device).view(1, n)
        length = (j_idx - i_idx + 1).clamp(min=1).to(kv.dtype)
        # span_mean[i, j] = (cum[j+1] - cum[i]) / (j - i + 1)
        span_sum = cum[j_idx + 1] - cum[i_idx]  # (N, N, d)
        span_mean = span_sum / length.unsqueeze(-1)
        cohesion = self.cohesion_proj(span_mean).squeeze(-1)  # (N, N)
        start, end = self.start_end(kv)
        score = start.view(n, 1) + end.view(1, n) + cohesion
        mask = (j_idx >= i_idx) & ((j_idx - i_idx + 1) <= max_span)
        return start, end, score, mask


class _CompanySpanHead(_NN_BASE):  # type: ignore[misc]
    """FOCUS-C span-cohesion head — 3 ``Linear(d, 1)`` projections.

    Mirrors :class:`_AddressSpanHead`: operates on the post-encoder
    pre-cross-attn ``kv`` tensor ``H`` (shape ``(N, d)``).  Produces
    per-token ``start`` / ``end`` logits plus a ``cohesion[i, j]``
    matrix derived from the cumulative-sum span-mean trick.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.start_proj = nn.Linear(hidden_dim, 1)
        self.end_proj = nn.Linear(hidden_dim, 1)
        self.cohesion_proj = nn.Linear(hidden_dim, 1)

    def start_end(self, kv: Tensor) -> tuple[Tensor, Tensor]:
        """Per-token start / end logits ``(N,)`` for one receipt's ``kv``."""
        return self.start_proj(kv).squeeze(-1), self.end_proj(kv).squeeze(-1)

    def score_matrix(
        self, kv: Tensor, max_span: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute ``(start, end, score, mask)`` for one receipt's ``kv``.

        ``score[i, j] = start[i] + end[j] + cohesion[i, j]`` with
        ``cohesion[i, j] = cohesion_proj(span_mean[i, j])`` and
        ``span_mean[i, j] = (cumH[j+1] - cumH[i]) / (j - i + 1)``.
        ``mask`` is ``True`` for valid cells (``j >= i`` AND
        ``j - i + 1 <= max_span``).
        """
        n, d = kv.shape
        zeros = kv.new_zeros(1, d)
        cum = torch.cat([zeros, kv], dim=0).cumsum(0)  # (N+1, d)
        i_idx = torch.arange(n, device=kv.device).view(n, 1)
        j_idx = torch.arange(n, device=kv.device).view(1, n)
        length = (j_idx - i_idx + 1).clamp(min=1).to(kv.dtype)
        # span_mean[i, j] = (cum[j+1] - cum[i]) / (j - i + 1)
        span_sum = cum[j_idx + 1] - cum[i_idx]  # (N, N, d)
        span_mean = span_sum / length.unsqueeze(-1)
        cohesion = self.cohesion_proj(span_mean).squeeze(-1)  # (N, N)
        start, end = self.start_end(kv)
        score = start.view(n, 1) + end.view(1, n) + cohesion
        mask = (j_idx >= i_idx) & ((j_idx - i_idx + 1) <= max_span)
        return start, end, score, mask


class _TotalHead(_NN_BASE):  # type: ignore[misc]
    """FOCUS-T relational head — 3 ``Linear(d, 1)`` projections.

    Operates on the post-encoder ``kv`` tensor ``H`` ``(N, d)`` for one
    receipt.  Forward returns the per-region ``final`` logits

        ``final = score_proj(H) + witness_weight * sigmoid(witness_gate(H))
                  * arithmetic_witness_self``

    with ``arithmetic_witness_self`` read from ``priors_v4[:, V4_WITNESS_IDX]``.
    The ``money_gate`` projection is exposed so the trainer can also gate
    on the per-region money-presence prior; it is unused inside
    :meth:`forward` (the witness column already encodes the money signal)
    but kept as a placeholder weight slot so future ablations that toggle
    a money-only baseline don't reshape the state_dict.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.witness_gate = nn.Linear(hidden_dim, 1)
        self.money_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self, kv: Tensor, prior_witness_col: Tensor, witness_weight: float,
    ) -> Tensor:
        """Per-region final logits ``(N,)`` for FOCUS-T pos-mass NLL."""
        score = self.score_proj(kv).squeeze(-1)
        gate = torch.sigmoid(self.witness_gate(kv).squeeze(-1))
        return cast("Tensor", score + witness_weight * gate * prior_witness_col)


class _CompanyHead(_NN_BASE):  # type: ignore[misc]
    """FOCUS-C positional head — 2 ``Linear(d, 1)`` projections.

    Operates on the post-encoder ``kv`` tensor ``H`` ``(N, d)`` for one
    receipt.  Forward returns the per-region ``final`` logits

        ``final = score_proj(H) - y_weight * y_norm
                  - boilerplate_weight * is_company_boilerplate``

    with the two prior columns read from
    ``priors_v4[:, V4_Y_NORM_IDX]`` and
    ``priors_v4[:, V4_IS_COMPANY_BOILERPLATE_IDX]``.  Both bias terms enter
    the logit ADDITIVELY so ``score_proj`` can compensate (the priors are
    a structural inductive bias, not a hard constraint).  The
    ``position_gate`` projection is held as a learnable slot for future
    work that may want to softly gate the y-bias on receipts where the
    top-of-page heuristic is unreliable; unused inside :meth:`forward`.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.position_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self, kv: Tensor, prior_y_col: Tensor, prior_boilerplate_col: Tensor,
        y_weight: float, boilerplate_weight: float,
    ) -> Tensor:
        """Per-region final logits ``(N,)`` for FOCUS-C pos-mass NLL."""
        score = self.score_proj(kv).squeeze(-1)
        return cast(
            "Tensor",
            score
            - y_weight * prior_y_col
            - boilerplate_weight * prior_boilerplate_col,
        )


class AttentionAssigner(_NN_BASE):  # type: ignore[misc]
    """Transformer + 4-query cross-attention field assigner (~380K params)."""

    def __init__(
        self,
        hidden_dim: int = SHIPPED_HIDDEN_DIM,
        n_fields: int = 4,
        n_heads: int = SHIPPED_N_HEADS,
        n_layers: int = SHIPPED_N_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        n_text_priors: int = N_TEXT_PRIORS,
        text_feat_dim: int = 768,
        text_pool_learned: bool = False,
        focus_enabled: bool = False,
        focus_max_span: int = 8,
        focus_total_enabled: bool = False,
        focus_total_witness_weight: float = 1.0,
        focus_company_enabled: bool = False,
        focus_company_y_weight: float = 1.0,
        focus_company_boilerplate_weight: float = 1.0,
        focus_company_span_enabled: bool = False,
        focus_company_span_max_span: int = 4,
        field_names: list[str] | None = None,
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
        # FOCUS — opt-in span-cohesion head over the post-encoder ``kv``.
        # ``None`` when disabled so legacy state_dicts load bit-exact.
        self.focus_enabled = focus_enabled
        self.focus_max_span = focus_max_span
        self._span_head: _AddressSpanHead | None
        if focus_enabled:
            self._span_head = _AddressSpanHead(hidden_dim)
        else:
            self._span_head = None
        # FOCUS-T / FOCUS-C — opt-in factored heads (paper §III-D rewrite).
        # The master ``focus_enabled`` toggle is preserved for back-compat
        # with PR #106; the sub-flags below gate FOCUS-T (relational) and
        # FOCUS-C (positional) independently so a baseline config with
        # ``focus_enabled=True`` and both sub-flags False reproduces PR #106
        # bit-for-bit (FOCUS-A only).  ``field_names`` is the optional
        # ``["company", "address", "date", "total"]``-shaped list used by
        # :meth:`forward` to find which row of ``attn_w`` to override
        # with the FOCUS-T / FOCUS-C output.  ``None`` keeps the legacy
        # behaviour (no override).
        self.focus_total_enabled = bool(focus_total_enabled)
        self.focus_total_witness_weight = float(focus_total_witness_weight)
        self.focus_company_enabled = bool(focus_company_enabled)
        self.focus_company_y_weight = float(focus_company_y_weight)
        self.focus_company_boilerplate_weight = float(focus_company_boilerplate_weight)
        self.field_names: list[str] | None = (
            list(field_names) if field_names is not None else None
        )
        self._total_head: _TotalHead | None
        if focus_enabled and focus_total_enabled:
            self._total_head = _TotalHead(hidden_dim)
        else:
            self._total_head = None
        self._company_head: _CompanyHead | None
        if focus_enabled and focus_company_enabled:
            self._company_head = _CompanyHead(hidden_dim)
        else:
            self._company_head = None
        # FOCUS-C span head (mirrors FOCUS-A).
        self.focus_company_span_enabled = bool(focus_company_span_enabled)
        self.focus_company_span_max_span = int(focus_company_span_max_span)
        self._company_span_head: _CompanySpanHead | None
        if focus_enabled and focus_company_span_enabled:
            self._company_span_head = _CompanySpanHead(hidden_dim)
        else:
            self._company_span_head = None

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

    def _encode_kv(
        self, text_feats: Tensor, bbox_feats: Tensor,
        text_priors: Tensor | None = None,
    ) -> Tensor:
        """Run the text+bbox+prior fusion + transformer encoder.

        Returns the post-encoder pre-cross-attn tensor ``kv`` of shape
        ``(B, N, hidden_dim)`` — the ``H`` consumed by the FOCUS span
        head.  Centralises the encode path so :meth:`forward` and
        :meth:`address_span` share a single implementation (and a
        single bit-exact code path when ``focus_enabled=False``).
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
        return cast("Tensor", kv)

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

        FOCUS-T / FOCUS-C — when the corresponding sub-head is configured
        AND :attr:`field_names` resolves the row for ``"total"`` /
        ``"company"``, the head's ``softmax(final)`` over regions
        replaces that row of ``attn_w``.  This keeps the pos-mass NLL in
        :func:`models.assigner_train._group_loss` unchanged: the trainer
        still reads ``attn_w[0, f_idx]`` per field, and the FOCUS heads
        simply furnish a different probability mass for ``total`` and
        ``company`` while the cross-attention output stays canonical for
        ``date`` and ``address``.
        """
        logits, attn_w, _kv = self.forward_with_kv(
            text_feats, bbox_feats, text_priors,
        )
        return logits, attn_w

    def forward_with_kv(
        self, text_feats: Tensor, bbox_feats: Tensor,
        text_priors: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sibling of :meth:`forward` that also returns the post-encoder ``kv``.

        Used by :func:`models.focus_pipeline._assign_learned_with_attn`
        (PR #113 / H1 fix) so the address branch can feed the trained
        FOCUS-A :meth:`address_span` head without re-encoding.  Output
        ``logits`` and ``attn_w`` are bit-identical to :meth:`forward`;
        ``kv`` has shape ``(B, N, hidden_dim)``.
        """
        kv = self._encode_kv(text_feats, bbox_feats, text_priors)
        q = self.field_queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        attn_out, attn_w = self.cross_attn(q, kv, kv)
        attn_out = self.cross_norm(attn_out + q)
        logits = self.classifier(attn_out).squeeze(-1)
        attn_w = self._maybe_focus_override(kv, attn_w, text_priors)
        return logits, attn_w, kv

    def _maybe_focus_override(
        self, kv: Tensor, attn_w: Tensor, text_priors: Tensor | None,
    ) -> Tensor:
        """Replace the ``total`` / ``company`` rows of ``attn_w`` with the
        FOCUS-T / FOCUS-C ``softmax(final)`` distributions when those heads
        are configured AND :attr:`field_names` resolves the row index AND
        ``text_priors`` carries the v4 columns.  No-op otherwise so legacy
        callers stay bit-exact.
        """
        if self.field_names is None or text_priors is None:
            return attn_w
        n_priors = text_priors.shape[-1]
        # FOCUS-T / FOCUS-C both index priors_v4 columns; the witness col
        # (last) is the highest index we need, so any tensor at least that
        # wide carries the FOCUS-T/C signal.  Narrower tensors mean the
        # caller is on v1/v2/v3 priors and the override is a no-op.
        if n_priors <= V4_WITNESS_IDX:
            return attn_w
        # Single-clone optimisation: clone ``attn_w`` at most once even
        # when both FOCUS-T and FOCUS-C fire.
        new_attn = attn_w
        cloned = False
        if (
            self._total_head is not None
            and "total" in self.field_names
        ):
            t_idx = self.field_names.index("total")
            witness_col = text_priors[..., V4_WITNESS_IDX]
            # text_priors is (B, N, P); witness_col is (B, N).  Iterate over
            # the (small) batch so each receipt's head fires on its own kv.
            t_rows: list[Tensor] = []
            for b in range(kv.size(0)):
                final_b = self._total_head(
                    kv[b], witness_col[b], self.focus_total_witness_weight,
                )
                t_rows.append(torch.softmax(final_b, dim=-1))
            new_attn = new_attn.clone()
            cloned = True
            new_attn[:, t_idx, :] = torch.stack(t_rows, dim=0)
        if (
            self._company_head is not None
            and "company" in self.field_names
        ):
            c_idx = self.field_names.index("company")
            y_col = text_priors[..., V4_Y_NORM_IDX]
            boil_col = text_priors[..., V4_IS_COMPANY_BOILERPLATE_IDX]
            c_rows: list[Tensor] = []
            for b in range(kv.size(0)):
                final_b = self._company_head(
                    kv[b], y_col[b], boil_col[b],
                    self.focus_company_y_weight,
                    self.focus_company_boilerplate_weight,
                )
                c_rows.append(torch.softmax(final_b, dim=-1))
            if not cloned:
                new_attn = new_attn.clone()
            new_attn[:, c_idx, :] = torch.stack(c_rows, dim=0)
        return new_attn

    def address_span(self, kv: Tensor, texts: list[str]) -> AddrPred:
        """FOCUS inference: argmax span over the post-encoder ``kv``.

        ``kv`` is the ``(N, d)`` slice for one receipt (the caller is
        responsible for batch-squeezing).  ``texts`` is the list of
        per-region texts in the same order as ``kv``; the predicted
        ``span_text`` is ``" ".join(texts[i:j+1])``.  ``confidence`` is
        ``softmax(score_flat)[argmax]`` with invalid cells masked to
        ``-inf`` before softmax so they contribute zero mass.

        Raises if the FOCUS head is not configured (caller should gate
        on :attr:`focus_enabled`).
        """
        if self._span_head is None:
            raise RuntimeError(
                "address_span called but focus_enabled=False; "
                "instantiate AttentionAssigner(focus_enabled=True).",
            )
        n = kv.shape[0]
        if n == 0 or len(texts) != n:
            return AddrPred(i=0, j=-1, span_text="", confidence=0.0)
        _, _, score, mask = self._span_head.score_matrix(kv, self.focus_max_span)
        # PR-ADDR-PREC-2 — repel boundary lines at the span endpoints.
        # ``prior`` is ``1.0`` on header / footer rows; subtracting
        # ``λ * (prior[i] + prior[j])`` from the ``(N, N)`` score so a
        # cell whose start OR end leaks into receipt boilerplate falls
        # below any clean interior cell.  Inference-side only — the
        # trainer's ``span_iou_boundary_ce`` loss path goes through
        # ``_span_head.start_end`` and never touches this method.
        prior_list = boundary_prior_vec(texts)
        if any(p > 0.0 for p in prior_list):
            prior = score.new_tensor(prior_list)
            penalty = FOCUS_ADDR_BOUNDARY_PENALTY * (
                prior.view(n, 1) + prior.view(1, n)
            )
            score = score - penalty
        neg_inf = torch.full_like(score, float("-inf"))
        score_masked = torch.where(mask, score, neg_inf)
        flat = score_masked.flatten()
        if not torch.isfinite(flat).any():
            return AddrPred(i=0, j=-1, span_text="", confidence=0.0)
        probs = torch.softmax(flat, dim=0)
        idx = int(torch.argmax(flat).item())
        i_star, j_star = idx // n, idx % n
        conf = float(probs[idx].item())
        return AddrPred(
            i=i_star, j=j_star,
            span_text=" ".join(texts[i_star : j_star + 1]),
            confidence=conf,
        )

    def total_pick(
        self, kv: Tensor, texts: list[str], prior_witness_col: Tensor,
    ) -> TotalPred:
        """FOCUS-T inference — argmax over ``final`` for one receipt.

        ``kv`` is the ``(N, d)`` slice for one receipt; ``texts`` is the
        per-region text list in the same order; ``prior_witness_col`` is
        the ``(N,)`` slice of ``priors_v4[:, V4_WITNESS_IDX]``.  Returns
        a :class:`TotalPred` with ``i = argmax(final)`` and
        ``confidence = softmax(final)[i]``.  Raises if FOCUS-T was not
        configured (caller should gate on
        :attr:`focus_total_enabled`).
        """
        if self._total_head is None:
            raise RuntimeError(
                "total_pick called but focus_total_enabled=False; "
                "instantiate AttentionAssigner(focus_total_enabled=True).",
            )
        n = kv.shape[0]
        if n == 0 or len(texts) != n:
            return TotalPred(i=-1, text="", confidence=0.0)
        final = self._total_head(
            kv, prior_witness_col, self.focus_total_witness_weight,
        )
        if not torch.isfinite(final).any():
            return TotalPred(i=-1, text="", confidence=0.0)
        probs = torch.softmax(final, dim=0)
        i_star = int(torch.argmax(final).item())
        return TotalPred(
            i=i_star, text=texts[i_star],
            confidence=float(probs[i_star].item()),
        )

    def company_pick(
        self, kv: Tensor, texts: list[str],
        prior_y_col: Tensor, prior_boilerplate_col: Tensor,
    ) -> CompanyPred:
        """FOCUS-C inference — argmax over ``final`` for one receipt.

        ``kv`` is the ``(N, d)`` slice for one receipt; ``texts`` is the
        per-region text list; ``prior_y_col`` and ``prior_boilerplate_col``
        are the ``(N,)`` slices of ``priors_v4`` columns
        :data:`V4_Y_NORM_IDX` and :data:`V4_IS_COMPANY_BOILERPLATE_IDX`.
        Returns a :class:`CompanyPred` with ``i = argmax(final)`` and
        ``confidence = softmax(final)[i]``.  Raises if FOCUS-C was not
        configured.
        """
        if self._company_head is None:
            raise RuntimeError(
                "company_pick called but focus_company_enabled=False; "
                "instantiate AttentionAssigner(focus_company_enabled=True).",
            )
        n = kv.shape[0]
        if n == 0 or len(texts) != n:
            return CompanyPred(i=-1, text="", confidence=0.0)
        final = self._company_head(
            kv, prior_y_col, prior_boilerplate_col,
            self.focus_company_y_weight,
            self.focus_company_boilerplate_weight,
        )
        if not torch.isfinite(final).any():
            return CompanyPred(i=-1, text="", confidence=0.0)
        probs = torch.softmax(final, dim=0)
        i_star = int(torch.argmax(final).item())
        return CompanyPred(
            i=i_star, text=texts[i_star],
            confidence=float(probs[i_star].item()),
        )

    def company_span(self, kv: Tensor, texts: list[str]) -> CompanySpanPred:
        """FOCUS-C span inference: argmax span over the post-encoder ``kv``.

        ``kv`` is the ``(N, d)`` slice for one receipt (the caller is
        responsible for batch-squeezing).  ``texts`` is the list of
        per-region texts in the same order as ``kv``; the predicted
        ``span_text`` is ``" ".join(texts[i:j+1])``.  ``confidence`` is
        ``softmax(score_flat)[argmax]`` with invalid cells masked to
        ``-inf`` before softmax so they contribute zero mass.

        NOTE: Unlike :meth:`address_span`, this method does NOT apply
        boundary-prior penalty internally — the lexical anchor filter
        happens downstream in :func:`models.focus_pipeline._company_anchor_filter`.

        Raises if the FOCUS-C span head is not configured (caller should gate
        on :attr:`focus_company_span_enabled`).
        """
        if self._company_span_head is None:
            raise RuntimeError(
                "company_span called but focus_company_span_enabled=False; "
                "instantiate AttentionAssigner(focus_company_span_enabled=True).",
            )
        n = kv.shape[0]
        if n == 0 or len(texts) != n:
            return CompanySpanPred(i=0, j=-1, span_text="", confidence=0.0)
        _, _, score, mask = self._company_span_head.score_matrix(
            kv, self.focus_company_span_max_span,
        )
        neg_inf = torch.full_like(score, float("-inf"))
        score_masked = torch.where(mask, score, neg_inf)
        flat = score_masked.flatten()
        if not torch.isfinite(flat).any():
            return CompanySpanPred(i=0, j=-1, span_text="", confidence=0.0)
        probs = torch.softmax(flat, dim=0)
        idx = int(torch.argmax(flat).item())
        i_star, j_star = idx // n, idx % n
        conf = float(probs[idx].item())
        return CompanySpanPred(
            i=i_star, j=j_star,
            span_text=" ".join(texts[i_star : j_star + 1]),
            confidence=conf,
        )
