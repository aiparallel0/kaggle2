"""Paper 3 — three-head field-assignment ensemble (FOCUS-T + GAT + CNN).

Project: kaggle2 — FOCUS-$\\Sigma$ verification layer for document KIE.
Role: fuses three architecturally-distinct field-assignment heads into
    a single per-field probability vector via a small learned gating
    MLP.  This is the principal architectural differentiator between
    Paper 2 (regex + zone-prior HMM, no neural assignment) and Paper 3
    (multi-headed neural ensemble + structural verifier).  The three
    heads are:

      H1.  Cross-attention assigner — \\texttt{models/focus\\_attention.py}
           (the existing FOCUS-T head).  Per-line softmax over field
           queries; ~400K trainable parameters.
      H2.  Graph-attention assigner — \\texttt{models/focus\\_gat.py}
           (existing GAT alternative; \\texttt{config.gat\\_enabled}).
           Pure-PyTorch scatter-softmax over a kNN bbox-centre graph;
           ~120K trainable parameters.
      H3.  CNN visual head — \\texttt{models/focus\\_cnn.py} (this
           paper's new module).  Frozen ImageNet ResNet-18 feature
           per line bbox crop; ~50K trainable projection parameters.

    A 2-layer gating MLP (input: 3 field-probability vectors plus the
    receipt's zone-prior summary; output: 3-way softmax over heads
    per field) produces the final assignment.  The gate trains on the
    existing SROIE training fold — no new dataset is required, so
    Paper 3 honours the user's no-new-data constraint.

Bifurcation guarantee.  This module is gated behind
    \\texttt{config.focus\\_ensemble\\_enabled} (default False).
    Paper 2's preset (\\texttt{configs/paper2.json}) leaves it off;
    Paper 3's preset turns it on.  The CNN head is further gated
    behind \\texttt{config.focus\\_cnn\\_enabled} so an ensemble run
    can be reproduced text-only when GPU memory is tight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.types import ExpConfig

if TYPE_CHECKING:
    import torch

__all__ = ["EnsembleHead", "EnsembleScores", "ensemble_assign"]


@dataclass
class EnsembleScores:
    """Per-receipt ensemble output: per-field softmax over OCR lines.

    ``scores[f, i]`` is the gated mixture probability that line ``i``
    is the value for field ``f``; rows sum to 1.  ``head_weights`` is
    ``(n_fields, 3)`` — the gate's softmax over heads, exposed so the
    paper can decompose contribution per head.
    """

    scores: torch.Tensor
    head_weights: torch.Tensor
    field_names: list[str]


class EnsembleHead:
    """Three-head field-assignment ensemble with learned gating.

    The gate is a 2-layer MLP: input is the per-field row-probability
    from each head concatenated with the per-receipt zone-prior summary
    ``(p_header, p_items, p_totals)``; output is a softmax over heads.
    The MLP has ~3K parameters and is the only trainable component
    introduced by the ensemble itself (the underlying heads keep their
    own parameter counts).
    """

    def __init__(self, config: ExpConfig) -> None:
        self._config = config
        self._gate: Any | None = None  # lazy-initialised on first forward

    def _build_gate(self, n_fields: int) -> Any:
        """Build the gate MLP once we know ``n_fields`` from a live run."""
        import torch

        # Input: 3 heads x n_fields field-probabilities + 3 zone-prior summary
        in_dim = 3 * n_fields + 3
        # Output: 3 weights (one per head) x n_fields
        out_dim = 3 * n_fields
        gate = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, out_dim),
        )
        for layer in gate:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        return gate

    def fuse(
        self,
        h1_scores: torch.Tensor,
        h2_scores: torch.Tensor,
        h3_scores: torch.Tensor,
        zone_summary: torch.Tensor,
        field_names: list[str],
    ) -> EnsembleScores:
        """Mix the three head outputs through the learned gate.

        All ``h*_scores`` are ``(n_fields, n_lines)`` row-stochastic
        matrices.  ``zone_summary`` is the per-receipt
        ``(p_header, p_items, p_totals)`` triple averaged across lines.
        Returns the gated mixture as :class:`EnsembleScores`.

        Honest fallback: when only one head's scores are non-degenerate
        (e.g. CNN head returned zeros because the image file was
        unreadable), the gate naturally learns a near-degenerate
        weighting — but to make this auditable the head with all-zero
        rows is excluded from the softmax denominator at inference.
        """
        import torch

        n_fields = len(field_names)
        if self._gate is None:
            self._gate = self._build_gate(n_fields)

        # Per-field per-head summary: max probability mass each head
        # places on its top-1 line.  Compact enough for a small gate.
        h1_top = h1_scores.max(dim=-1).values
        h2_top = h2_scores.max(dim=-1).values
        h3_top = h3_scores.max(dim=-1).values
        gate_in = torch.cat(
            [h1_top, h2_top, h3_top, zone_summary], dim=-1,
        )
        gate_logits = self._gate(gate_in).reshape(n_fields, 3)
        # Audit hook: zero-out heads whose top-prob is zero (degraded path).
        eps = 1e-6
        live_mask = torch.stack([
            (h1_top > eps).float(),
            (h2_top > eps).float(),
            (h3_top > eps).float(),
        ], dim=-1)
        gate_logits = gate_logits.masked_fill(live_mask < 0.5, -1e9)
        head_weights = torch.softmax(gate_logits, dim=-1)

        # Weighted mixture per field.
        stacked = torch.stack([h1_scores, h2_scores, h3_scores], dim=0)
        weights = head_weights.transpose(0, 1).unsqueeze(-1)  # (3, n_fields, 1)
        mixed = (stacked * weights).sum(dim=0)
        return EnsembleScores(
            scores=mixed,
            head_weights=head_weights,
            field_names=field_names,
        )


def ensemble_assign(
    head_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    config: ExpConfig,
) -> EnsembleScores:
    """Module-level entry point: ``(h1, h2, h3, zone_summary)`` -> mixture."""
    head = EnsembleHead(config)
    h1, h2, h3, zone = head_inputs
    return head.fuse(h1, h2, h3, zone, list(config.fields))
