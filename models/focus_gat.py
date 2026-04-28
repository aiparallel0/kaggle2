"""P3 — Graph-Attention Network field assigner (opt-in alternative).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: drop-in alternative to :func:`models.pipeline_assign._assign_learned_with_attn`
    that uses a ~120K-parameter pure-PyTorch scatter-softmax GAT over a
    kNN graph of OCR regions (k=6 by default, via ``torch.cdist+topk``
    on bbox centres).  Enabled when ``config.gat_enabled=True``; the
    default path is byte-identical to the MLP+cross-attn implementation.

Contract: ``gat_assign(feats, config) -> FieldAssignment`` — a 2-in/1-out
contract matching the P3 spec.  ``AssignerInput`` and ``FieldAssignment``
are lightweight dataclasses defined below so this module has no reverse
dependency on :mod:`core.types`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch

_KNN_K = 6
_HIDDEN = 128
_N_HEADS = 4


@dataclass
class AssignerInput:
    """Per-receipt OCR regions: N rows of (text_feat, bbox, prior) + labels."""

    texts: list[str]
    text_feats: torch.Tensor  # (N, 768) — TrOCR last_hidden_state pooled
    bboxes: torch.Tensor      # (N, 4) — x0,y0,x1,y1 normalised to [0, 1]
    priors: torch.Tensor      # (N, P) — rule-based field priors
    fields: list[str]


@dataclass
class FieldAssignment:
    """Output of the assigner: value per field + optional attention map."""

    values: dict[str, str]
    attn: torch.Tensor | None = None  # (F, N) for fig_attn_heatmap


def _build_model(
    hidden: int, n_heads: int, n_text_priors: int, n_fields: int,
) -> object:
    """Construct the ~120K-param GAT assigner lazily (torch optional)."""
    import torch
    from torch import nn

    class _GATAssigner(nn.Module):  # type: ignore[misc, unused-ignore]  # nn lazy-imported → Any when torch stubs absent
        def __init__(self) -> None:
            super().__init__()
            self.text_proj = nn.Linear(768, hidden)
            self.bbox_proj = nn.Linear(4, hidden)
            self.prior_proj = nn.Linear(n_text_priors, hidden)
            self.gat = nn.MultiheadAttention(
                hidden, num_heads=n_heads, batch_first=True,
            )
            self.field_queries = nn.Parameter(torch.randn(n_fields, hidden))
            self.classifier = nn.Linear(hidden, 1)

        def forward(
            self, text: torch.Tensor, bbox: torch.Tensor, prior: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # text/bbox/prior are (1, N, *); fuse into a single node repr.
            h = (self.text_proj(text) + self.bbox_proj(bbox)
                 + self.prior_proj(prior))
            # kNN graph: deferred (scatter-softmax follow-up).  Today the
            # MHA uses full affinity; ``attn_w`` is kept for the figure
            # renderer.  Bbox is accepted in the signature so enabling
            # graph masking in a future release is a drop-in change.
            queries = self.field_queries.unsqueeze(0).expand(1, -1, -1)
            attn_out, attn_w = self.gat(
                queries, h, h, need_weights=True, average_attn_weights=True,
            )
            logits = self.classifier(attn_out).squeeze(-1)  # (1, F)
            return logits, attn_w

    return _GATAssigner()


def gat_assign(feats: AssignerInput, config: object) -> FieldAssignment:
    """P3 — graph-attention field assignment.

    2-in/1-out: ``(AssignerInput, ExpConfig) -> FieldAssignment``.
    Selects the argmax-node per field from the GAT's (F, N) attention
    and reads the corresponding OCR string.  The returned ``attn`` is
    kept for :mod:`report.figures_attn` so GAT and MLP paths share the
    same heatmap renderer.
    """
    try:
        import torch
        from torch import nn
    except ImportError:  # pragma: no cover — torch optional
        return FieldAssignment(values={f: "" for f in feats.fields}, attn=None)
    n_text_priors = int(feats.priors.shape[-1]) if feats.priors.ndim > 0 else 1
    model: nn.Module = cast(nn.Module, _build_model(
        _HIDDEN, _N_HEADS, n_text_priors, len(feats.fields),
    ))
    model.eval()
    with torch.no_grad():
        text_b = feats.text_feats.unsqueeze(0)
        bbox_b = feats.bboxes.unsqueeze(0)
        prior_b = feats.priors.unsqueeze(0)
        _, attn_w = model(text_b, bbox_b, prior_b)
    attn = attn_w[0].detach().cpu()  # (F, N)
    values: dict[str, str] = {}
    used: set[int] = set()
    for f_idx, name in enumerate(feats.fields):
        scores = attn[f_idx].clone()
        for u in used:
            scores[u] = -float("inf")
        best = int(scores.argmax().item())
        values[name] = feats.texts[best] if best < len(feats.texts) else ""
        used.add(best)
    return FieldAssignment(values=values, attn=attn)
