"""Cross-attention field assigner — Transformer encoder + cross-attention.

The assigner takes N text regions produced by YOLO+TrOCR and maps them to
the four SROIE fields (company / date / address / total) via learned
field-specific query vectors.

Architecture (default ~380 K params, trains in <10 min on an RTX 4090):

  region inputs  ---[text_proj]--->  (B, N, H)
                +[bbox_proj(enriched)]
                +[prior_proj]                 ↓
                                     LayerNorm
                                     TransformerEncoder (n_layers)
                                              ↓
   field queries  ---[cross-attn]--->  (B, F, H)
                                     LayerNorm + residual
                                              ↓
                                     MLP classifier  → logits (B, F)
                                              ↑
                                   attn_w (B, F, N)  ←  field-to-region soft
                                                       assignment, used at
                                                       inference.

Why each piece:

* The previous hidden_dim=64, 1-layer cross-attention assigner had ~50 K
  parameters and no region-to-region contextualisation — each region
  was scored in isolation. On SROIE that starves the model of the
  "which line is this relative to its neighbours?" signal (company is
  top, total is near a money line, address is between them). A 2-layer
  TransformerEncoder before the cross-attention lets every region
  attend to every other region, restoring that context.

* ``bbox_feats`` is enriched from 4-d ``(x1, y1, x2, y2)`` to 8-d
  ``(x1, y1, x2, y2, cx, cy, w, h)``. Centre + size are cheap derived
  features but they let the bbox_proj head separate "this is a header"
  (small y-centre, wide) from "this is a line item" (mid y, narrow
  width) without the model having to re-derive them per forward pass.

* ``text_priors`` is a 6-d feature vector per region
  ``(length_log, digit_ratio, upper_ratio, has_money, has_date,
  has_colon)``. These are cheap textual signals that regex already
  extracts for the rule-based baseline; feeding them alongside the
  TrOCR embedding lets the assigner condition on "this line has a
  money figure" directly instead of having to re-extract it from
  768-d encoder hidden states.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn as nn

    _NN_BASE: type = nn.Module
except ImportError:  # lightweight CI — torch not installed
    _NN_BASE = object

if TYPE_CHECKING:
    from torch import Tensor


# ---------------------------------------------------------------------------
# Text-prior extraction (torch-free, safe to call in diagnostic scripts).
# ---------------------------------------------------------------------------
N_TEXT_PRIORS = 6

# Local copies so attention_assign stays importable without rule_based
# (lightweight CI just needs the module to import, not to run correctly).
_MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}\b")
_DATE_RE = re.compile(
    r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
    r"|\b\d{1,2}[\s/\-\.](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[\w/\-\.\s]*\d{2,4}\b",
    re.IGNORECASE,
)


def text_priors(text: str) -> list[float]:
    """Compute the 6-d text-prior feature vector for one region.

    Features:
      * ``length_log``   — log(1 + len), z-scored to [-1, 1]-ish range.
      * ``digit_ratio``  — fraction of digits (money/date lines score high).
      * ``upper_ratio``  — fraction of uppercase letters (company header cue).
      * ``has_money``    — 1.0 if a money regex matches, else 0.0.
      * ``has_date``     — 1.0 if a date regex matches, else 0.0.
      * ``has_colon``    — 1.0 if ':' appears (TOTAL: / DATE: label cue).
    """
    if not text:
        return [0.0] * N_TEXT_PRIORS
    s = text.strip()
    n = len(s)
    if n == 0:
        return [0.0] * N_TEXT_PRIORS
    n_digit = sum(c.isdigit() for c in s)
    n_letter = sum(c.isalpha() for c in s)
    n_upper = sum(c.isupper() for c in s)
    import math
    length_log = math.log1p(n) / 6.0  # ≈ [0, 1] for lines up to ~400 chars
    digit_ratio = n_digit / n
    upper_ratio = n_upper / max(n_letter, 1)
    has_money = 1.0 if _MONEY_RE.search(s) else 0.0
    has_date = 1.0 if _DATE_RE.search(s) else 0.0
    has_colon = 1.0 if ":" in s else 0.0
    return [length_log, digit_ratio, upper_ratio, has_money, has_date, has_colon]


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------

# Architecture defaults. Exposed as constants so save/load stay consistent
# and the tests can check them without string-munging.
DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_HEADS = 8
DEFAULT_N_LAYERS = 2
DEFAULT_DROPOUT = 0.1


def _pick_n_heads(hidden_dim: int, requested: int) -> int:
    """Pick the largest divisor of ``hidden_dim`` that is ≤ ``requested``.

    ``nn.MultiheadAttention`` requires ``hidden_dim % n_heads == 0``.
    Rather than failing loudly when a caller passes ``hidden_dim=4``
    (our tiny-network unit test), we degrade to fewer heads.
    """
    if hidden_dim <= 0:
        return 1
    for h in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % h == 0:
            return h
    return 1


class AttentionAssigner(_NN_BASE):  # type: ignore[misc]
    """Transformer + cross-attention field assigner.

    Args:
        hidden_dim: Width of every internal tensor (default 128).
        n_fields: Number of output field queries (default 4).
        n_heads: Multi-head count for both the encoder and the cross-
            attention. Coerced down to a divisor of ``hidden_dim`` if
            necessary.
        n_layers: TransformerEncoder depth.
        dropout: Dropout applied in the encoder + cross-attention.
        n_text_priors: Width of the per-region text-prior feature. Set
            to 0 to disable priors (backward-compatible with the previous
            signature).
    """

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
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _enrich_bbox(bbox: Tensor) -> Tensor:
        """``(B, N, 4)`` → ``(B, N, 8)``, adding centre (cx, cy) + size (w, h).

        Pre-enriched 8-d inputs pass through unchanged.
        """
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
        self,
        text_feats: Tensor,
        bbox_feats: Tensor,
        text_priors: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute field assignment logits and cross-attention weights.

        Args:
            text_feats: ``(B, N, 768)`` TrOCR encoder hidden states.
            bbox_feats: ``(B, N, 4)`` or ``(B, N, 8)`` normalised boxes.
            text_priors: ``(B, N, n_text_priors)``. Optional — when
                ``None`` and the module has a prior head, a zero tensor
                is substituted (lets existing call sites pass only the
                two legacy args without raising).

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


def _architecture_config(model: AttentionAssigner) -> dict[str, int]:
    """Extract the architecture hyperparameters that ``load_assigner``
    needs to rebuild an equivalent module. Kept simple: we save alongside
    the state dict so mismatched checkpoints fail loudly at load time
    rather than silently producing junk attention weights."""
    return {
        "hidden_dim": model.hidden_dim,
        "n_fields": model.n_fields,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "n_text_priors": model.n_text_priors,
    }


def save_assigner(model: AttentionAssigner, path: str) -> None:
    """Save state_dict + architecture config in a single torch file."""
    torch.save(
        {"state_dict": model.state_dict(), "config": _architecture_config(model)},
        path,
    )


def load_assigner(
    path: str,
    n_fields: int | None = None,
    hidden_dim: int | None = None,
) -> AttentionAssigner:
    """Load an ``AttentionAssigner`` from ``path``.

    Accepts both the new bundle format (dict with ``state_dict`` + ``config``)
    and the legacy format (bare state_dict). For the legacy format we infer
    ``hidden_dim`` / ``n_fields`` from the ``field_queries`` shape; all
    other architectural knobs fall back to the new defaults, which means
    legacy-format checkpoints only load if they were also produced with
    the new defaults.

    Positional overrides (``n_fields``, ``hidden_dim``) are honoured when
    the checkpoint leaves them ambiguous — typically used by callers that
    know the field list from their ``ExpConfig`` at load time.
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
            "hidden_dim": int(fq.shape[1]),
            "n_fields": int(fq.shape[0]),
            "n_heads": DEFAULT_N_HEADS,
            "n_layers": DEFAULT_N_LAYERS,
            "n_text_priors": N_TEXT_PRIORS,
        }
    if n_fields is not None:
        cfg["n_fields"] = n_fields
    if hidden_dim is not None:
        cfg["hidden_dim"] = hidden_dim
    m = AttentionAssigner(
        hidden_dim=cfg["hidden_dim"],
        n_fields=cfg["n_fields"],
        n_heads=cfg.get("n_heads", DEFAULT_N_HEADS),
        n_layers=cfg.get("n_layers", DEFAULT_N_LAYERS),
        n_text_priors=cfg.get("n_text_priors", N_TEXT_PRIORS),
    )
    m.load_state_dict(sd)
    return m
