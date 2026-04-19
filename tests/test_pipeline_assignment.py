"""Verify the multi-region + regex-postprocess fixes in pipeline_eval.

Two distinct training/inference mismatches motivated the rewrite of
``_assign_learned`` and the new ``_postprocess_value`` helper:

A. **Multi-line address.**  ``assigner_train._group_loss`` minimises
   ``-log(sum(probs[positives]))`` so the address query learns to spread
   probability mass across every address line on the receipt.  Picking
   ``argmax`` at inference therefore yields one of three correct lines and
   caps token-F1 on the address field at ~0.33.  Picking every region
   above ``0.5 * max_attention`` and concatenating them in spatial order
   recovers the multi-line ground truth the model was actually trained for.

B. **Field-specific regex priors.**  TrOCR transcribes
   ``"TOTAL    12.30"`` for a line whose SROIE ground truth is just
   ``"12.30"``.  Token-F1 on that prediction is 1/(1+1+...) ≈ 0.5.
   Stripping with the same ``MONEY_RE`` / ``DATE_RE`` constants used by
   the rule-based baseline lifts the per-region F1 to ~1.0 without any
   additional model capacity.
"""
from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from models import pipeline_eval  # noqa: E402
from models.attention_assign import AttentionAssigner  # noqa: E402


def _attn_module(weights: list[list[float]]) -> Any:
    """Return a fake assigner whose ``__call__`` produces the given attention.

    ``weights[f][r]`` is the attention from field f over region r.
    """
    weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)  # (1, F, N)

    class _Fake(AttentionAssigner):
        def __init__(self) -> None:
            # hidden_dim=16 keeps the nn.MultiheadAttention(num_heads=…)
            # divisibility constraint happy under the new defaults without
            # forcing the test to hard-code them.
            super().__init__(hidden_dim=16, n_fields=weights_t.shape[1])

        def forward(  # type: ignore[override]
            self, text_feats: Any, bbox_feats: Any, text_priors: Any = None,
        ) -> tuple[Any, Any]:  # noqa: ARG002
            logits = torch.zeros(1, weights_t.shape[1])
            return logits, weights_t

    return _Fake().eval()


def test_multi_line_address_concatenated_in_spatial_order() -> None:
    """Three address regions at different y positions must all be picked
    and joined top→bottom, not just the highest-attention one."""
    fields = ["company", "date", "address", "total"]
    texts = ["ACME CORP", "2024-01-01", "789 Springfield", "456 Oak Ave",
             "123 Main St", "TOTAL 12.30"]
    feats = [torch.randn(1, 768) for _ in texts]
    bboxes = [
        [0.0, 0.05, 1.0, 0.10],   # company  (top)
        [0.0, 0.20, 1.0, 0.25],   # date
        [0.0, 0.55, 1.0, 0.60],   # addr line 3 (lowest of the three)
        [0.0, 0.45, 1.0, 0.50],   # addr line 2
        [0.0, 0.35, 1.0, 0.40],   # addr line 1 (top of the three)
        [0.0, 0.95, 1.0, 1.00],   # total (bottom)
    ]
    # Attention: each query points at its correct region(s).  Address spreads
    # ~equal mass over the three address lines (>= 0.5 * max).
    attn = [
        [0.9, 0.0, 0.0,  0.0,  0.0,  0.0],   # company → idx 0
        [0.0, 0.9, 0.0,  0.0,  0.0,  0.0],   # date    → idx 1
        [0.0, 0.0, 0.7,  0.8,  1.0,  0.0],   # address → idx 2,3,4
        [0.0, 0.0, 0.0,  0.0,  0.0,  0.9],   # total   → idx 5
    ]
    out = pipeline_eval._assign_learned(
        _attn_module(attn), texts, feats, bboxes, fields, device="cpu",
    )
    # Address must be all three lines, joined in y1-ascending order.
    assert out["address"] == "123 Main St 456 Oak Ave 789 Springfield"


def test_multi_line_threshold_excludes_weak_regions() -> None:
    """Regions with attention < 0.5 * max_attention are excluded from a
    multi-line field — this is the threshold that prevents the assigner
    from sweeping in arbitrary distractor lines."""
    fields = ["address"]
    texts = ["MAIN ST", "DISTRACTOR", "CITY"]
    feats = [torch.randn(1, 768) for _ in texts]
    bboxes = [[0, 0.1, 1, 0.2], [0, 0.5, 1, 0.6], [0, 0.7, 1, 0.8]]
    # max=1.0, threshold=0.5 → idx 1 (0.2) excluded, idx 0,2 included.
    attn = [[1.0, 0.2, 0.6]]
    out = pipeline_eval._assign_learned(
        _attn_module(attn), texts, feats, bboxes, fields, device="cpu",
    )
    assert out["address"] == "MAIN ST CITY"


def test_total_regex_strips_to_money_substring() -> None:
    fields = ["company", "date", "address", "total"]
    texts = ["TOTAL    12.30", "ACME", "01/01/2024", "MAIN ST"]
    feats = [torch.randn(1, 768) for _ in texts]
    bboxes = [[0, 0.9, 1, 1.0], [0, 0.0, 1, 0.1],
              [0, 0.1, 1, 0.2], [0, 0.5, 1, 0.6]]
    attn = [
        [0.0, 1.0, 0.0, 0.0],   # company → ACME
        [0.0, 0.0, 1.0, 0.0],   # date    → 01/01/2024
        [0.0, 0.0, 0.0, 1.0],   # address → MAIN ST
        [1.0, 0.0, 0.0, 0.0],   # total   → "TOTAL    12.30" → "12.30"
    ]
    out = pipeline_eval._assign_learned(
        _attn_module(attn), texts, feats, bboxes, fields, device="cpu",
    )
    assert out["total"] == "12.30"
    assert out["date"] == "01/01/2024"


def test_postprocess_keeps_raw_text_when_no_regex_match() -> None:
    """Defensive: a TrOCR transcription that omits the money pattern must
    not be silently emptied by the post-processor."""
    assert pipeline_eval._postprocess_value("total", "TOTAL: HANDWRITTEN") == "TOTAL: HANDWRITTEN"
    assert pipeline_eval._postprocess_value("date", "January First") == "January First"
    # Non-regex fields pass through.
    assert pipeline_eval._postprocess_value("company", "ACME CORP") == "ACME CORP"
    assert pipeline_eval._postprocess_value("address", "MAIN ST") == "MAIN ST"


def test_empty_texts_returns_empty() -> None:
    """Belt-and-suspenders for the YOLO-empty path (already handled at the
    eval_pipeline level, but the assigner must be safe in isolation)."""
    out = pipeline_eval._assign_learned(
        _attn_module([[0.0]]), [], [], [], ["company"], device="cpu",
    )
    assert out == {}
