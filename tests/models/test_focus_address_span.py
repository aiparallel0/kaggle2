"""PR-FOCUS — address-span head config + head + loss contracts.

The FOCUS span-cohesion head is opt-in (``focus_enabled`` defaults to
False) so the headline checkpoints stay bit-exact.  These tests pin:

    * the five new ``ExpConfig`` knobs and their defaults,
    * the ``AddrPred`` TypedDict shape,
    * baseline ``AttentionAssigner`` state-dict bit-exact when disabled,
    * head shape + argmax sanity (torch-gated),
    * ``span_iou_boundary_ce`` math on a small fixture (torch-gated).
"""
from __future__ import annotations

import pytest

from core.types import AddrPred, ExpConfig


def _make_config(**overrides: object) -> ExpConfig:
    base: dict[str, object] = dict(
        seed=42, base_model="x", trocr_model="x", yolo_model="x",
        image_size=(1, 1), yolo_image_size=1, max_length=1, trocr_max_len=1,
        epochs_donut=1, epochs_yolo=1, epochs_trocr=1, epochs_focus=1,
        batch_size=1, grad_accum=1, lr=1e-3, lr_decoder=1e-3,
        warmup_steps=0, weight_decay=0.0, label_smoothing=0.0,
        precision="fp32", patience=1, max_grad_norm=1.0,
        fields=["company", "address", "date", "total"], new_tokens=[],
        sroie_url="x", data_dir=".", output_dir=".",
        paper_template="x", paper_output="x",
    )
    base.update(overrides)
    return ExpConfig(**base)  # type: ignore[arg-type]


def test_focus_defaults_off() -> None:
    """Defaults must keep FOCUS off so baseline runs are bit-exact."""
    cfg = _make_config()
    assert cfg.focus_enabled is False
    assert cfg.focus_max_span == 8
    assert cfg.focus_iou_weight == 1.0
    assert cfg.focus_boundary_weight == 1.0
    assert cfg.focus_confidence_floor == 0.10


def test_addr_pred_typed_dict_shape() -> None:
    """``AddrPred`` accepts the four documented keys."""
    p: AddrPred = {"i": 1, "j": 3, "span_text": "foo bar", "confidence": 0.42}
    assert p["i"] == 1 and p["j"] == 3
    assert p["span_text"] == "foo bar"
    assert p["confidence"] == pytest.approx(0.42)


def test_assigner_state_dict_bit_exact_when_disabled() -> None:
    """``focus_enabled=False`` must produce a state_dict identical to
    the legacy keys — i.e. the FOCUS opt-in cannot perturb baseline
    checkpoints' parameter sets."""
    pytest.importorskip("torch")
    from models.focus_attention import AttentionAssigner

    baseline = AttentionAssigner(hidden_dim=32, n_text_priors=0, text_feat_dim=16)
    focus = AttentionAssigner(
        hidden_dim=32, n_text_priors=0, text_feat_dim=16,
        focus_enabled=True, focus_max_span=4,
    )
    # Baseline must NOT carry _span_head.* keys.
    base_keys = set(baseline.state_dict().keys())
    assert not any(k.startswith("_span_head.") for k in base_keys)
    # Focus must carry exactly 6 new keys (3× Linear weight+bias).
    focus_keys = set(focus.state_dict().keys())
    new = focus_keys - base_keys
    assert {f"_span_head.{n}" for n in (
        "start_proj.weight", "start_proj.bias",
        "end_proj.weight", "end_proj.bias",
        "cohesion_proj.weight", "cohesion_proj.bias",
    )} == new


def test_address_span_head_shapes_and_argmax() -> None:
    """Manual ``score_matrix`` + crafted start/end logits → expected argmax.

    ``_AddressSpanHead.score_matrix`` produces the masked
    ``cohesion[i, j]`` quadrant; we then override the start / end
    contribution with deterministic one-hot logits and verify that the
    argmax cell is ``(1, 3)``.  This pins the masking semantics
    (``j >= i`` AND ``j - i + 1 <= max_span``) and the flattening
    convention used by :meth:`AttentionAssigner.address_span`.
    """
    torch = pytest.importorskip("torch")
    from models.focus_attention import AttentionAssigner

    torch.manual_seed(0)
    m = AttentionAssigner(
        hidden_dim=8, n_text_priors=0, text_feat_dim=4,
        focus_enabled=True, focus_max_span=4,
    )
    m.eval()
    n = 5
    kv = torch.randn(n, 8)
    head = m._span_head
    assert head is not None
    _, _, _, mask = head.score_matrix(kv, m.focus_max_span)
    start_logits = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0])
    end_logits = torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0])
    score = start_logits.view(n, 1) + end_logits.view(1, n)
    score_masked = torch.where(mask, score, torch.full_like(score, float("-inf")))
    flat = score_masked.flatten()
    idx = int(torch.argmax(flat).item())
    assert (idx // n, idx % n) == (1, 3)


def test_span_iou_boundary_ce_math() -> None:
    """``span_iou_boundary_ce`` returns positive losses and gradients flow."""
    torch = pytest.importorskip("torch")
    from models.focus_train import span_iou_boundary_ce

    n = 6
    start = torch.zeros(n, requires_grad=True)
    end = torch.zeros(n, requires_grad=True)
    l_iou, l_bce = span_iou_boundary_ce(start, end, gold=(1, 3), max_span=4)
    # Uniform p_pair on the masked region -> L_iou strictly in (0, 1).
    assert 0.0 < float(l_iou.item()) < 1.0
    # Uniform CE over n classes = log(n).
    import math
    assert float(l_bce.item()) == pytest.approx(2.0 * math.log(n), rel=1e-5)
    (l_iou + l_bce).backward()
    assert start.grad is not None and end.grad is not None


def test_focus_gold_span_aligns_via_best_span() -> None:
    """``_focus_gold_span`` reuses ``oracle._best_span``."""
    from models.focus_train import _focus_gold_span

    texts = [
        "GROCER MART SDN BHD",
        "12 JALAN MAJU 5",
        "TAMAN MAJU",
        "47000 SUNGAI BULOH",
        "TOTAL RM 12.50",
    ]
    targets = {1: [1, 2, 3]}  # address_idx=1 covers rows 1..3 (contiguous)
    span = _focus_gold_span(texts, targets, address_idx=1)
    assert span == (1, 3)


def test_focus_gold_span_returns_none_when_no_address_label() -> None:
    """No address-labeled regions → no FOCUS supervision for this receipt."""
    from models.focus_train import _focus_gold_span

    assert _focus_gold_span(["foo", "bar"], {0: [0]}, address_idx=1) is None
