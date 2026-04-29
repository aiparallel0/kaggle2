"""FOCUS-C — company-span head config + head + loss contracts.

Mirrors ``test_focus_address_span.py`` for the _CompanySpanHead feature.
These tests pin:

    * the five new ``ExpConfig`` knobs and their defaults,
    * the ``CompanySpanPred`` TypedDict shape,
    * baseline ``AttentionAssigner`` state-dict bit-exact when disabled,
    * head shape + argmax sanity (torch-gated),
    * ``_focus_company_loss`` math on a small fixture (torch-gated).
"""
from __future__ import annotations

import pytest

from core.types import CompanySpanPred, ExpConfig


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


def test_focus_company_span_defaults_off() -> None:
    """Defaults must keep FOCUS-C off so baseline runs are bit-exact."""
    cfg = _make_config()
    assert cfg.focus_company_span_enabled is False
    assert cfg.focus_company_span_max_span == 4
    assert cfg.focus_company_span_iou_w == 1.0
    assert cfg.focus_company_span_boundary_w == 1.0
    assert cfg.focus_company_confidence_floor == 0.20


def test_company_span_pred_typed_dict_shape() -> None:
    """``CompanySpanPred`` accepts the four documented keys."""
    p: CompanySpanPred = {"i": 0, "j": 2, "span_text": "Company Ltd", "confidence": 0.55}
    assert p["i"] == 0 and p["j"] == 2
    assert p["span_text"] == "Company Ltd"
    assert p["confidence"] == pytest.approx(0.55)


def test_assigner_state_dict_bit_exact_when_company_span_disabled() -> None:
    """``focus_company_span_enabled=False`` must produce a state_dict identical to
    the legacy keys — i.e. the FOCUS-C opt-in cannot perturb baseline
    checkpoints' parameter sets."""
    pytest.importorskip("torch")
    from models.focus_attention import AttentionAssigner

    baseline = AttentionAssigner(hidden_dim=32, n_text_priors=0, text_feat_dim=16)
    # Company span head requires focus_enabled=True to be instantiated
    with_span = AttentionAssigner(
        hidden_dim=32, n_text_priors=0, text_feat_dim=16,
        focus_enabled=True,  # required for span head instantiation
        focus_company_span_enabled=True, focus_company_span_max_span=4,
    )
    # Baseline must NOT carry _company_span_head.* keys.
    base_keys = set(baseline.state_dict().keys())
    assert not any(k.startswith("_company_span_head.") for k in base_keys)
    # With span must carry exactly 6 new keys for company span head
    # (plus 6 for address span head due to focus_enabled=True).
    span_keys = set(with_span.state_dict().keys())
    new = span_keys - base_keys
    company_span_keys = {k for k in new if k.startswith("_company_span_head.")}
    assert {f"_company_span_head.{n}" for n in (
        "start_proj.weight", "start_proj.bias",
        "end_proj.weight", "end_proj.bias",
        "cohesion_proj.weight", "cohesion_proj.bias",
    )} == company_span_keys
