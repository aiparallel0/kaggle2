"""PR-FOCUS — FOCUS-T (relational) and FOCUS-C (positional) head contracts.

The FOCUS framework (paper §III-D rewrite) factors the AttentionAssigner
into four field-specific sub-heads.  PR #106 shipped FOCUS-A (address
span); this PR adds FOCUS-T (total) and FOCUS-C (company).  These tests
pin:

    * the six new ``ExpConfig`` knobs and their defaults,
    * the ``TotalPred`` / ``CompanyPred`` TypedDict shapes,
    * baseline state-dict bit-exact when sub-flags are off,
    * head shape + argmax sanity (torch-gated),
    * priors_v4 column-index pin and 20-d shape,
    * arithmetic_witnesses_v4 fires exactly when SUB+TAX = TOTAL.
"""
from __future__ import annotations

import pytest

from core.types import CompanyPred, ExpConfig, TotalPred


def _make_config(**overrides: object) -> ExpConfig:
    base: dict[str, object] = dict(
        seed=42, base_model="x", trocr_model="x", yolo_model="x",
        image_size=(1, 1), yolo_img_size=1, max_length=1, trocr_max_len=1,
        epochs_donut=1, epochs_yolo=1, epochs_trocr=1, epochs_assigner=1,
        batch_size=1, grad_accum=1, lr=1e-3, lr_decoder=1e-3,
        warmup_steps=0, weight_decay=0.0, label_smoothing=0.0,
        precision="fp32", patience=1, max_grad_norm=1.0,
        fields=["company", "address", "date", "total"], new_tokens=[],
        sroie_url="x", data_dir=".", output_dir=".",
        paper_template="x", paper_output="x",
    )
    base.update(overrides)
    # ExpConfig has 60+ typed fields; constructing it from a kwargs dict
    # (so tests can override individual fields) requires erasing the
    # mapping's type — mypy cannot prove every key matches its parameter.
    return ExpConfig(**base)  # type: ignore[arg-type]  # mirrors test_focus_address_span.py


def test_focus_t_c_defaults_off() -> None:
    """All FOCUS-T / FOCUS-C / priors_v4 flags must default to False / 1.0."""
    cfg = _make_config()
    assert cfg.focus_total_enabled is False
    assert cfg.focus_total_witness_weight == 1.0
    assert cfg.focus_company_enabled is False
    assert cfg.focus_company_y_weight == 1.0
    assert cfg.focus_company_boilerplate_weight == 1.0
    assert cfg.priors_v4 is False


def test_total_company_pred_shapes() -> None:
    """``TotalPred`` / ``CompanyPred`` accept the documented keys."""
    t: TotalPred = {"i": 2, "text": "TOTAL RM 12.50", "confidence": 0.42}
    c: CompanyPred = {"i": 0, "text": "GROCER MART", "confidence": 0.91}
    assert t["i"] == 2 and c["i"] == 0
    assert t["text"].startswith("TOTAL") and c["text"].startswith("GROCER")
    assert t["confidence"] == pytest.approx(0.42)
    assert c["confidence"] == pytest.approx(0.91)


def test_priors_v4_shape() -> None:
    """priors_v4 must be 20-d and the column-index constants must agree."""
    from models.attention_priors import (
        N_TEXT_PRIORS_V4,
        V4_IS_COMPANY_BOILERPLATE_IDX,
        V4_MONEY_NORM_IDX,
        V4_WITNESS_IDX,
        V4_Y_NORM_IDX,
        text_priors_v4,
    )

    assert N_TEXT_PRIORS_V4 == 20
    assert V4_WITNESS_IDX == N_TEXT_PRIORS_V4 - 1
    out = text_priors_v4(
        "TOTAL RM 12.50", y_norm=0.5, is_last_money=True,
        money_value_norm=1.0, witness_self=1.0,
    )
    assert len(out) == N_TEXT_PRIORS_V4
    # The named columns must carry the right values.
    assert out[V4_Y_NORM_IDX] == pytest.approx(0.5)
    assert out[V4_MONEY_NORM_IDX] == pytest.approx(1.0)
    assert out[V4_WITNESS_IDX] == pytest.approx(1.0)
    # SDN BHD / BERHAD / etc. should fire on a known boilerplate line.
    boil = text_priors_v4(
        "GROCER MART SDN BHD", y_norm=0.0, is_last_money=False,
        money_value_norm=0.0, witness_self=0.0,
    )
    assert boil[V4_IS_COMPANY_BOILERPLATE_IDX] == pytest.approx(1.0)


def test_arithmetic_witnesses_v4_fires_on_sub_plus_tax() -> None:
    """Witness=1 only on the line whose money equals SUB+TAX (ε=2¢)."""
    from models.attention_priors import arithmetic_witnesses_v4

    texts = [
        "GROCER MART SDN BHD",
        "ITEM A           5.00",
        "ITEM B           5.00",
        "SUBTOTAL        10.00",
        "TAX              0.60",
        "TOTAL           10.60",
        "CASH            20.00",
    ]
    w = arithmetic_witnesses_v4(texts)
    assert w[5] == 1.0  # the TOTAL line: 10.00 + 0.60 == 10.60
    # The SUBTOTAL and TAX lines themselves must NOT carry the witness.
    assert w[3] == 0.0
    assert w[4] == 0.0
    # Lines without money — also 0.
    assert w[0] == 0.0


def test_arithmetic_witnesses_v4_silent_when_no_sub_tax_pair() -> None:
    """Receipts without both SUBTOTAL and TAX yield an all-zero witness col."""
    from models.attention_priors import arithmetic_witnesses_v4

    texts = ["ITEM A 5.00", "ITEM B 5.00", "TOTAL 10.00"]
    assert all(v == 0.0 for v in arithmetic_witnesses_v4(texts))


def test_assigner_state_dict_bit_exact_when_focus_t_c_off() -> None:
    """``focus_total_enabled=False`` + ``focus_company_enabled=False`` must
    produce a state_dict with no FOCUS-T/C parameters — the opt-in cannot
    perturb baseline checkpoints' parameter sets.
    """
    pytest.importorskip("torch")
    from models.attention_model import AttentionAssigner

    baseline = AttentionAssigner(hidden_dim=32, n_text_priors=0, text_feat_dim=16)
    foct = AttentionAssigner(
        hidden_dim=32, n_text_priors=0, text_feat_dim=16,
        focus_enabled=True,
        focus_total_enabled=True, focus_company_enabled=True,
    )
    base_keys = set(baseline.state_dict().keys())
    full_keys = set(foct.state_dict().keys())
    new = full_keys - base_keys
    # FOCUS-A span head (PR #106) parameters MUST also be present when
    # focus_enabled=True; FOCUS-T / FOCUS-C add their own.
    assert {f"_total_head.{n}" for n in (
        "score_proj.weight", "score_proj.bias",
        "witness_gate.weight", "witness_gate.bias",
        "money_gate.weight", "money_gate.bias",
    )} <= new
    assert {f"_company_head.{n}" for n in (
        "score_proj.weight", "score_proj.bias",
        "position_gate.weight", "position_gate.bias",
    )} <= new
    # And the legacy span head is also there because focus_enabled=True.
    assert any(k.startswith("_span_head.") for k in new)


def test_focus_t_head_shapes_and_argmax() -> None:
    """The FOCUS-T head's argmax follows the witness column when the
    score_proj is near-zero.  We instantiate the head, zero out
    ``score_proj`` weights/bias, set the witness column to one-hot at
    index 2, and verify ``total_pick`` returns that index.
    """
    torch = pytest.importorskip("torch")
    from models.attention_model import AttentionAssigner

    torch.manual_seed(0)
    m = AttentionAssigner(
        hidden_dim=8, n_text_priors=0, text_feat_dim=4,
        focus_enabled=True, focus_total_enabled=True,
    )
    m.eval()
    head = m._total_head
    assert head is not None
    # Zero score_proj so the witness column dominates.
    with torch.no_grad():
        head.score_proj.weight.zero_()
        head.score_proj.bias.zero_()
        # Saturate the witness gate so sigmoid(witness_gate(.)) ≈ 1.
        head.witness_gate.weight.zero_()
        head.witness_gate.bias.fill_(10.0)
    n = 5
    kv = torch.randn(n, 8)
    witness = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    pred = m.total_pick(kv, ["a", "b", "c", "d", "e"], witness)
    assert pred["i"] == 2
    assert pred["text"] == "c"
    assert 0.0 < pred["confidence"] <= 1.0


def test_focus_c_head_shapes_and_argmax() -> None:
    """The FOCUS-C head's argmax follows ``-y_weight * y_norm
    - boilerplate_weight * boilerplate``.  We zero score_proj, set the
    boilerplate prior on every line except index 0, and verify the head
    selects index 0 (the line WITHOUT the legal-entity suffix).
    """
    torch = pytest.importorskip("torch")
    from models.attention_model import AttentionAssigner

    torch.manual_seed(0)
    m = AttentionAssigner(
        hidden_dim=8, n_text_priors=0, text_feat_dim=4,
        focus_enabled=True, focus_company_enabled=True,
        focus_company_y_weight=0.0,  # isolate the boilerplate term
        focus_company_boilerplate_weight=10.0,
    )
    m.eval()
    head = m._company_head
    assert head is not None
    with torch.no_grad():
        head.score_proj.weight.zero_()
        head.score_proj.bias.zero_()
    n = 4
    kv = torch.randn(n, 8)
    y = torch.zeros(n)
    boil = torch.tensor([0.0, 1.0, 1.0, 1.0])  # 0 is the trade-name line
    pred = m.company_pick(kv, ["GROCER MART", "X SDN BHD", "Y BHD", "Z LTD"], y, boil)
    assert pred["i"] == 0
    assert pred["text"] == "GROCER MART"


def test_total_pick_raises_when_focus_t_off() -> None:
    """Calling ``total_pick`` without ``focus_total_enabled`` must raise."""
    torch = pytest.importorskip("torch")
    from models.attention_model import AttentionAssigner

    m = AttentionAssigner(hidden_dim=4, n_text_priors=0, text_feat_dim=4)
    with pytest.raises(RuntimeError, match="focus_total_enabled=False"):
        m.total_pick(torch.zeros(2, 4), ["a", "b"], torch.zeros(2))


def test_company_pick_raises_when_focus_c_off() -> None:
    """Calling ``company_pick`` without ``focus_company_enabled`` must raise."""
    torch = pytest.importorskip("torch")
    from models.attention_model import AttentionAssigner

    m = AttentionAssigner(hidden_dim=4, n_text_priors=0, text_feat_dim=4)
    with pytest.raises(RuntimeError, match="focus_company_enabled=False"):
        m.company_pick(
            torch.zeros(2, 4), ["a", "b"], torch.zeros(2), torch.zeros(2),
        )


def test_forward_overrides_total_row_when_focus_t_enabled() -> None:
    """When FOCUS-T is on and field_names resolves "total", the
    corresponding ``attn_w`` row must equal ``softmax(final)`` from the
    FOCUS-T head — making the existing pos-mass NLL automatically use the
    head's distribution at training time.
    """
    torch = pytest.importorskip("torch")
    from models.attention_model import AttentionAssigner
    from models.attention_priors import N_TEXT_PRIORS_V4, V4_WITNESS_IDX

    torch.manual_seed(0)
    fields = ["company", "address", "date", "total"]
    m = AttentionAssigner(
        hidden_dim=8, n_fields=4, text_feat_dim=4,
        n_text_priors=N_TEXT_PRIORS_V4,
        focus_enabled=True, focus_total_enabled=True,
        field_names=fields,
    )
    m.eval()
    with torch.no_grad():
        # Saturate the witness gate; witness col one-hot at index 1.
        head = m._total_head
        assert head is not None
        head.score_proj.weight.zero_()
        head.score_proj.bias.zero_()
        head.witness_gate.weight.zero_()
        head.witness_gate.bias.fill_(10.0)
    b, n = 1, 3
    text_feats = torch.randn(b, n, 4)
    bbox_feats = torch.randn(b, n, 4)
    priors = torch.zeros(b, n, N_TEXT_PRIORS_V4)
    priors[0, 1, V4_WITNESS_IDX] = 1.0
    _, attn_w = m(text_feats, bbox_feats, priors)
    # The "total" row (index 3) must put most mass on region 1.
    total_row = attn_w[0, fields.index("total")]
    assert int(torch.argmax(total_row).item()) == 1
    # And it must be a valid distribution (sum to 1).
    assert float(total_row.sum().item()) == pytest.approx(1.0, abs=1e-5)


def test_priors_v4_dispatch_in_pipeline_assign() -> None:
    """``_build_priors`` accepts ``N_TEXT_PRIORS_V4`` and emits 20-d rows."""
    from models.attention_priors import N_TEXT_PRIORS_V4
    from models.pipeline_assign import _build_priors

    texts = [
        "MERCHANT NAME",
        "ITEM A 5.00",
        "SUBTOTAL 10.00",
        "TAX 0.60",
        "TOTAL 10.60",
    ]
    bboxes = [[0.0, i * 0.1, 1.0, (i + 1) * 0.1] for i in range(len(texts))]
    out = _build_priors(texts, bboxes, N_TEXT_PRIORS_V4)
    assert all(len(row) == N_TEXT_PRIORS_V4 for row in out)


def test_focus_diagnostics_sidecar(tmp_path: object) -> None:
    """``_emit_focus_diagnostics`` writes a JSON sidecar enumerating which
    of FOCUS-A/T/C/D are active for the run, so downstream auditors do not
    need to re-read the run config."""
    import json as _json
    from pathlib import Path

    from stages.eval import _emit_focus_diagnostics

    out = Path(tmp_path)  # type: ignore[arg-type]  # pytest tmp_path fixture is os.PathLike
    cfg = _make_config(
        output_dir=str(out),
        focus_enabled=True,
        focus_total_enabled=True,
        focus_company_enabled=False,
        priors_v4=True,
    )
    _emit_focus_diagnostics(cfg)
    payload = _json.loads((out / "metrics" / "focus_diagnostics.json").read_text())
    assert payload["focus_enabled"] is True
    assert payload["priors_v4"] is True
    assert payload["sub_heads"]["FOCUS-A"]["enabled"] is True
    assert payload["sub_heads"]["FOCUS-T"]["enabled"] is True
    assert payload["sub_heads"]["FOCUS-C"]["enabled"] is False
    # FOCUS-D stays off by design.
    assert payload["sub_heads"]["FOCUS-D"]["enabled"] is False
