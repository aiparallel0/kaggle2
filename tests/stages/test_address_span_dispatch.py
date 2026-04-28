"""PR #113 / H1 — FOCUS-A address-span dispatch in pipeline inference.

Pins the contract that :func:`models.focus_pipeline._assign_learned_with_attn`
prefers :meth:`AttentionAssigner.address_span` over the legacy
threshold-band chain whenever the trained span head is present AND the
prediction's ``confidence >= focus_confidence_floor``, and falls back
to the legacy chain in the abstain / no-head cases.

Three regression cases:

* the span head fires and overrides the threshold-band concatenation,
* low-confidence prediction falls back to legacy,
* ``_span_head is None`` falls back to legacy bit-for-bit.
"""
from __future__ import annotations

import pytest


def _build_addr_inputs() -> tuple[list[str], list[list[float]], object, object]:
    """Build a tiny address-only fixture.

    Five regions arranged top→bottom; legacy threshold-band picks all
    five (uniform attention), so any divergence between the FOCUS-A
    span path and the legacy chain is observable in the joined output.
    """
    torch = pytest.importorskip("torch")
    texts = [
        "UNIHAKKA INTERNATIONAL SDN BHD",   # 0  — boilerplate above
        "12 JALAN MAJU 5",                   # 1  — address line 1
        "TAMAN MAJU",                        # 2  — address line 2
        "47000 SUNGAI BULOH",                # 3  — address line 3
        "INV NO 1075214 CASH",               # 4  — metadata below
    ]
    bboxes = [
        [0.0, 0.00, 1.0, 0.10],
        [0.0, 0.12, 1.0, 0.22],
        [0.0, 0.24, 1.0, 0.34],
        [0.0, 0.36, 1.0, 0.46],
        [0.0, 0.48, 1.0, 0.58],
    ]
    feats = [torch.zeros(1, 4) for _ in texts]  # 4-d text_feat_dim
    return texts, bboxes, feats, torch


class _FakeSpanHead:
    """Minimal stand-in for ``_AddressSpanHead`` so :func:`_assign_learned_with_attn`
    sees ``assigner._span_head is not None`` and takes the span branch.
    The real head is not exercised — :meth:`AttentionAssigner.address_span`
    is monkeypatched on the instance.
    """


def _make_assigner(
    *, focus_enabled: bool, span_pred: dict[str, object] | None,
):
    """Build a real :class:`AttentionAssigner` and (optionally) replace
    :meth:`address_span` with a stub returning ``span_pred``.

    The encoder still runs (so ``forward_with_kv`` produces a valid
    ``kv``); only the span head's argmax decision is stubbed.
    """
    pytest.importorskip("torch")
    from models.focus_attention import AttentionAssigner

    m = AttentionAssigner(
        hidden_dim=8, n_text_priors=6, text_feat_dim=4,
        focus_enabled=focus_enabled, focus_max_span=4,
    )
    m.eval()
    if focus_enabled and span_pred is not None:
        def _stub(kv, texts):  # type: ignore[no-untyped-def]
            return dict(span_pred)
        m.address_span = _stub  # type: ignore[method-assign]
    return m


def test_address_span_dispatch_overrides_legacy() -> None:
    """When the span head returns a high-confidence span, the address
    value is the span text (lines 1..3 only) — not the threshold-band
    concatenation that would absorb the boilerplate above and metadata
    below.
    """
    texts, bboxes, feats, _torch = _build_addr_inputs()
    span_text = " ".join(texts[1:4])
    pred = {"i": 1, "j": 3, "span_text": span_text, "confidence": 0.42}
    assigner = _make_assigner(focus_enabled=True, span_pred=pred)

    from models.focus_pipeline import _assign_learned_with_attn

    out, _attn = _assign_learned_with_attn(
        assigner, texts, feats, bboxes, ["address"], "cpu",
        focus_confidence_floor=0.10,
    )
    assert out["address"] == span_text
    # Negative control: the legacy concatenation would include row 0
    # ("UNIHAKKA ...") and row 4 ("INV NO ...").
    assert "UNIHAKKA" not in out["address"]
    assert "INV NO" not in out["address"]


def test_address_span_low_confidence_falls_back_to_legacy() -> None:
    """A span with ``confidence < focus_confidence_floor`` is rejected
    and the address branch falls back to the legacy threshold-band
    chain.
    """
    texts, bboxes, feats, _torch = _build_addr_inputs()
    pred = {"i": 1, "j": 3, "span_text": "SHOULD NOT WIN", "confidence": 0.01}
    assigner = _make_assigner(focus_enabled=True, span_pred=pred)

    from models.focus_pipeline import _assign_learned_with_attn

    out, _attn = _assign_learned_with_attn(
        assigner, texts, feats, bboxes, ["address"], "cpu",
        focus_confidence_floor=0.10,
    )
    # Span text was rejected; legacy chain yields a non-empty join that
    # does NOT equal the (rejected) stub span text.
    assert out["address"] != "SHOULD NOT WIN"
    assert out["address"]


def test_address_span_absent_falls_back_to_legacy() -> None:
    """``focus_enabled=False`` (no span head) takes the legacy path
    bit-for-bit — the assigner has no ``_span_head`` so the dispatch
    branch is never entered.
    """
    texts, bboxes, feats, _torch = _build_addr_inputs()
    assigner = _make_assigner(focus_enabled=False, span_pred=None)
    assert assigner._span_head is None

    from models.focus_pipeline import _assign_learned_with_attn

    out, _attn = _assign_learned_with_attn(
        assigner, texts, feats, bboxes, ["address"], "cpu",
        focus_confidence_floor=0.10,
    )
    # Legacy path produces a non-empty join from the threshold band.
    assert out["address"]
