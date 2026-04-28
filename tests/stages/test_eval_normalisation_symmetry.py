"""Regression test for the PR #110 follow-up normaliser-symmetry bug.

When the eval-time pred contains the field-normaliser-canonical form
(``"no 1 jalan abc"``) and the gold contains the original SROIE form
(``"NO. 1, JALAN ABC,"``), the *headline* F1/NED/EM are computed on
the bundle returned by :func:`models.normalize_bundle.normalize_bundle`
— which strips both sides symmetrically — and so report ≈ 1.0.  But
before this fix the extended-metrics producer ran on a different
bundle (raw gold + normalised pred), producing P ≈ R ≈ 0.5 even
though the data round-trips perfectly through the symmetric
normaliser.

This test asserts the producer-side invariant: feeding a punctuation-
drift fixture through ``normalize_bundle`` followed by
``summarise_extended`` yields per-field precision and recall ≈ 1.0
on the address column.
"""
from __future__ import annotations

from pathlib import Path

from core.extended_metrics import summarise_extended
from core.metrics import compute_metrics
from core.types import EvalBundle, Field, Prediction, Receipt
from models.normalize_bundle import (
    FIELD_NORMALISERS_PIPELINE,
    normalize_bundle,
)


def _bundle_with_punctuation_drift() -> tuple[EvalBundle, EvalBundle]:
    """Return ``(raw, normalised)`` bundles for the same address pair."""
    receipts_raw = [
        Receipt(image_path=Path("/tmp/r1.jpg"),
                fields=[Field(name="address", value="NO. 1, JALAN ABC,")]),
        Receipt(image_path=Path("/tmp/r2.jpg"),
                fields=[Field(name="address",
                              value="LOT 9, TAMAN MELATI, 53100 KL.")]),
    ]
    preds_raw = [
        Prediction(receipt_id="r1",
                   fields=[Field(name="address", value="no 1 jalan abc")]),
        Prediction(receipt_id="r2",
                   fields=[Field(name="address",
                                 value="lot 9 taman melati 53100 kl")]),
    ]
    raw = EvalBundle(predictions=preds_raw, receipts=receipts_raw,
                     fields=["address"])
    n_preds, n_recs = normalize_bundle(
        preds_raw, receipts_raw, FIELD_NORMALISERS_PIPELINE,
    )
    normalised = EvalBundle(predictions=n_preds, receipts=n_recs,
                            fields=["address"])
    return raw, normalised


def test_normalised_bundle_has_unit_precision_and_recall() -> None:
    """Symmetric normalisation absorbs punctuation/casefold drift.

    Token-set intersection on the *normalised* bundle is total →
    ``precision_address ≈ recall_address ≈ 1.0``.  Pre-fix the
    producer received the raw bundle, intersection collapsed to ~50 %,
    and the bootstrap CI dragged the extended-metrics F1 below the
    headline F1 reported by ``compute_metrics``.
    """
    _, normalised = _bundle_with_punctuation_drift()
    m = compute_metrics(normalised)
    out = summarise_extended(m, normalised, n_iter=50, level=0.95)
    p = float(out["precision_address"])  # type: ignore[arg-type]
    r = float(out["recall_address"])  # type: ignore[arg-type]
    assert p > 0.99, f"precision_address={p:.4f} (expected ≈ 1.0)"
    assert r > 0.99, f"recall_address={r:.4f} (expected ≈ 1.0)"


def test_raw_bundle_demonstrates_the_pre_fix_asymmetry() -> None:
    """Sanity check: the *raw* bundle is what produced ``F1 > max(P, R)``.

    We deliberately do NOT assert symmetric scores on the raw bundle —
    its asymmetry is exactly the bug.  But we do assert that running
    the headline scorer on the raw bundle produces a *lower* F1 than
    on the normalised bundle, demonstrating that the normaliser is
    doing real work.  This pins the regression diagnostic so a future
    refactor cannot silently strip the address normaliser.
    """
    raw, normalised = _bundle_with_punctuation_drift()
    f1_raw = compute_metrics(raw).per_field_f1["address"]
    f1_norm = compute_metrics(normalised).per_field_f1["address"]
    assert f1_norm > f1_raw + 0.1, (
        f"normaliser is a no-op: f1_raw={f1_raw:.4f}, "
        f"f1_norm={f1_norm:.4f} — punctuation/casefold drift not absorbed"
    )
