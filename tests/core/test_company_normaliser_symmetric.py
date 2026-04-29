"""FOCUS-C company-normaliser pred/GT symmetry contract.

The legacy :func:`models.normalize.normalize_company` strips trailing
legal-registration suffixes (``126926-H``) and internal text
punctuation, but a residual ``"SDN. BHD."`` vs ``"SDN BHD"`` mismatch
still survived on a handful of SROIE receipts because the punctuation
collapse left an extra blank token.  :func:`normalize_company_focus`
(mirror of :func:`models.postprocess_address.normalize_address_focus`)
casefolds + token-level strips ``,.:;`` from non-numeric tokens so the
two sides reduce to the same token set.

These tests pin the pred/GT symmetry contract.
"""
from __future__ import annotations

from models.postprocess_company import normalize_company_focus


def test_empty_value_passthrough() -> None:
    assert normalize_company_focus("") == ""


def test_whitespace_collapse_and_casefold() -> None:
    assert normalize_company_focus("UNIHAKKA   INTERNATIONAL") == (
        "unihakka international"
    )


def test_period_in_sdn_bhd_symmetric() -> None:
    """The headline bug — ``SDN. BHD.`` and ``SDN BHD`` must collapse to
    the same casefolded token sequence.
    """
    pred = "UNIHAKKA INTERNATIONAL SDN. BHD."
    gt = "UNIHAKKA INTERNATIONAL SDN BHD"
    assert normalize_company_focus(pred) == normalize_company_focus(gt)


def test_trailing_punctuation_stripped_from_alpha_tokens() -> None:
    """``,.:;`` are dropped from purely-alpha tokens but NOT from tokens
    carrying any digit (so registration numbers like ``(123456-A)``
    keep their internal punctuation intact for the comparison).
    """
    out = normalize_company_focus("ACME TRADING CO; MARKETING.")
    assert "co" in out.split()
    assert "marketing" in out.split()
    assert ";" not in out
    assert "." not in out


def test_digit_bearing_token_punctuation_preserved() -> None:
    """``(123456-A)`` is digit-bearing → punctuation preserved (only
    casefolded)."""
    out = normalize_company_focus("ACME (123456-A)")
    assert "(123456-a)" in out


def test_multiline_input_collapses_to_space_separated() -> None:
    """``"GROCER MART\\nSDN BHD"`` must reduce to a single line so the
    pipeline-eval comparison sees the same token set as a one-line GT.
    """
    out = normalize_company_focus("GROCER MART\nSDN BHD")
    assert out == "grocer mart sdn bhd"


def test_pred_gt_symmetric_on_ocr_drift_pairs() -> None:
    """Symmetry battery — every realistic OCR-drift pair from the live
    miss table must collapse pred and GT to the same string.
    """
    pairs = [
        ("UNIHAKKA INTERNATIONAL SDN BHD", "UNIHAKKA INTERNATIONAL SDN. BHD."),
        ("AEON CO. (M) SDN BHD", "AEON CO (M) SDN BHD"),
        ("GARDENIA BAKERIES KL SDN BHD", "GARDENIA BAKERIES KL SDN BHD."),
        ("BLUE OCEAN MART", "BLUE OCEAN MART;"),
    ]
    for a, b in pairs:
        assert normalize_company_focus(a) == normalize_company_focus(b), (
            f"asymmetric: {a!r} vs {b!r}"
        )


def test_pipeline_normaliser_is_normalize_company_focus_composed() -> None:
    """:data:`models.normalize_bundle.FIELD_NORMALISERS_PIPELINE['company']`
    must compose the legacy ``normalize_company`` with
    :func:`normalize_company_focus` so both passes apply symmetrically.
    """
    from models.normalize import normalize_company
    from models.normalize_bundle import FIELD_NORMALISERS_PIPELINE

    f = FIELD_NORMALISERS_PIPELINE["company"]
    pred = "UNIHAKKA INTERNATIONAL SDN. BHD. 126926-H"
    gt = "UNIHAKKA INTERNATIONAL SDN BHD"
    # ``normalize_company`` strips the registration-number suffix
    # symmetrically; the FOCUS-C casefold pass collapses the trailing
    # period.  End-to-end, pred and GT must match.
    assert f(pred) == f(gt)
    # And the composed normaliser is NOT a no-op against the legacy
    # ``normalize_company`` alone (it casefolds + strips trailing dot).
    assert f(pred) != normalize_company(pred)
