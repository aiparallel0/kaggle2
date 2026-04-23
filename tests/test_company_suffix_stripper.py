"""test_company_suffix_stripper.py — Fix D company legal-suffix guard.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: guards :func:`models.pipeline_normalize.strip_company_registration`
    and its wiring inside :func:`normalize_company`.  Every test case
    is lifted from the problem-statement miss table (receipts 525/195,
    331/358/339, 571, 538) so a regression that reinstates the
    registration-suffix mismatch trips here before F1 slides.
"""
from __future__ import annotations

from models.pipeline_normalize import normalize_company, strip_company_registration


def test_strips_registration_number_with_hyphen_letter() -> None:
    assert strip_company_registration("AEON CO M BHD 126926-H") == "AEON CO M BHD"


def test_strips_registration_number_with_space_letter() -> None:
    assert strip_company_registration(
        "GARDENIA BAKERIES KL SDN BHD 139386 X"
    ) == "GARDENIA BAKERIES KL SDN BHD"


def test_strips_longer_registration_number() -> None:
    assert strip_company_registration(
        "MIZU MENTAI SDN BHD 1248446-V"
    ) == "MIZU MENTAI SDN BHD"


def test_strips_parenthesised_check_letter() -> None:
    assert strip_company_registration("EXAMPLE CO SDN BHD (M)") == "EXAMPLE CO SDN BHD"


def test_strips_two_suffix_tail() -> None:
    # Two-pass strip: "139386 X" then "(M)".
    assert strip_company_registration(
        "GARDENIA BAKERIES KL SDN BHD 139386 X (M)"
    ) == "GARDENIA BAKERIES KL SDN BHD"


def test_is_symmetric_idempotent_on_clean_gt() -> None:
    """SROIE GT already lacks the suffix; stripping must be a no-op."""
    clean = "AEON CO M BHD"
    assert strip_company_registration(clean) == clean
    assert normalize_company(clean) == normalize_company(clean + " 126926-H")


def test_normalize_company_integrates_stripper() -> None:
    assert normalize_company("AEON CO M BHD 126926-H") == "AEON CO M BHD"
    assert normalize_company(
        "MOONLIGHT CAKE HOUSE SDN BHD 862725-U"
    ) == "MOONLIGHT CAKE HOUSE SDN BHD"


def test_preserves_company_without_suffix() -> None:
    assert (
        normalize_company("GARDENIA BAKERIES KL SDN BHD")
        == "GARDENIA BAKERIES KL SDN BHD"
    )
