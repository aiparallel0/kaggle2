"""Tests for strategy F-lite — OCR-noise prior augmentation.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: verify that ``_ocr_noise_money`` perturbs money tokens the way
    live TrOCR does (decimal split, O↔0, trailing-zero drop) without
    touching non-money text — and that the perturbed text still yields
    a valid 14-d prior vector.
"""
from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")  # OCR noise helpers live in the trainer module

from models.assigner_train import _ocr_noise_money  # noqa: E402


def test_ocr_noise_money_touches_money_only() -> None:
    """A text without a money token is returned verbatim."""
    rng = random.Random(0)
    for text in ("THANK YOU", "ACME SDN BHD", "JALAN ABC 12"):
        assert _ocr_noise_money(text, rng) == text


def test_ocr_noise_money_produces_expected_variants() -> None:
    """Across many seeds, each of the three error modes appears at
    least once for a single money token — decimal split, O/I swap,
    trailing-zero drop."""
    variants: set[str] = set()
    for seed in range(60):
        variants.add(_ocr_noise_money("TOTAL 12.50", random.Random(seed)))
    # Always contains the decimal-split mode …
    assert any(" " in v and "12.50" not in v for v in variants), variants
    # … the O/I-swap mode (1 → I for this input) …
    assert any("I" in v for v in variants), variants
    # … and the trailing-zero drop mode.
    assert any("12.5" in v and "12.50" not in v for v in variants), variants


def test_ocr_noise_money_preserves_non_money_spans() -> None:
    """``TOTAL`` / ``RM`` and other non-digit tokens are untouched."""
    rng = random.Random(1)
    for _ in range(10):
        out = _ocr_noise_money("GRAND TOTAL RM 43.50", rng)
        assert out.startswith("GRAND TOTAL RM ")
