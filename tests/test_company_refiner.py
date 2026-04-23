"""Tests for the strengthened ``_refine_company`` scorer.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: verifies that when the learned assigner picks a tagline / slogan
    but the rule-based ``_pick_company`` would have chosen the real
    store name, the scorer now prefers the store name.  Also verifies
    strategy H (diffuse-attention delegation) on the company field.
"""
from __future__ import annotations

from models.pipeline_consensus import _refine_company


def _bb(y0: float, y1: float) -> list[float]:
    return [0.0, y0, 1.0, y1]


def test_learned_tagline_loses_to_rule_pick_with_company_token() -> None:
    """Learned picks the slogan; rule picks ``ACME SDN BHD``.  The
    ``has_company_token`` bit on the rule pick breaks the tie."""
    texts = [
        "ACME SDN BHD",                # 0 — true company, rule pick
        "THE BEST CHICKEN RICE",       # 1 — tagline, learned pick
        "12 JALAN ABC",                # 2 — address
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    # Sharp attention on line 1 (the slogan) — not diffuse, but the
    # score ranking still promotes the rule pick thanks to SDN BHD.
    attn = [0.05, 0.9, 0.05]
    assert _refine_company(
        "THE BEST CHICKEN RICE", texts, bboxes, attn_row=attn,
    ) == "ACME SDN BHD"


def test_learned_valid_company_kept_when_sharp() -> None:
    """When the learned pick is itself a valid company header and
    attention is sharp, the refiner must keep it (no regression on
    receipts the assigner already gets right)."""
    texts = [
        "WELCOME",                     # 0 — header junk (not really, but lower-scoring)
        "ACME ENTERPRISE",             # 1 — valid learned pick
        "12 JALAN ABC",                # 2
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    attn = [0.05, 0.9, 0.05]
    assert _refine_company(
        "ACME ENTERPRISE", texts, bboxes, attn_row=attn,
    ) == "ACME ENTERPRISE"


def test_diffuse_attention_demotes_learned_pick() -> None:
    """When attention is diffuse and both candidates score equally on
    ``not_junk`` / ``has_token`` / ``upper``, strategy H demotes the
    learned candidate so the (topmost) rule pick wins."""
    texts = [
        "ACME STORE",                  # 0 — rule pick (topmost, no token)
        "BETA STORE",                  # 1 — learned pick (no token either)
        "12 JALAN ABC",                # 2
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    flat = [0.3, 0.35, 0.35]
    # Diffuse → learned demoted → rule's topmost wins.
    assert _refine_company(
        "BETA STORE", texts, bboxes, attn_row=flat,
    ) == "ACME STORE"


def test_no_attention_falls_back_to_score_only() -> None:
    """When no attention row is supplied (legacy caller / eval paths),
    the scorer still ranks candidates consistently."""
    texts = ["ACME SDN BHD", "RANDOM SLOGAN", "10 JLN X"]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    # No attn_row → treated as diffuse → rule pick wins on both score
    # and the demotion tier.
    assert _refine_company("RANDOM SLOGAN", texts, bboxes) == "ACME SDN BHD"


def test_learned_pick_dominates_when_it_has_company_token() -> None:
    """If the learned pick carries a company token and the rule pick
    doesn't, the learned pick must win regardless of attention
    confidence — we are not giving up legitimate signal."""
    texts = [
        "SHOP #12",                    # 0 — rule pick (topmost, no token)
        "ACME CORPORATION",            # 1 — learned pick with token
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    flat = [0.5, 0.5]
    assert _refine_company(
        "ACME CORPORATION", texts, bboxes, attn_row=flat,
    ) == "ACME CORPORATION"
