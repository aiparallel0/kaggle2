"""Unit tests for the 3-state monotone receipt-zone HMM.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: lock the structural invariants of
    :func:`models.zone_prior.decode_zone_posterior` — empty input,
    disabled config, monotone H→I→T decode, and the relational
    guarantee that ``argmax_y(p_header) < argmax_y(p_total)`` on a
    well-formed receipt with header / item / total lines.
"""
from __future__ import annotations

from core.types import ZoneConfig
from models.zone_prior import decode_zone_posterior


def _synth_receipt() -> list[tuple[str, float, float]]:
    """A synthetic 10-line SROIE-shaped receipt: 3 header / 4 item /
    3 totals, with the canonical y-ordering preserved."""
    return [
        ("ABC SDN BHD", 0.05, 1.0),
        ("NO. 1, JALAN MAIN", 0.12, 1.0),
        ("TEL: 03-1234-5678", 0.18, 1.0),
        ("1 x BURGER 5.50", 0.40, 0.0),
        ("2 x DRINK 4.00", 0.50, 0.0),
        ("1 x FRIES 3.00", 0.60, 0.0),
        ("SUBTOTAL 12.50", 0.72, 0.0),
        ("GST 6% 0.75", 0.78, 0.0),
        ("TOTAL 13.25", 0.85, 0.0),
        ("CASH 20.00", 0.92, 0.0),
    ]


def test_empty_input_returns_empty() -> None:
    """Empty line list returns an empty posterior."""
    assert decode_zone_posterior([], ZoneConfig()) == []


def test_disabled_returns_uniform() -> None:
    """``cfg.enabled=False`` returns a uniform 1/3 posterior so legacy
    callers see no zone signal."""
    lines = _synth_receipt()
    post = decode_zone_posterior(lines, ZoneConfig(enabled=False))
    assert len(post) == len(lines)
    for h, i, t in post:
        assert abs(h - 1.0 / 3) < 1e-9
        assert abs(i - 1.0 / 3) < 1e-9
        assert abs(t - 1.0 / 3) < 1e-9


def test_monotone_decode_assigns_zones() -> None:
    """Forward–backward decode lands header / items / totals in the
    expected y-order on a clean synthetic receipt."""
    lines = _synth_receipt()
    post = decode_zone_posterior(lines, ZoneConfig())
    # First line is hard-pinned to header (HMM start-state == H).
    assert post[0][0] > 0.9
    # Last line is hard-pinned to total (HMM end-state == T).
    assert post[-1][2] > 0.9
    # Item lines (3..5) should be argmax-items.
    for li in (3, 4, 5):
        h, it, t = post[li]
        assert it >= h and it >= t


def test_argmax_zones_are_strictly_ordered() -> None:
    """The relational invariant: ``argmax_y(p_header) < argmax_y(p_items)
    < argmax_y(p_total)``.  Back-transitions are impossible by
    construction; this test pins the contract that ``company`` (header
    zone) cannot land below ``total`` (totals zone) on any decoded
    receipt.
    """
    lines = _synth_receipt()
    post = decode_zone_posterior(lines, ZoneConfig())
    p_header = [p[0] for p in post]
    p_items = [p[1] for p in post]
    p_total = [p[2] for p in post]
    h_arg = max(range(len(p_header)), key=lambda i: p_header[i])
    i_arg = max(range(len(p_items)), key=lambda i: p_items[i])
    t_arg = max(range(len(p_total)), key=lambda i: p_total[i])
    assert h_arg < i_arg < t_arg


def test_posterior_rows_sum_to_one() -> None:
    """Every posterior row is a valid probability distribution."""
    lines = _synth_receipt()
    post = decode_zone_posterior(lines, ZoneConfig())
    for h, i, t in post:
        assert abs(h + i + t - 1.0) < 1e-6
        for v in (h, i, t):
            assert 0.0 <= v <= 1.0
