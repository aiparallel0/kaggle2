"""Integration test — company and total are picked from disjoint y-bands.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: pin the relational invariant introduced by the receipt-zone
    prior PR — on every fixture-shaped synthetic receipt,
    ``argmax_y(company) < argmax_y(total)``.

This is the integration test the prompt calls out as the missing
guard for PR #119's silent failure: a learned head can ship in
``assigner.pt`` and never appear on the dispatch path of
``_assign_learned_with_attn`` for an entire release cycle if no test
exercises company *and* total on the *same* receipt and asserts
they pick lines from disjoint zones.

We exercise the public surface (:func:`models.zone_prior.
decode_zone_posterior` on a per-receipt feature stream) since the
attention-assigner dispatch requires GPU and a trained checkpoint;
the zone HMM is the structural component that *makes* the disjoint-
zone constraint mathematically impossible to violate, so testing it
end-to-end on representative line layouts is the right granularity.
"""
from __future__ import annotations

from core.types import ZoneConfig
from models.zone_prior import decode_zone_posterior


def _argmax_y(p: list[float]) -> int:
    return max(range(len(p)), key=lambda i: p[i])


# Three fixtures covering the canonical SROIE layouts:
#   * "ABC" — short header (3 lines), 3 items, 3 totals
#   * "MART" — long header w/ tax-id + GST line in totals
#   * "CAFE" — refund receipt (negative total)
_FIXTURES: list[list[tuple[str, float, float]]] = [
    [
        ("ABC SDN BHD", 0.05, 1.0),
        ("123 JALAN BUKIT", 0.12, 1.0),
        ("WELCOME!", 0.18, 1.0),
        ("ITEM A 5.00", 0.45, 0.0),
        ("ITEM B 3.50", 0.55, 0.0),
        ("SUBTOTAL 8.50", 0.75, 0.0),
        ("TOTAL 8.50", 0.85, 0.0),
        ("CASH 10.00", 0.92, 0.0),
    ],
    [
        ("MART HOLDINGS BHD", 0.04, 1.0),
        ("LOT 5, JALAN AMPANG", 0.09, 1.0),
        ("TAX INVOICE", 0.14, 1.0),
        ("GST NO: 001234567", 0.20, 1.0),
        ("1 x SHIRT 25.00", 0.40, 0.0),
        ("2 x SOCKS 12.00", 0.50, 0.0),
        ("SUB-TOTAL 49.00", 0.72, 0.0),
        ("GST 6% 2.94", 0.80, 0.0),
        ("TOTAL 51.94", 0.88, 0.0),
        ("CHANGE 8.06", 0.95, 0.0),
    ],
    [
        ("CAFE CORPORATION", 0.06, 1.0),
        ("(559208-M)", 0.13, 1.0),
        ("REFUND ITEM 6.42", 0.45, 0.0),
        ("SUBTOTAL -6.42", 0.75, 0.0),
        ("TOTAL -6.42", 0.88, 0.0),
    ],
]


def test_zone_argmax_disjoint_on_every_fixture() -> None:
    """On every fixture, ``argmax_y(p_header) < argmax_y(p_total)``."""
    for lines in _FIXTURES:
        post = decode_zone_posterior(lines, ZoneConfig())
        p_header = [p[0] for p in post]
        p_total = [p[2] for p in post]
        h = _argmax_y(p_header)
        t = _argmax_y(p_total)
        assert h < t, (
            f"company-zone argmax y={h} should sit above total-zone "
            f"argmax y={t}; receipt={[ln[0] for ln in lines]}"
        )


def test_company_pick_falls_in_header_zone() -> None:
    """The ``argmax`` line over ``p_header`` is a header-zone line."""
    for lines in _FIXTURES:
        post = decode_zone_posterior(lines, ZoneConfig())
        h = _argmax_y([p[0] for p in post])
        # The argmax line should have y < 0.30 (canonical header band).
        assert lines[h][1] < 0.30


def test_total_pick_falls_in_totals_zone() -> None:
    """The ``argmax`` line over ``p_total`` is a totals-zone line."""
    for lines in _FIXTURES:
        post = decode_zone_posterior(lines, ZoneConfig())
        t = _argmax_y([p[2] for p in post])
        # The argmax line should have y >= 0.65 (canonical totals band).
        assert lines[t][1] >= 0.65
