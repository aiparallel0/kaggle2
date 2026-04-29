"""PR-C / S2 — priors_v3 shape (14 columns) pin.

The V3 prior vector is 14-dimensional (V2 was 9-d, V1 was 6-d).
``models.focus_priors.N_TEXT_PRIORS_V3`` must equal 14 and
:func:`text_priors_v3` must return that many features per region.
"""
from __future__ import annotations


def test_n_text_priors_v3_is_14() -> None:
    from models.focus_priors import N_TEXT_PRIORS_V3

    assert N_TEXT_PRIORS_V3 == 14


def test_text_priors_v3_returns_14_columns() -> None:
    from models.focus_priors import N_TEXT_PRIORS_V3, text_priors_v3

    rows = ["LINE A 123", "JUMLAH BESAR RM 12.50", "CASH 20.00"]
    for i, text in enumerate(rows):
        out = text_priors_v3(text, y_norm=i / max(len(rows) - 1, 1),
                             is_last_money=False)
        assert len(out) == N_TEXT_PRIORS_V3, (text, len(out))
        assert all(isinstance(v, int | float) for v in out)
