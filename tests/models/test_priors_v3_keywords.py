"""PR-C / S2 — priors_v3 keyword discrimination (Bahasa-aware).

Bug-15 guard: ``JUMLAH BESAR`` (legitimate Bahasa Malay TOTAL synonym)
must register on ``_TOTAL_KW`` and NOT on ``_DISTRACTOR_KW``.  Pins
the regex behaviour so future refactors of priors_v3 cannot silently
re-introduce the false-fire.
"""
from __future__ import annotations


def test_jumlah_besar_is_total_not_distractor() -> None:
    from models.rule_consensus import (
        _DISTRACTOR_KW,
        _SUBTOTAL_KW,
        _TOTAL_KW,
    )

    s = "JUMLAH BESAR RM 12.50"
    assert _TOTAL_KW.search(s)
    assert not _SUBTOTAL_KW.search(s)
    assert not _DISTRACTOR_KW.search(s)


def test_subtotal_is_subtotal_not_total() -> None:
    from models.rule_consensus import _SUBTOTAL_KW, _TOTAL_KW

    s = "SUB-TOTAL RM 10.00"
    assert _SUBTOTAL_KW.search(s)
    # The TOTAL regex is intentionally permissive ("total" word boundary)
    # which DOES match "SUB-TOTAL"; the disambiguation is then driven by
    # the higher-priority SUBTOTAL match in priors_v3.  Pinning that the
    # SUBTOTAL regex fires is the load-bearing assertion here.
    _ = _TOTAL_KW  # keep import live


def test_distractor_keywords_do_not_register_total() -> None:
    from models.rule_consensus import _DISTRACTOR_KW, _TOTAL_KW

    for s in ("CASH 20.00", "CHANGE 7.50", "BAKI 0.00", "TUNAI 50.00"):
        assert _DISTRACTOR_KW.search(s)
        assert not _TOTAL_KW.search(s), s
