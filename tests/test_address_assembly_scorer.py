"""PR-C / S0 — address-assembly scorer pin.

The S0 scorer (``models.pipeline_consensus_score.score_address_assembly``)
is the dominant residual lever on SROIE — confirms the multi-line
address F1 floor.  This test pins the scoring function's monotonicity
so a refactor cannot silently invert the sign.
"""
from __future__ import annotations


def test_clean_address_outranks_money_polluted() -> None:
    from models.pipeline_consensus_score import score_address_assembly

    anchor = "JALAN ALPHA 12 50000 KUALA LUMPUR MALAYSIA"
    clean = ["JALAN ALPHA 12", "50000 KUALA LUMPUR", "MALAYSIA"]
    polluted = ["JALAN ALPHA 12", "TOTAL RM 12.50", "MALAYSIA"]
    s_clean = score_address_assembly(clean, anchor)
    s_poll = score_address_assembly(polluted, anchor)
    assert s_clean > s_poll, (s_clean, s_poll)


def test_three_line_outranks_one_line() -> None:
    from models.pipeline_consensus_score import score_address_assembly

    anchor = "JALAN ALPHA 12 50000 KUALA LUMPUR MALAYSIA"
    one_line = ["JALAN ALPHA 12 50000 KUALA LUMPUR MALAYSIA"]
    three_line = ["JALAN ALPHA 12", "50000 KUALA LUMPUR", "MALAYSIA"]
    assert (
        score_address_assembly(three_line, anchor)
        > score_address_assembly(one_line, anchor)
    )


def test_empty_inputs_score_zero() -> None:
    from models.pipeline_consensus_score import score_address_assembly

    assert score_address_assembly([], "anchor") == 0.0
    assert score_address_assembly(["line"], "") == 0.0
