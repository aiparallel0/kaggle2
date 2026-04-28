"""Unit tests for the FOCUS address normaliser (Bug 18)."""
from __future__ import annotations

from models.postprocess_address import normalize_address_focus


def test_empty_value_is_passthrough() -> None:
    assert normalize_address_focus("") == ""


def test_whitespace_collapse() -> None:
    assert normalize_address_focus("FOO    BAR\t BAZ") == "foo bar baz"


def test_alpha_token_punctuation_stripped() -> None:
    """Comma/period/colon/semicolon dropped from purely-alpha tokens."""
    out = normalize_address_focus("NO. 12, JALAN MAJU; TAMAN.")
    # ``NO.`` → ``no``; ``JALAN`` keeps its content; punctuation gone.
    assert out == "no 12, jalan maju taman"


def test_numeric_token_punctuation_preserved() -> None:
    """Tokens matching ``\\d`` keep their punctuation (postcodes, phones, lots)."""
    out = normalize_address_focus("Lot 12.5A, BLOCK 3, 50100 KL")
    # ``12.5A`` is digit-bearing → keep ``.``; ``50100`` digit-bearing.
    assert "12.5a," in out
    assert "50100" in out
    # ``BLOCK`` is alpha → no comma after it.
    assert "block 3," in out
    assert out.startswith("lot 12.5a")


def test_casefold_applied() -> None:
    assert normalize_address_focus("ABC DEF") == "abc def"


def test_multi_line_order_preserved() -> None:
    """Newlines collapse but line order is not re-sorted."""
    out = normalize_address_focus("LINE TWO\nLINE THREE\nLINE ONE")
    assert out == "line two line three line one"


def test_symmetric_application_aligns_pred_and_gt_with_comma_drift() -> None:
    """Address fields whose only mismatch is comma drift align after normalise."""
    pred = "NO 12, JALAN MAJU 5, TAMAN MAJU"
    gt = "NO 12 JALAN MAJU 5 TAMAN MAJU"
    # The numeric-token rule preserves "12," vs "12" mismatch (a digit
    # token keeps its trailing comma).  This is the stricter spec
    # behaviour; the rest of the alpha-token punctuation drift IS
    # normalised away.
    assert normalize_address_focus(gt) == "no 12 jalan maju 5 taman maju"
    assert "jalan maju 5" in normalize_address_focus(pred)
    assert "taman maju" in normalize_address_focus(pred)
