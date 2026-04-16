"""Unit tests for report.inject — pure string substitution, no ML deps."""
from __future__ import annotations

from report.inject import inject_results


def test_float_is_formatted_to_four_decimals() -> None:
    out = inject_results("F1 = \\VAR{f1}", {"f1": 0.8523456})
    assert out == "F1 = 0.8523"


def test_int_and_string_passthrough() -> None:
    tmpl = "epochs=\\VAR{epochs} precision=\\VAR{precision}"
    out = inject_results(tmpl, {"epochs": 10, "precision": "bf16"})
    assert out == "epochs=10 precision=bf16"


def test_missing_key_leaves_placeholder_intact() -> None:
    # Unresolved placeholders must stay visible so broken metrics are obvious
    # in the generated paper rather than silently deleted.
    out = inject_results("\\VAR{never_set}", {"other": 1})
    assert "\\VAR{never_set}" in out


def test_multiple_occurrences_of_same_key_replaced() -> None:
    out = inject_results("\\VAR{f1}/\\VAR{f1}", {"f1": 0.5})
    assert out == "0.5000/0.5000"


def test_negative_float_formatted() -> None:
    out = inject_results("gap=\\VAR{gap}", {"gap": -0.1234})
    assert out == "gap=-0.1234"
