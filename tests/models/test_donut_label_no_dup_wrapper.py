"""Tests for Bug 8 fix: _build_label must NOT include leading <s_sroie> tag."""
from __future__ import annotations

from pathlib import Path

from core.types import Field, Receipt
from models.donut_dataset import _build_label


def _make_receipt() -> Receipt:
    return Receipt(
        image_path=Path("/tmp/fake/X00016469612.jpg"),
        fields=[
            Field(name="company", value="ACME Corp"),
            Field(name="date", value="01/01/2024"),
            Field(name="address", value="1 Main St"),
            Field(name="total", value="12.50"),
        ],
    )


def test_build_label_does_not_start_with_s_sroie() -> None:
    """Labels must not carry the leading <s_sroie> start tag (Bug 8 fix)."""
    label = _build_label(_make_receipt())
    assert not label.startswith("<s_sroie>"), (
        f"Label starts with <s_sroie> — duplicated wrapper would corrupt "
        f"training decoder_input_ids.  Got: {label[:60]!r}"
    )


def test_build_label_ends_with_eos_sroie() -> None:
    """Closing </s_sroie> must be present as the EOS sentinel."""
    label = _build_label(_make_receipt())
    assert label.endswith("</s_sroie>"), (
        f"Label must end with </s_sroie> EOS sentinel.  Got: {label[-30:]!r}"
    )


def test_build_label_has_exactly_one_closing_sroie() -> None:
    """Exactly one </s_sroie> at the end; no stray opening <s_sroie>."""
    label = _build_label(_make_receipt())
    assert label.count("<s_sroie>") == 0, "Unexpected <s_sroie> open tag in label"
    assert label.count("</s_sroie>") == 1, "Expected exactly one </s_sroie> close tag"


def test_build_label_contains_all_four_fields() -> None:
    """All four SROIE field tags appear in the label."""
    label = _build_label(_make_receipt())
    for field in ("company", "date", "address", "total"):
        assert f"<s_{field}>" in label, f"Missing opening tag <s_{field}>"
        assert f"</s_{field}>" in label, f"Missing closing tag </s_{field}>"


def test_build_label_field_values_present() -> None:
    """Field values appear verbatim between their open/close tags."""
    label = _build_label(_make_receipt())
    assert "<s_company>ACME Corp</s_company>" in label
    assert "<s_total>12.50</s_total>" in label


def test_build_label_first_tag_is_first_field() -> None:
    """Label starts immediately with the first field's open tag."""
    label = _build_label(_make_receipt())
    assert label.startswith("<s_company>"), (
        f"Expected label to start with <s_company>.  Got: {label[:40]!r}"
    )
