"""Unit test for Bug 3 longest-value list-merge (DONUT token2json)."""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from models.donut_eval import _token2json_safe  # noqa: E402


class _FakeProcessor:
    """Stand-in for DonutProcessor.token2json with list-return behaviour."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def token2json(self, tokens: str) -> Any:  # noqa: ARG002
        return self._result


def test_dict_passthrough() -> None:
    p = _FakeProcessor({"company": "ACME", "total": "10.00"})
    assert _token2json_safe(p, "") == {"company": "ACME", "total": "10.00"}


def test_list_merge_prefers_longest_value() -> None:
    # Multi-page CORD-style return: the second page has a longer address.
    p = _FakeProcessor([
        {"address": "123 MAIN ST", "total": "10.00"},
        {"address": "123 MAIN ST, SPRINGFIELD IL 62701"},
    ])
    merged = _token2json_safe(p, "")
    assert merged["address"] == "123 MAIN ST, SPRINGFIELD IL 62701"
    assert merged["total"] == "10.00"


def test_list_merge_handles_non_dict_entries() -> None:
    p = _FakeProcessor([{"company": "ACME"}, "stray", None])
    assert _token2json_safe(p, "") == {"company": "ACME"}


def test_unrecognised_return_yields_empty() -> None:
    p = _FakeProcessor(42)
    assert _token2json_safe(p, "") == {}


def test_outer_sroie_wrapper_is_unwrapped() -> None:
    # HuggingFace's DonutProcessor.token2json parses our `<s_sroie>…</s_sroie>`
    # label wrapper as a root key, yielding {"sroie": {...fields...}}. Without
    # unwrapping, per-field lookup ("company" / "date" / ...) finds nothing
    # and global F1 collapses to exactly 0.0000 — even when the model decoded
    # the fields correctly.
    p = _FakeProcessor({
        "sroie": {
            "company": "ACME",
            "date": "2023-01-01",
            "address": "123 MAIN ST",
            "total": "10.00",
        },
    })
    assert _token2json_safe(p, "") == {
        "company": "ACME",
        "date": "2023-01-01",
        "address": "123 MAIN ST",
        "total": "10.00",
    }


def test_string_value_with_embedded_tags_is_regex_extracted() -> None:
    # Regression: when the model emits a duplicated outer <s_sroie> wrapper
    # (decoder_start_token_id == labels[0]), token2json falls through to its
    # string branch and returns {"sroie": "<s_company>FOO</s_company>…"}.
    # _flatten_token2json must regex-extract the child (key, value) pairs.
    p = _FakeProcessor({
        "sroie": (
            "<s_company>ACME</s_company>"
            "<s_date>2023-01-01</s_date>"
            "<s_address>123 MAIN ST</s_address>"
            "<s_total>10.00</s_total>"
        ),
    })
    assert _token2json_safe(p, "") == {
        "company": "ACME",
        "date": "2023-01-01",
        "address": "123 MAIN ST",
        "total": "10.00",
    }


def test_string_value_without_tags_is_preserved_under_key() -> None:
    # Guard against over-eager matching: plain-text string leaves must still be
    # stored under their original key, not silently dropped.
    p = _FakeProcessor({"company": "ACME CORP", "total": "10.00"})
    assert _token2json_safe(p, "") == {"company": "ACME CORP", "total": "10.00"}


def test_nested_wrapper_with_list_pages_is_flattened() -> None:
    # CORD-style <sep/> pages inside the outer <s_sroie> wrapper → nested
    # list under a root key. Longest value wins on duplicate keys.
    p = _FakeProcessor({
        "sroie": [
            {"address": "123 MAIN ST", "total": "10.00"},
            {"address": "123 MAIN ST, SPRINGFIELD IL 62701"},
        ],
    })
    merged = _token2json_safe(p, "")
    assert merged["address"] == "123 MAIN ST, SPRINGFIELD IL 62701"
    assert merged["total"] == "10.00"
