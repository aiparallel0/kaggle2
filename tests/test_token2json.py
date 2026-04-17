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
