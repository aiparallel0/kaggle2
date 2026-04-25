"""Tests for :func:`report.inject.collect_unresolved`."""
from __future__ import annotations

from report.inject import collect_unresolved, inject_results


def test_collect_unresolved_reports_missing_keys() -> None:
    template = "F1 = \\VAR{f1}, latency = \\VAR{ms}"
    unresolved = collect_unresolved(template, {"f1": 0.8})
    assert unresolved == ["ms"]


def test_collect_unresolved_empty_on_full_resolution() -> None:
    template = "F1 = \\VAR{f1}"
    assert collect_unresolved(template, {"f1": 0.8}) == []


def test_collect_unresolved_dedupes() -> None:
    """Repeated \\VAR{missing} counts once."""
    template = "\\VAR{missing} and \\VAR{missing} and \\VAR{missing}"
    assert collect_unresolved(template, {}) == ["missing"]


def test_collect_unresolved_sorted_output() -> None:
    template = "\\VAR{zebra} \\VAR{apple} \\VAR{mango}"
    assert collect_unresolved(template, {}) == ["apple", "mango", "zebra"]


def test_inject_results_keeps_no_raw_vars() -> None:
    """Unresolved keys must not leak to the PDF as ``\\VAR{...}``."""
    template = "resolved=\\VAR{a}, missing=\\VAR{b}"
    out = inject_results(template, {"a": "X"})
    assert "\\VAR" not in out
    # v4 contract: missing keys render as typed \MissingCell{key},
    # not silent ``---`` em-dashes.  ``check_artefacts`` then flags
    # them at build time.
    assert "---" not in out
    assert "\\MissingCell{b}" in out
    assert "X" in out


def test_inject_results_two_pass_with_formatters() -> None:
    """Directive-keys resolve to formatted values."""
    template = "pct=\\VAR{f:pct1} raw=\\VAR{f}"
    out = inject_results(template, {"f": 0.842})
    assert "84.2\\%" in out
    # Raw also renders (default float format).
    assert "\\VAR" not in out
