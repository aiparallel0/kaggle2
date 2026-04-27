r"""Tests for the cosmetic fix pass (paper_corrections items).

These tests verify the following fixes:
  * Item 1: TexSource bypasses _latex_escape_text (no double-escape)
  * Item 10: all_off keys are on the MISSING_OK_KEYS allow-list
  * Item 11: Algorithm 1 notation uses T_f consistently
  * Item 14: \code{} macro is defined in template
"""
from __future__ import annotations

from pathlib import Path


def test_texsource_bypasses_escape() -> None:
    """TexSource-marked values are emitted raw, not double-escaped."""
    from report.inject import TexSource, _format_value

    # A plain string containing underscores gets escaped
    plain = _format_value("test_key", "foo_bar", {})
    assert r"\_" in plain, "plain strings should escape underscores"

    # A TexSource string passes through unchanged
    raw_latex = TexSource(r"$x_1 + x_2$")
    result = _format_value("test_key", raw_latex, {})
    assert result == r"$x_1 + x_2$", "TexSource should not be escaped"


def test_texsource_is_a_string() -> None:
    """TexSource inherits from str so it can be used anywhere strings work."""
    from report.inject import TexSource

    ts = TexSource("hello")
    assert isinstance(ts, str)
    assert len(ts) == 5
    assert ts.upper() == "HELLO"


def test_all_off_keys_on_allow_list() -> None:
    """Item 10: all_off keys removed from table, added to MISSING_OK_KEYS."""
    from report.missing import MISSING_OK_KEYS, is_missing_ok

    # all_off row was removed from bugs.tex; keys must be allow-listed
    for key in ("all_off_delta", "all_off_ci_low", "all_off_ci_high"):
        assert key in MISSING_OK_KEYS, f"{key} should be in MISSING_OK_KEYS"
        assert is_missing_ok(key), f"is_missing_ok({key!r}) should be True"


def test_algorithm1_uses_T_f_notation() -> None:
    """Item 11: Algorithm 1 and the equation above it both use T_f notation."""
    section = Path("report/sections/method_pipeline.tex").read_text()

    # The equation should use T_f not \mathrm{target}_f
    assert r"\mathrm{target}_f" not in section, (
        "equation should use T_f not \\mathrm{target}_f"
    )
    assert r"\in T_f" in section, "equation should reference T_f"

    # Algorithm 1 should define {T_f} as target sets
    assert r"\{T_f\}" in section, "Algorithm 1 should define {T_f}"


def test_code_macro_defined_in_template() -> None:
    """Item 14: \\code{} macro is defined in template_advanced.tex."""
    template = Path("report/template_advanced.tex").read_text()
    assert r"\newcommand{\code}" in template, "\\code{} macro should be defined"
    assert "hyphenchar" in template, "\\code{} should use hyphenation fix"


def test_bugs_table_no_all_off_row() -> None:
    """Item 10: bugs.tex table should not have the all_off row."""
    section = Path("report/sections/bugs.tex").read_text()
    assert r"all\_off" not in section, "all_off row should be removed"
    # The ceiling baseline terminology should be used
    assert "Ceiling" in section or "ceiling" in section, (
        "caption should reference ceiling baseline"
    )


def test_figure_emitters_use_logging_not_warnings() -> None:
    """Suppress warnings/INFO spam: figure emitters use log.debug."""
    bugs_py = Path("report/figures_bugs.py").read_text()
    common_py = Path("report/figures_common.py").read_text()

    # figures_bugs.py should not use warnings.warn
    assert "import warnings" not in bugs_py, (
        "figures_bugs.py should not import warnings"
    )
    assert "warnings.warn" not in bugs_py, (
        "figures_bugs.py should not use warnings.warn"
    )

    # figures_common.py guard_empty should use log.debug not log.info
    assert "log.debug" in common_py, "guard_empty should use log.debug"
    # Check that guard_empty specifically uses log.debug (not log.info)
    assert 'log.info("figures_common: skipping' not in common_py, (
        "guard_empty should use log.debug not log.info"
    )
