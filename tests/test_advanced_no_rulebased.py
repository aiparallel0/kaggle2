"""Item 2: advanced variant must not contain rulebased/gtocr_rulebased keys.

Tests verify:
  * assert_no_rulebased_in_advanced raises on bad keys for variant==advanced
  * assert_no_rulebased_in_advanced passes for variant==basic with same keys
  * Template source check: advanced sections have no raw rulebased \\VAR keys
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_assertion_raises_on_rulebased_keys_advanced() -> None:
    """advanced variant must reject rulebased/gtocr_rulebased keys."""
    from report.missing import assert_no_rulebased_in_advanced

    metrics = {"donut_f1": 0.9, "gtocr_rulebased_f1": 0.5, "rulebased_ned": 0.6}
    with pytest.raises(ValueError, match="rulebased"):
        assert_no_rulebased_in_advanced(metrics, "advanced")


def test_assertion_passes_for_basic_variant() -> None:
    """basic variant may have rulebased keys."""
    from report.missing import assert_no_rulebased_in_advanced

    metrics = {"donut_f1": 0.9, "gtocr_rulebased_f1": 0.5}
    # Should not raise
    assert_no_rulebased_in_advanced(metrics, "basic")


def test_assertion_passes_when_no_rulebased_keys() -> None:
    """advanced variant passes when no rulebased keys present."""
    from report.missing import assert_no_rulebased_in_advanced

    metrics = {"donut_f1": 0.9, "pipeline_f1": 0.88}
    # Should not raise
    assert_no_rulebased_in_advanced(metrics, "advanced")


def test_advanced_sections_no_raw_rulebased_vars() -> None:
    """Advanced-specific .tex files should not have rulebased VAR keys in code.

    Comments are excluded from the check (lines starting with %).
    """
    import re

    advanced_files = [
        Path("report/template_advanced.tex"),
        Path("report/sections/results_tables_advanced.tex"),
        Path("report/sections/conclusion_advanced.tex"),
    ]
    var_re = re.compile(r"\\VAR\{([^}]+)\}")
    bad_prefixes = ("gtocr_rulebased_", "rulebased_")

    for path in advanced_files:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            # Skip comment lines
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue
            for match in var_re.finditer(line):
                key = match.group(1).split(":")[0]  # strip directive
                for prefix in bad_prefixes:
                    assert not key.startswith(prefix), (
                        f"{path.name}:{lineno} contains rulebased VAR key: {key}"
                    )
