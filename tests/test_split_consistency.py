"""Tests for Item 3: basic split literals forbidden in advanced sections."""
from pathlib import Path

import pytest

from report.missing import (
    _BASIC_SPLIT_PATTERNS,
    assert_no_basic_split_in_advanced_sections,
)


def test_patterns_are_defined() -> None:
    """Sanity: patterns tuple is non-empty."""
    assert len(_BASIC_SPLIT_PATTERNS) >= 3


def test_assertion_passes_on_clean_directory(tmp_path: Path) -> None:
    """Directory with no advanced files should pass."""
    (tmp_path / "intro.tex").write_text("Hello world\n")
    assert_no_basic_split_in_advanced_sections(str(tmp_path))  # no raise


def test_assertion_passes_when_advanced_clean(tmp_path: Path) -> None:
    """Advanced file without basic split literals passes."""
    (tmp_path / "results_advanced.tex").write_text("347-image test set\n")
    assert_no_basic_split_in_advanced_sections(str(tmp_path))


def test_assertion_raises_on_500_63_63(tmp_path: Path) -> None:
    """Advanced file with '500/63/63' should raise."""
    (tmp_path / "results_advanced.tex").write_text("Using 500/63/63 split\n")
    with pytest.raises(ValueError, match="500/63/63"):
        assert_no_basic_split_in_advanced_sections(str(tmp_path))


def test_assertion_raises_on_63_image_test(tmp_path: Path) -> None:
    """Advanced file with '63-image test' should raise."""
    (tmp_path / "conclusion_advanced.tex").write_text("On the 63-image test set\n")
    with pytest.raises(ValueError, match="63-image test"):
        assert_no_basic_split_in_advanced_sections(str(tmp_path))


def test_assertion_ignores_comments(tmp_path: Path) -> None:
    """LaTeX comments should be ignored."""
    content = "% Example: uses 500/63/63 for basic\nReal content here\n"
    (tmp_path / "results_advanced.tex").write_text(content)
    assert_no_basic_split_in_advanced_sections(str(tmp_path))  # no raise


def test_real_advanced_sections_clean() -> None:
    """Actual report/sections/ should not contain basic split in advanced files."""
    sections_dir = Path(__file__).parent.parent / "report" / "sections"
    assert_no_basic_split_in_advanced_sections(str(sections_dir))
