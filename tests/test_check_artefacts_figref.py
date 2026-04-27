"""test_check_artefacts_figref.py — \\figref macro definition must not cause false-positives.

Regression guard: the ``\\figref`` macro defined in ``report/template{,_basic,_advanced}.tex``
contains the literal string ``[figure absent]`` as its undefined-label fallback.
The ``_FIGREF_BODY_RE`` regex in ``report.check_artefacts`` strips this definition
before scanning.  If the regex is too rigid (whitespace-sensitive), the strip fails
and ``scan_paper`` fires a false-positive ``figure_absent_markers`` finding on every
build — even ones where no figure is actually absent.

This module uses the verbatim ``\\figref`` definition from ``report/template.tex``
to guard against any future template refactoring that could break the strip.
"""
from __future__ import annotations

from pathlib import Path

from report.check_artefacts import scan_paper

_TEMPLATE_TEX = Path(__file__).resolve().parents[1] / "report" / "template.tex"
_TEMPLATE_BASIC = Path(__file__).resolve().parents[1] / "report" / "template_basic.tex"
_TEMPLATE_ADVANCED = Path(__file__).resolve().parents[1] / "report" / "template_advanced.tex"


def _extract_figref_block(template: Path) -> str:
    """Extract the \\makeatletter...\\makeatother block containing \\figref."""
    import re
    text = template.read_text()
    m = re.search(r"\\makeatletter.*?\\makeatother", text, re.DOTALL)
    assert m is not None, f"\\makeatletter block not found in {template.name}"
    return m.group(0)


def _write_paper(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "paper_filled.tex"
    p.write_text(body)
    return p


def test_figref_template_tex_alone_does_not_trigger(tmp_path: Path) -> None:
    """Verbatim \\figref from template.tex must not cause figure_absent_markers."""
    figref_block = _extract_figref_block(_TEMPLATE_TEX)
    paper = _write_paper(
        tmp_path,
        figref_block + "\n\\begin{document}\nSome prose.\n\\end{document}\n",
    )
    findings = scan_paper(paper)
    assert "figure_absent_markers" not in findings, (
        f"False positive from template.tex \\figref macro: {findings}"
    )


def test_figref_template_basic_alone_does_not_trigger(tmp_path: Path) -> None:
    """Verbatim \\figref from template_basic.tex must not cause figure_absent_markers."""
    figref_block = _extract_figref_block(_TEMPLATE_BASIC)
    paper = _write_paper(
        tmp_path,
        figref_block + "\n\\begin{document}\nBody.\n\\end{document}\n",
    )
    findings = scan_paper(paper)
    assert "figure_absent_markers" not in findings, (
        f"False positive from template_basic.tex \\figref macro: {findings}"
    )


def test_figref_template_advanced_alone_does_not_trigger(tmp_path: Path) -> None:
    """Verbatim \\figref from template_advanced.tex must not cause figure_absent_markers."""
    figref_block = _extract_figref_block(_TEMPLATE_ADVANCED)
    paper = _write_paper(
        tmp_path,
        figref_block + "\n\\begin{document}\nBody.\n\\end{document}\n",
    )
    findings = scan_paper(paper)
    assert "figure_absent_markers" not in findings, (
        f"False positive from template_advanced.tex \\figref macro: {findings}"
    )


def test_real_figure_absent_after_macro_still_flagged(tmp_path: Path) -> None:
    """An actual ``[figure absent]`` in prose (outside the macro) must still be flagged."""
    figref_block = _extract_figref_block(_TEMPLATE_TEX)
    paper = _write_paper(
        tmp_path,
        figref_block + "\n\\begin{document}\n"
        "See Figure \\textit{[figure absent]} for details.\n"
        "\\end{document}\n",
    )
    findings = scan_paper(paper)
    assert findings.get("figure_absent_markers") == ["[figure absent]"], (
        f"Expected figure_absent_markers to be flagged but got: {findings}"
    )
