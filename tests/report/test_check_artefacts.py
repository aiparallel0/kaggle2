"""Tests for the build-gate ``report.check_artefacts`` regression scanner.

The historical false-positive these tests guard against: the ``\\figref``
macro definition in ``report/template{,_baseline,_focus}.tex`` contains
the literal string ``[figure absent]`` as its undefined-label fallback,
which made the ``figure_absent_markers`` check fire on every successful
build.  See the regression in the user log
(``FAIL — paper artefact has unresolved blockers: figure_absent_markers``).
"""
from __future__ import annotations

from pathlib import Path

from report.check_artefacts import scan_paper

_FIGREF_MACRO = (
    "\\makeatletter\n"
    "\\newcommand{\\figref}[1]{%\n"
    "  \\@ifundefined{r@#1}{\\textit{[figure absent]}}{\\ref{#1}}%\n"
    "}\n"
    "\\makeatother\n"
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "paper_filled.tex"
    p.write_text(body)
    return p


def test_figref_macro_alone_does_not_trigger(tmp_path: Path) -> None:
    """Macro definition is the *only* legitimate occurrence of the literal."""
    paper = _write(tmp_path, _FIGREF_MACRO + "\\begin{document}\nbody\n\\end{document}\n")
    findings = scan_paper(paper)
    assert "figure_absent_markers" not in findings, findings


def test_real_figure_absent_outside_macro_still_flagged(tmp_path: Path) -> None:
    """If something else emits the sentinel into prose, we still catch it."""
    body = (
        _FIGREF_MACRO
        + "\\begin{document}\n"
        + "See Fig. \\textit{[figure absent]} for details.\n"
        + "\\end{document}\n"
    )
    paper = _write(tmp_path, body)
    findings = scan_paper(paper)
    assert findings.get("figure_absent_markers") == ["[figure absent]"]
