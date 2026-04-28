"""test_paper_figure_paths.py — figure paths resolve under the runs layout.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: regression guard for the "paper PDF is empty" class of bugs.
    Before the fix, ``report/sections/results_figures.tex`` hard-coded
    ``\\IfFileExists{../results/fig_X.pdf}`` paths that were authored
    for the legacy flat ``./results/`` layout and silently resolved to
    non-existent paths under the current ``runs/<run_id>/paper/``
    layout, producing a PDF where every Section-V figure was missing.

    These tests lock in two invariants:

    1. No file under ``report/`` references the broken
       ``../results/fig_*.pdf`` pattern any more.
    2. Every ``\\includegraphics`` in ``results_figures.tex`` uses a
       bare basename (resolved through the template's ``\\graphicspath``)
       and is guarded by a matching ``\\iffigurefile`` macro so missing
       figures degrade to empty output without LaTeX errors.
    3. ``report/template.tex`` declares the expected ``\\graphicspath``
       candidates so both the flat-emitter PDFs (``runs/<id>/fig_*.pdf``)
       and the Section-C emitter PDFs (``runs/<id>/figures/fig_*.pdf``)
       are findable from ``runs/<id>/paper/paper_filled.tex``.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "report"
SECTIONS_DIR = REPORT_DIR / "sections"
TEMPLATE = REPORT_DIR / "template.tex"
RESULTS_FIGURES = SECTIONS_DIR / "results_figures.tex"


def test_no_broken_results_prefix_anywhere() -> None:
    """No ``.tex`` in ``report/`` references ``../results/fig_*.pdf``.

    This was the single biggest cause of "empty paper" symptoms under
    the runs/<id>/paper/ layout.  The fix is to use bare basenames and
    let graphicspath resolve them.
    """
    offenders: list[tuple[Path, int, str]] = []
    for tex in REPORT_DIR.rglob("*.tex"):
        for idx, line in enumerate(tex.read_text().splitlines(), start=1):
            if "../results/fig_" in line:
                offenders.append((tex.relative_to(REPO_ROOT), idx, line.strip()))
    assert not offenders, (
        "Found broken ../results/fig_*.pdf references (these fail under "
        "the runs/<id>/paper/ layout):\n"
        + "\n".join(f"  {p}:{n}  {ln}" for p, n, ln in offenders)
    )


def test_results_figures_uses_iffigurefile_only() -> None:
    """Every figure in results_figures.tex is guarded by ``\\iffigurefile``.

    A raw ``\\IfFileExists`` in this file would mean either (a) the
    lookup only consults kpathsea without our multi-candidate fallback,
    or (b) the guard and the ``\\includegraphics`` path can drift apart.
    The repo's convention is ``\\iffigurefile{basename}{body}`` with the
    body's ``\\includegraphics`` using the same bare basename.
    """
    text = RESULTS_FIGURES.read_text()
    assert "\\IfFileExists{" not in text, (
        "results_figures.tex still uses raw \\IfFileExists — replace with "
        "the multi-candidate \\iffigurefile macro defined in template.tex."
    )
    # Every includegraphics must reference a bare basename (no '/').
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        path = m.group(1)
        assert "/" not in path, (
            f"\\includegraphics{{{path}}} in results_figures.tex uses a "
            "slashed path; use the bare basename so graphicspath resolves it."
        )
        assert path.startswith("fig_"), (
            f"unexpected figure name {path!r} — emitters prefix all files "
            "with 'fig_'."
        )


def _extract_balanced(text: str, start: int) -> tuple[str, int]:
    """Return the contents of the ``{...}`` group starting at ``text[start]``.

    ``start`` must index the opening ``{``.  Returns ``(body, end)``
    where ``end`` is the index *after* the matching ``}``.  Supports
    nested braces but not LaTeX-level escape nuances — sufficient for
    the small, well-formed regions we parse here.
    """
    assert text[start] == "{", f"expected '{{' at {start}, got {text[start]!r}"
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError("unbalanced braces")


def test_iffigurefile_and_includegraphics_paired() -> None:
    """For every ``\\iffigurefile{X}{...}`` the body does ``\\includegraphics{X}``.

    Catches the class of bug where the conditional checks one basename
    but the body references a different one.  We parse brace-balanced
    groups because the body contains ``\\begin{figure}`` etc., which
    defeats trivial ``[^{}]*`` regexes.
    """
    text = RESULTS_FIGURES.read_text()
    pairs: list[tuple[str, str]] = []
    idx = 0
    while True:
        # Only match a real macro invocation (``\iffigurefile{``), not the
        # word in prose comments (``Every \iffigurefile expands to ...``).
        m = re.search(r"\\iffigurefile(?=\{)", text[idx:])
        if m is None:
            break
        pos = idx + m.end()
        # First argument: {basename}
        assert text[pos] == "{"
        name, pos = _extract_balanced(text, pos)
        # Second argument: {body...}
        assert text[pos] == "{"
        body, pos = _extract_balanced(text, pos)
        idx = pos
        inc = re.search(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body,
        )
        assert inc is not None, (
            f"\\iffigurefile{{{name}}} body has no \\includegraphics."
        )
        pairs.append((name, inc.group(1)))
    assert pairs, (
        "no \\iffigurefile/\\includegraphics pairs found in "
        "results_figures.tex — template probably regressed."
    )
    for guard_name, body_name in pairs:
        assert guard_name == body_name, (
            f"\\iffigurefile guard {guard_name!r} does not match "
            f"\\includegraphics path {body_name!r}."
        )


def test_template_declares_graphicspath_candidates() -> None:
    """template.tex's ``\\graphicspath`` must cover every live figure dir.

    The figure emitters currently write to two locations:
      * ``runs/<id>/fig_*.pdf``           — flat emitters in report.figures
      * ``runs/<id>/figures/fig_*.pdf``   — Section-C emitters
    When the filled tex lives at ``runs/<id>/paper/paper_filled.tex`` we
    need ``../`` and ``../figures/`` on the search path.  The legacy
    ``./results/`` paths are kept for back-compat with the flat layout.
    """
    text = TEMPLATE.read_text()
    assert "\\graphicspath" in text, (
        "template.tex does not declare \\graphicspath — figure resolution "
        "will fall back to the working directory only."
    )
    # Pull out the \graphicspath{{...}{...}{...}} block.
    m = re.search(r"\\graphicspath\{((?:\{[^}]*\})+)\}", text)
    assert m, "could not parse \\graphicspath block in template.tex"
    candidates = set(re.findall(r"\{([^}]*)\}", m.group(1)))
    for needed in ("./", "./figures/", "../", "../figures/"):
        assert needed in candidates, (
            f"\\graphicspath missing candidate {needed!r}; "
            f"declared: {sorted(candidates)}"
        )


def test_template_iffigurefile_searches_every_candidate() -> None:
    """``\\iffigurefile`` body exhaustively probes every graphicspath dir.

    ``\\IfFileExists`` does not consult ``\\graphicspath``, so the macro
    must check each candidate explicitly or a figure that graphicx
    *would* find still renders an empty float.
    """
    text = TEMPLATE.read_text()
    # Locate ``\newcommand{\iffigurefile}[2]`` and parse the body with
    # brace-balanced matching (trivial regex fails on nested ``{...}``).
    m = re.search(r"\\newcommand\{\\iffigurefile\}\[2\]", text)
    assert m is not None, "template.tex lost the \\iffigurefile macro"
    body, _ = _extract_balanced(text, m.end())
    for needed in ("./#1", "./figures/#1", "../#1", "../figures/#1"):
        assert needed in body, (
            f"\\iffigurefile body does not probe {needed!r}; "
            "a figure that lives there will be silently dropped."
        )
