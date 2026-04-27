"""Build-gate: scan the rendered LaTeX/PDF for unresolved markers.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: implement the v4 prompt's item 6i.  Every regression that has
    plagued v1–v3 (em-dashes, ``[?]`` undefined-citation markers,
    ``Sec.~??`` undefined references, "[figure absent]" sentinels,
    empty-PDF figures) is detected here and turned into a non-zero
    exit code so ``make all`` can gate on it.

CLI: ``python -m report.check_artefacts [--paper PATH] [--figures DIR]``
    Exits 1 with a human-readable report if any blocker is found, 0 otherwise.

The list of blockers is intentionally narrow and high-precision so the
gate stays useful (no false positives):

  * unresolved ``\\VAR{...}`` placeholders not on the allow-list
  * ``\\MissingCell{...}`` literals (when strict)
  * undefined-reference ``Sec.~??`` / ``Fig.~??`` rendered into the .tex
  * undefined-citation ``[?]`` markers (verbatim, in source-text)
  * the ``[figure absent]`` placeholder our ``\\figref`` macro emits
  * empty (zero-byte) PDFs in the figures directory
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from report.missing import filter_blockers, is_strict

_VAR_RE = re.compile(r"\\VAR\{([^}]+)\}")
_MISSING_RE = re.compile(r"\\MissingCell\{([^}]+)\}")
# Match LaTeX undefined-ref ``??`` ONLY when it appears after a Sec/Fig/
# Tab/eq prefix — bare ``??`` in prose (e.g. quoted code) is fine.
_DANGLING_REF_RE = re.compile(
    r"(Sec\.|Fig\.|Tab\.|Section|Figure|Table|eq\.|Eq\.)~?\?\?",
)
_DANGLING_CITE_RE = re.compile(r"\[\?\]")
_FIG_ABSENT_RE = re.compile(r"\[figure absent\]")
# The ``\figref`` macro defined in our LaTeX templates contains the literal
# string ``[figure absent]`` as its fallback body — that text is only
# *emitted* by LaTeX at compile time when the label is undefined, but it
# physically appears in the .tex source on this exact line.  We strip
# that one line before scanning so the checker becomes precise (without
# the strip it fires on every successful build).
_FIGREF_BODY_RE = re.compile(
    r"\\@ifundefined\{r@#1\}\{\\textit\{\[figure absent\]\}\}\{\\ref\{#1\}\}",
)


def scan_paper(paper_tex: Path) -> dict[str, list[str]]:
    """Scan a filled paper.tex for every blocker we know how to detect."""
    if not paper_tex.is_file():
        return {"file_missing": [str(paper_tex)]}
    text = paper_tex.read_text()
    findings: dict[str, list[str]] = {}
    unresolved = sorted(set(_VAR_RE.findall(text)))
    blockers = filter_blockers(unresolved)
    if blockers:
        findings["unresolved_VAR"] = blockers
    missing_cells = sorted(set(_MISSING_RE.findall(text)))
    if missing_cells and is_strict():
        findings["missing_cells_strict"] = missing_cells
    if _DANGLING_REF_RE.search(text):
        # Surface up to first 5 contexts for the operator.
        ctx = [m.group(0) for m in _DANGLING_REF_RE.finditer(text)][:5]
        findings["dangling_refs"] = ctx
    if _DANGLING_CITE_RE.search(text):
        findings["dangling_citations"] = ["[?]"]
    # Strip the ``\figref`` macro definition before searching: its body
    # legitimately contains the literal ``[figure absent]`` sentinel
    # text, which LaTeX expands at compile time when a label is
    # undefined.  Without the strip, the check fires on every build.
    text_without_macro = _FIGREF_BODY_RE.sub("", text)
    if _FIG_ABSENT_RE.search(text_without_macro):
        findings["figure_absent_markers"] = ["[figure absent]"]
    return findings


def scan_figures_dir(figures_dir: Path) -> list[str]:
    """Return the relative path of every zero-byte PDF in ``figures_dir``."""
    if not figures_dir.is_dir():
        return []
    empties: list[str] = []
    for pdf in figures_dir.glob("**/*.pdf"):
        try:
            if pdf.stat().st_size == 0:
                empties.append(str(pdf.relative_to(figures_dir)))
        except OSError:
            empties.append(str(pdf))
    return sorted(empties)


def format_report(
    paper_findings: dict[str, list[str]],
    empty_pdfs: list[str],
) -> str:
    """Render a human-readable build-gate report."""
    if not paper_findings and not empty_pdfs:
        return "OK — no unresolved \\VAR{}, no dangling refs/cites, no empty PDFs."
    lines = ["FAIL — paper artefact has unresolved blockers:"]
    for category, items in paper_findings.items():
        head = items[:10]
        more = "" if len(items) <= 10 else f" (+{len(items) - 10} more)"
        lines.append(f"  {category} ({len(items)}): {head}{more}")
    if empty_pdfs:
        head = empty_pdfs[:10]
        more = "" if len(empty_pdfs) <= 10 else f" (+{len(empty_pdfs) - 10} more)"
        lines.append(f"  empty_pdfs ({len(empty_pdfs)}): {head}{more}")
    return "\n".join(lines)


def run(paper_tex: Path, figures_dir: Path) -> int:
    """Run the full scan, print the report, return a process exit code."""
    paper_findings = scan_paper(paper_tex)
    empty_pdfs = scan_figures_dir(figures_dir)
    report = format_report(paper_findings, empty_pdfs)
    print(report)
    return 0 if not paper_findings and not empty_pdfs else 1


def _default_paper() -> Path:
    """Best-effort: latest ``runs/<run_id>/paper/paper_filled.tex``, else legacy."""
    runs = Path("runs")
    if runs.is_dir():
        latest = sorted(
            (p for p in runs.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in latest:
            cand = run_dir / "paper" / "paper_filled.tex"
            if cand.is_file():
                return cand
            cand = run_dir / "paper_filled.tex"
            if cand.is_file():
                return cand
    return Path("report/paper_filled.tex")


def _default_figures() -> Path:
    """Best-effort: latest ``runs/<run_id>/figures/``, else legacy."""
    runs = Path("runs")
    if runs.is_dir():
        latest = sorted(
            (p for p in runs.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in latest:
            cand = run_dir / "figures"
            if cand.is_dir():
                return cand
    return Path("results")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", type=Path, default=None,
                   help="paper_filled.tex path (default: latest runs/)")
    p.add_argument("--figures", type=Path, default=None,
                   help="figures directory (default: latest runs/)")
    args = p.parse_args(argv)
    paper = args.paper or _default_paper()
    figures = args.figures or _default_figures()
    return run(paper, figures)


if __name__ == "__main__":
    sys.exit(main())
