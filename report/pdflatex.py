"""Compile .tex to PDF via tectonic (preferred) or pdflatex fallback.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: produces the final IEEE-format PDF from paper_filled.tex.  Gracefully
    falls back to pdflatex and warns (rather than errors) when no LaTeX
    engine is installed.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from core.errors import EvalError

log = logging.getLogger("kaggle2")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _compile_tectonic(tex_path: Path) -> Path:
    result = _run(
        [
            "tectonic", "--keep-intermediates", "--keep-logs",
            "--chatter", "minimal", "--continue-on-errors", tex_path.name,
        ],
        cwd=tex_path.parent,
    )
    if result.returncode != 0:
        raise EvalError(
            f"tectonic failed for {tex_path.name}:\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}",
        )
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.exists():
        raise EvalError(f"tectonic finished but {pdf} was not produced.")
    return pdf


def _compile_pdflatex(tex_path: Path) -> Path:
    stem = tex_path.stem
    cmds: list[list[str]] = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        ["bibtex", stem],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
    ]
    for cmd in cmds:
        # bibtex exits non-zero if there are no citations on the first run;
        # tolerate that specifically, but let genuine pdflatex failures raise.
        result = _run(cmd, cwd=tex_path.parent)
        if result.returncode != 0 and cmd[0] == "pdflatex":
            raise EvalError(
                f"pdflatex failed for {tex_path.name}:\n"
                f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}",
            )
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.exists():
        raise EvalError(f"pdflatex finished but {pdf} was not produced.")
    return pdf


def compile_paper_pdf(tex_path: Path, bib_src: Path) -> Path | None:
    """Compile tex_path to PDF; prefer tectonic, fall back to pdflatex."""
    work = tex_path.parent
    if bib_src.exists() and bib_src.resolve() != (work / bib_src.name).resolve():
        shutil.copy(bib_src, work / bib_src.name)
    if shutil.which("tectonic") is not None:
        return _compile_tectonic(tex_path)
    if shutil.which("pdflatex") is None:
        log.warning(
            "No LaTeX engine found (tried tectonic, pdflatex) — skipping PDF "
            "compilation. Install tectonic (scripts/vastai_bootstrap.sh does "
            "this) to generate %s.pdf.", tex_path.stem,
        )
        return None
    return _compile_pdflatex(tex_path)
