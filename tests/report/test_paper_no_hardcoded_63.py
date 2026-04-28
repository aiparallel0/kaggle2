"""Regression: advanced-variant template + sections must NOT hard-code
the basic-split test-set size.

Audit C (PR #116): the advanced paper variant evaluates on the
canonical ICDAR-2019 SROIE Task-3 test set (347 images), but several
captions, the abstract, and the appendix used to read ``63-image
test split`` / ``500/63/63 train/val/test split``
literally --- internally inconsistent with ``test_set_size = 347``
written to ``combined_metrics.json`` for the very same run.

The fix replaces those literals with ``\\VAR{n_test_images}`` (and
companions for train/val) so the rendered PDF matches the actual
split.  This test pins the contract by scanning the advanced template
plus the sections it ``\\input{}``s and asserting no banned literal
remains in non-comment, non-``_basic`` content.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO / "report" / "template.tex"
_SECTIONS = _REPO / "report" / "sections"

# Sections \input{}-ed by the advanced template.tex (mirrors lines
# 147-161 of template.tex).  results_figures.tex is included via
# results.tex.  ``_basic`` and ``competitors`` sections are NOT
# included by the advanced template and therefore out of scope.
_ADVANCED_SECTIONS = (
    "intro.tex", "related.tex", "problem.tex", "method.tex",
    "bugs.tex", "experiments.tex", "results.tex", "results_tables.tex",
    "results_figures.tex", "results_gallery.tex", "discussion.tex",
    "limitations.tex", "broader_impact.tex", "conclusion.tex",
    "appendix.tex", "appendix_env.tex",
)

# Patterns forbidden in the advanced variant.  The ``500/63/63``
# triple is also forbidden — it is the basic split's signature.
_BANNED = (
    re.compile(r"\b63-image\b"),
    re.compile(r"\b63\\,images?\b"),
    re.compile(r"\b63\s+receipts\b"),
    re.compile(r"500\\,/\\,63\\,/\\,63"),
    re.compile(r"\b500/63/63\b"),
)


def _strip_comments(text: str) -> str:
    """Strip LaTeX line comments (``%`` to EOL); preserve ``\\%``."""
    out: list[str] = []
    for line in text.splitlines():
        # Find first unescaped '%'.
        idx = 0
        while True:
            j = line.find("%", idx)
            if j < 0:
                break
            if j > 0 and line[j - 1] == "\\":
                idx = j + 1
                continue
            line = line[:j]
            break
        out.append(line)
    return "\n".join(out)


def _scan(path: Path) -> list[str]:
    text = _strip_comments(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for pat in _BANNED:
        for m in pat.finditer(text):
            # Compute 1-based line number.
            line_no = text[: m.start()].count("\n") + 1
            hits.append(f"{path.name}:{line_no}: {m.group(0)!r}")
    return hits


def test_advanced_template_no_hardcoded_basic_split_literals() -> None:
    hits = _scan(_TEMPLATE)
    for sec in _ADVANCED_SECTIONS:
        sec_path = _SECTIONS / sec
        if sec_path.exists():
            # ``limitations.tex`` legitimately discusses the basic
            # variant (``the \texttt{basic} variant evaluates on a
            # 500\,/\,63\,/\,63 internal split…``) — that paragraph
            # is descriptive, not a claim about THIS run.  Allow it.
            if sec == "limitations.tex":
                continue
            hits.extend(_scan(sec_path))
    assert hits == [], (
        "advanced-variant tex must use \\VAR{n_*_images} instead of "
        "hard-coded basic-split literals; offending lines:\n  "
        + "\n  ".join(hits)
    )


def test_advanced_template_uses_n_test_images_var() -> None:
    """At least one caption in results.tex now references the var."""
    text = (_SECTIONS / "results.tex").read_text(encoding="utf-8")
    assert "\\VAR{n_test_images}" in text


def test_splits_table_is_parametric() -> None:
    """``tab:splits`` must read its row counts from \\VAR{} keys."""
    text = (_SECTIONS / "experiments.tex").read_text(encoding="utf-8")
    # Body cells must be VAR-driven.
    assert "\\VAR{n_train_images}" in text
    assert "\\VAR{n_val_images}" in text
    assert "\\VAR{n_test_images}" in text
    assert "\\VAR{n_train_fields}" in text
    assert "\\VAR{n_val_fields}" in text
    assert "\\VAR{n_test_fields}" in text
    # The static "Test  &  63 &  252" body row must be gone.
    assert re.search(r"Test\s*&\s*63\s*&\s*252", text) is None
