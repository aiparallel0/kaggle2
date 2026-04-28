"""Audit C1 regression: assigner-architecture literals are gone from sections.

Greps every ``report/sections/*.tex`` for the historical assigner
literals (``1157``, ``161``, ``192``, ``128``, ``L=2``, ``L=3``,
``d=128``, ``d=192``).  Inside-comment matches are tolerated since
they describe historical drift; in-body matches fail the test.

The single source of truth is now ``\\VAR{assigner_d_model}`` /
``\\VAR{assigner_n_layers}`` / ``\\VAR{assigner_params_k}`` —
populated by ``report.combine.merge_pipeline_diagnostics`` /
``merge_assigner_metrics`` from the live config + checkpoint.
"""
from __future__ import annotations

import re
from pathlib import Path

_SECTIONS = Path(__file__).resolve().parents[2] / "report" / "sections"
_PATTERNS = (
    r"\b1157\b",
    r"\b161\b",
    r"\b192\b",
    r"\b128\b",
    r"d\s*=\s*128",
    r"d\s*=\s*192",
    r"L\s*=\s*2",
    r"L\s*=\s*3",
    r"d\\\{=\\\}128",
    r"d\\\{=\\\}192",
    r"L\\\{=\\\}2",
    r"L\\\{=\\\}3",
)


def _strip_comments(text: str) -> str:
    """Drop LaTeX line-comments (``%`` to end-of-line, accounting for ``\\%``)."""
    out: list[str] = []
    for line in text.splitlines():
        # find first un-escaped %
        i = 0
        cleaned = ""
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                cleaned += line[i:i + 2]
                i += 2
                continue
            if line[i] == "%":
                break
            cleaned += line[i]
            i += 1
        out.append(cleaned)
    return "\n".join(out)


def test_no_arch_literals_in_section_bodies() -> None:
    """No assigner-architecture literal appears outside comments in any .tex."""
    offenders: list[str] = []
    for path in _SECTIONS.glob("*.tex"):
        body = _strip_comments(path.read_text(encoding="utf-8"))
        for pat in _PATTERNS:
            for m in re.finditer(pat, body):
                offenders.append(f"{path.name}: {pat!r} at offset {m.start()}: {body[max(0, m.start() - 20):m.start() + 30]!r}")
    assert offenders == [], (
        "assigner-architecture literals must be sourced from "
        "\\VAR{assigner_d_model}/\\VAR{assigner_n_layers}/\\VAR{assigner_params_k}; "
        f"offenders: {offenders[:5]}"
    )
