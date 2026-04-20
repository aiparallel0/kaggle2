"""Inject real metrics into LaTeX template by replacing \\VAR{name} placeholders.

Also resolves ``\\input{path}`` directives textually before substitution so
\\VAR{} placeholders in section files are also replaced and the filled output
is a single flat ``.tex`` (no extra files needed at compile time).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def _read_section(base: Path, name: str) -> str:
    """Read ``<base>/<name>(.tex)``; raise ``FileNotFoundError`` if missing."""
    # LaTeX \input accepts names with or without the .tex suffix.
    candidates = [base / name, base / f"{name}.tex"]
    for c in candidates:
        if c.is_file():
            return c.read_text()
    raise FileNotFoundError(f"\\input{{{name}}} not found near {base}")


def expand_inputs(template: str, base: Path, max_depth: int = 4) -> str:
    """Recursively inline ``\\input{path}`` directives in *template*.

    Resolves relative paths against *base* (the directory containing the
    top-level ``.tex`` file). Guards against cyclic / runaway inclusions
    with a shallow depth limit — one level of section files is expected.
    """
    if max_depth <= 0:
        return template

    def _replace(match: re.Match[str]) -> str:
        inner = _read_section(base, match.group(1))
        return expand_inputs(inner, base, max_depth - 1)

    return _INPUT_RE.sub(_replace, template)


def inject_results(template: str, metrics: dict[str, Any]) -> str:
    """Replace \\VAR{key} placeholders with formatted metric values.

    Args:
        template: LaTeX source containing \\VAR{key} placeholders.
        metrics: Flat dict of metric name → value.

    Returns:
        LaTeX source with all placeholders replaced. Any \\VAR{} key not
        present in ``metrics`` is replaced by ``---`` so the resulting
        LaTeX always compiles (unresolved macros otherwise produce
        ``Undefined control sequence`` errors in pdflatex/tectonic).
    """
    result = template
    for key, value in metrics.items():
        placeholder = f"\\VAR{{{key}}}"
        if isinstance(value, float):
            result = result.replace(placeholder, f"{value:.4f}")
        else:
            result = result.replace(placeholder, str(value))
    # Backstop: any \VAR{...} that was NOT in the metrics dict becomes ---.
    # Prevents half-rendered \VAR{rulebased_f1_company} tokens leaking into
    # the PDF when a newer results.tex adds placeholders the orchestrator
    # hasn't learned to emit yet.
    result = re.sub(r"\\VAR\{[^}]+\}", "---", result)
    return result
