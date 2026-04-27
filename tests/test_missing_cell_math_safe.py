"""test_missing_cell_math_safe.py — \\MissingCell is wrapped in \\text{} in all templates.

Regression guard: the ``\\MissingCell`` macro previously used
``\\textbf``/``\\texttt``/``\\textsubscript`` directly, which are text-mode-only
commands.  When ``\\MissingCell{key}`` appeared inside a ``$...$`` math context
(e.g. ``$=\\VAR{gtocr_rulebased_f1}$`` in appendix.tex), tectonic aborted with
``Missing $ inserted``.  The fix wraps the macro body in ``\\text{...}``
(amsmath, already loaded in all three templates) so the macro compiles safely
in both text and math mode.
"""
from __future__ import annotations

from pathlib import Path

_REPORT_DIR = Path(__file__).resolve().parents[1] / "report"

_TEMPLATES = [
    _REPORT_DIR / "template.tex",
    _REPORT_DIR / "template_basic.tex",
    _REPORT_DIR / "template_advanced.tex",
]


def _get_missing_cell_region(text: str) -> str | None:
    """Return a few lines around the \\MissingCell definition for inspection."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if r"\newcommand{\MissingCell}" in line:
            # Return lines from the definition start to the closing brace
            start = i
            for j in range(i, min(i + 20, len(lines))):
                if lines[j].strip() == "}":
                    return "\n".join(lines[start : j + 1])
            return "\n".join(lines[start : start + 10])
    return None


def test_missing_cell_uses_text_wrap_in_all_templates() -> None:
    """Each template's ``\\MissingCell`` definition must contain ``\\text{``."""
    for tmpl in _TEMPLATES:
        assert tmpl.exists(), f"Template not found: {tmpl}"
        text = tmpl.read_text()
        region = _get_missing_cell_region(text)
        assert region is not None, f"\\MissingCell not found in {tmpl.name}"
        # Non-comment lines in the body must contain the \text{...} wrap
        content_lines = [
            line for line in region.splitlines()
            if not line.strip().startswith("%") and line.strip()
        ]
        content = "\n".join(content_lines)
        assert "\\text{" in content, (
            f"{tmpl.name}: \\MissingCell body should contain \\text{{...}} "
            f"(math-safe wrap) but got: {content[:160]!r}"
        )


def test_amsmath_loaded_in_all_templates() -> None:
    """amsmath must be loaded in every template (required by \\text{})."""
    import re
    for tmpl in _TEMPLATES:
        text = tmpl.read_text()
        assert re.search(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\bamsmath\b", text), (
            f"{tmpl.name}: amsmath package not loaded — required for \\text{{}} in \\MissingCell"
        )
