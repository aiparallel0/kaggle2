"""Inject experiment metrics into LaTeX template via \\VAR{} placeholders.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: replaces \\VAR{key} tokens with formatted metric values from
    combined_metrics.json.  Also resolves \\input{} directives so the
    filled output is a single flat .tex needing no extra files.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")

_MEAN_STD_KEYS = {"donut_f1", "pipeline_f1"}


def _read_section(base: Path, name: str) -> str:
    """Read ``<base>/<name>(.tex)``; raise ``FileNotFoundError`` if missing."""
    # LaTeX \input accepts names with or without the .tex suffix.
    candidates = [base / name, base / f"{name}.tex"]
    for c in candidates:
        if c.is_file():
            return c.read_text()
    raise FileNotFoundError(f"\\input{{{name}}} not found near {base}")


def expand_inputs(template: str, base: Path, max_depth: int = 4) -> str:
    """Recursively inline \\input{path} directives in template."""
    if max_depth <= 0:
        return template

    def _replace(match: re.Match[str]) -> str:
        inner = _read_section(base, match.group(1))
        return expand_inputs(inner, base, max_depth - 1)

    return _INPUT_RE.sub(_replace, template)


def _format_lr(value: float) -> str:
    """Render learning rate in scientific notation so small LRs don't round
    to ``0.0001`` under a generic ``{:.4f}`` formatter (e.g. ``5e-5``)."""
    if value == 0.0:
        return "0"
    mantissa, exp = f"{value:.0e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exp)}}}$"


def _has_multi_seed(metrics: dict[str, Any], base_key: str) -> bool:
    seeds = metrics.get("seeds_used")
    return (
        f"{base_key}_std" in metrics
        and isinstance(seeds, list)
        and len(seeds) >= 2
    )


def _format_value(key: str, value: Any, metrics: dict[str, Any]) -> str:
    if key == "lr" and isinstance(value, int | float):
        return _format_lr(float(value))
    if key == "seeds_used" and isinstance(value, list):
        ids = ", ".join(str(s) for s in value)
        return f"{len(value)} seeds ({ids})" if value else "0 seeds"
    if key in _MEAN_STD_KEYS and _has_multi_seed(metrics, key):
        mean = float(metrics[f"{key}_mean"])
        std = float(metrics[f"{key}_std"])
        return f"{mean:.4f} $\\pm$ {std:.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def inject_results(template: str, metrics: dict[str, Any]) -> str:
    """Replace \\VAR{key} placeholders with formatted metric values."""
    result = template
    for key, value in metrics.items():
        placeholder = f"\\VAR{{{key}}}"
        result = result.replace(placeholder, _format_value(key, value, metrics))
    # Backstop: any \VAR{...} that was NOT in the metrics dict becomes ---.
    # Prevents half-rendered \VAR{rulebased_f1_company} tokens leaking into
    # the PDF when a newer results.tex adds placeholders the orchestrator
    # hasn't learned to emit yet.
    result = re.sub(r"\\VAR\{[^}]+\}", "---", result)
    return result
