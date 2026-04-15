"""Inject real metrics into LaTeX template by replacing \\VAR{name} placeholders."""
from __future__ import annotations

from typing import Any


def inject_results(template: str, metrics: dict[str, Any]) -> str:
    """Replace \\VAR{key} placeholders with formatted metric values.

    Args:
        template: LaTeX source containing \\VAR{key} placeholders.
        metrics: Flat dict of metric name → value.

    Returns:
        LaTeX source with all placeholders replaced.
    """
    result = template
    for key, value in metrics.items():
        placeholder = f"\\VAR{{{key}}}"
        if isinstance(value, float):
            result = result.replace(placeholder, f"{value:.4f}")
        else:
            result = result.replace(placeholder, str(value))
    return result
