"""Numeric-formatting directives for ``\\VAR{key:directive}`` injection.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: extend :func:`report.inject.inject_results` with a small
    typed DSL so ``\\VAR{donut_f1_company:pct1}`` renders ``84.2\\%``
    and ``\\VAR{lat_p95:ms}`` renders ``142\\,ms``.  The DSL is
    intentionally minimal — every directive maps a float to a
    LaTeX-safe string.  Unknown directives fall through to
    ``{value:.4f}`` which is :func:`report.inject._format_value`'s
    default contract.

Supported directives:
    * ``:pct1`` — percentage with one decimal (``0.842 → 84.2\\%``)
    * ``:pct2`` — percentage with two decimals
    * ``:pct0`` — integer percentage (``0.842 → 84\\%``)
    * ``:ms``   — milliseconds with no decimals (``142.3 → 142\\,ms``)
    * ``:ms1``  — milliseconds with one decimal
    * ``:usd``  — USD amount (``1.234 → \\$1.23``)
    * ``:usd4`` — USD amount with four decimals (for per-image)
    * ``:wh``   — watt-hours integer
    * ``:gb1``  — GiB with one decimal (``12.34 → 12.3\\,GiB``)
    * ``:sig3`` — 3 significant figures
    * ``:sig4`` — 4 significant figures
    * ``:int``  — integer with thousands separator
    * ``:bits`` — bits with one decimal
    * ``:p``    — p-value for prose/text-mode contexts.  Renders large
      values as ``0.0432`` and small values (``< 1e-4``) as
      ``$3.2\\times 10^{-5}$`` (with its own ``$...$`` math wrap so the
      directive is safe in text mode such as ``$p=$\\VAR{mcnemar_p:p}``).
      Distinct from the bare ``\\VAR{mcnemar_p}`` renderer in
      :func:`report.inject._format_pvalue`, which assumes the caller
      already opened math mode (e.g. ``$p=\\VAR{mcnemar_p}$`` in
      ``results.tex``) and therefore returns *raw* math content.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("kaggle2")

# Match ``\VAR{key:directive}`` where the directive name can contain
# lowercase letters, digits, and underscores.  Underscores were added
# in v4 so composite directives like ``mean_std_pct1`` parse —
# previously the regex was ``[a-z0-9]+`` which silently fell through to
# the unresolved-VAR audit on any directive containing ``_``.
_VAR_RE = re.compile(r"\\VAR\{([A-Za-z_][A-Za-z0-9_]*):([a-z0-9_]+)\}")


def _to_float(value: object) -> float | None:
    """Cast ``value`` to float; return None if impossible."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def apply_directive(value: object, directive: str) -> str | None:
    """Render ``value`` per ``directive``; return None for unknown directives."""
    fv = _to_float(value)
    if fv is None:
        return None
    if directive == "pct1":
        return f"{fv * 100.0:.1f}\\%"
    if directive == "pct2":
        return f"{fv * 100.0:.2f}\\%"
    if directive == "pct0":
        return f"{fv * 100.0:.0f}\\%"
    if directive == "ms":
        return f"{fv:.0f}\\,ms"
    if directive == "ms1":
        return f"{fv:.1f}\\,ms"
    if directive == "usd":
        return f"\\${fv:.2f}"
    if directive == "usd4":
        return f"\\${fv:.4f}"
    if directive == "wh":
        return f"{fv:.0f}\\,Wh"
    if directive == "gb1":
        return f"{fv:.1f}\\,GiB"
    if directive == "bits":
        return f"{fv:.2f}\\,bits"
    if directive == "sig3":
        return _sig_fig(fv, 3)
    if directive == "sig4":
        return _sig_fig(fv, 4)
    if directive == "int":
        return f"{int(round(fv)):,}"
    if directive == "p":
        # Text-mode-safe p-value formatter.  Template usage is
        # ``$p=$\VAR{mcnemar_p:p}`` (math mode CLOSED before VAR), so
        # this directive must NOT emit raw math content like ``\times``
        # without its own ``$...$`` wrap — that would trigger
        # "Missing $ inserted" in tectonic/pdflatex.
        if fv <= 0.0:
            # Numerical underflow; IEEE 754 doubles bottom out near 1e-308
            # so any p that rounds to literal 0.0 is well below 1e-12.
            return "$<10^{-12}$"
        if fv < 1e-4:
            mantissa, exp = f"{fv:.1e}".split("e")
            return f"${mantissa}\\times 10^{{{int(exp)}}}$"
        return f"{fv:.4f}"
    return None


def _sig_fig(x: float, digits: int) -> str:
    """Format ``x`` with ``digits`` significant figures (no scientific unless needed)."""
    if x == 0.0:
        return "0"
    from math import floor, log10
    mag = 10.0 ** (digits - 1 - int(floor(log10(abs(x)))))
    rounded = round(x * mag) / mag
    if abs(rounded) >= 1e4 or abs(rounded) < 1e-3:
        return f"{rounded:.{digits - 1}e}"
    # Trim trailing zeros but keep at least ``digits`` significant digits.
    return f"{rounded:.{max(0, digits - 1 - int(floor(log10(abs(rounded)))))}f}"


def apply_formatters(template: str, metrics: dict[str, object]) -> str:
    """Resolve every ``\\VAR{key:directive}`` in ``template``.

    Unknown directives pass through to the default injector by leaving
    the placeholder intact, then :func:`report.inject.inject_results`
    replaces them as regular ``\\VAR{key}`` (ignoring the directive).
    Missing keys are left intact too so the unresolved-VAR audit in
    :func:`report.inject.collect_unresolved` can count them.

    v4 — adds ``mean_std_<base>`` directives (``mean_std_pct1``,
    ``mean_std_pct2``, ``mean_std_sig3``) which read ``<key>_mean``
    and ``<key>_std`` and render ``85.2 \\pm 0.7\\%`` for multi-seed
    runs.  When ``<key>_std`` is absent or zero (single-seed run),
    only the mean is rendered so n=1 builds remain readable.
    """
    def _replace(match: re.Match[str]) -> str:
        key, directive = match.group(1), match.group(2)
        # v4 — mean ± std composite directive: ``mean_std_<base>`` reads
        # ``key_mean`` + ``key_std`` from metrics directly.
        if directive.startswith("mean_std_"):
            base = directive[len("mean_std_"):]
            mean_key, std_key = f"{key}_mean", f"{key}_std"
            if mean_key not in metrics:
                return match.group(0)  # let unresolved-audit flag it
            mean_str = apply_directive(metrics[mean_key], base)
            if mean_str is None:
                return f"\\VAR{{{key}_mean}}"
            std_val = metrics.get(std_key)
            std_f = std_val if isinstance(std_val, int | float) else None
            if std_f is None or std_f <= 0.0:
                # Single-seed run or zero spread — just emit the mean.
                return mean_str
            std_str = apply_directive(std_val, base)
            if std_str is None:
                return mean_str
            return f"{mean_str} \\pm {std_str}"
        if key not in metrics:
            # Leave intact; the base injector's audit will flag it.
            return match.group(0)
        rendered = apply_directive(metrics[key], directive)
        if rendered is None:
            log.warning(
                "inject_format: unknown directive ':%s' on key '%s' "
                "— falling back to default formatter.",
                directive, key,
            )
            # Strip directive so the base injector formats the raw value.
            return f"\\VAR{{{key}}}"
        return rendered

    return _VAR_RE.sub(_replace, template)
