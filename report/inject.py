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


class TexSource(str):
    """Marker class for pre-escaped LaTeX strings that must NOT be re-escaped.

    Usage:
        metrics["my_latex_key"] = TexSource(r"$x_1 + x_2$")

    When inject_results encounters a TexSource value, it emits the raw string
    instead of running it through _latex_escape_text().  Use sparingly — only
    for values that contain intentional LaTeX math or macros (e.g. the
    \\MissingCell{} marker or pre-built "95\\% CI" strings).
    """

    pass

_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")

_MEAN_STD_KEYS = {"donut_f1", "pipeline_f1"}

# Map each LaTeX text-mode special character to its safe representation.
# Applied only to arbitrary string metric values (e.g. ``test_set_kind =
# "canonical_347"``) — numeric formatters and explicit LaTeX literals such
# as ``\ensuremath{...}`` are already safe and must NOT be passed through
# this escaper (see _format_value).
_LATEX_TEXT_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape_text(s: str) -> str:
    """Escape LaTeX text-mode special characters.

    Used by ``_format_value`` for arbitrary string metric values
    (e.g. ``test_set_kind = "canonical_347"``) which would otherwise
    inject unescaped ``_`` into text mode and crash tectonic with
    ``Missing $ inserted`` (paper_filled.tex line 1468 in run
    20260427T071206Z-fd9d7b0).  Numeric formatters and explicit LaTeX
    literals (``\\ensuremath{...}``, ``\\MissingCell{...}``) are already
    safe and must NOT be passed through this escaper.
    """
    return "".join(_LATEX_TEXT_ESCAPES.get(c, c) for c in s)


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
    """Render learning rate in scientific notation wrapped in ``\\ensuremath{}``.

    ``\\ensuremath{}`` makes the output safe in both text and math contexts:
    it is a no-op inside existing math mode and auto-switches to math mode
    in text mode.  Mirrors the ``\\ensuremath{\\pm}`` fix in inject_format.py.
    """
    if value == 0.0:
        return "0"
    mantissa, exp = f"{value:.0e}".split("e")
    return f"\\ensuremath{{{mantissa}\\times 10^{{{int(exp)}}}}}"


def _format_pvalue(value: float) -> str:
    """Render a p-value without the ``%.4f`` trap that turns ``p=3e-5`` into
    a meaningless ``0.0000``.  Tiny values are printed as ``<10^{-k}`` so
    the McNemar test result is not silently contradicted by the bootstrap
    CI in the discussion (review item S5).

    Returns ``\\ensuremath{...}``-wrapped math content so the output is safe
    in both text and math contexts.  ``\\ensuremath`` is a no-op inside an
    existing math environment and auto-switches to math mode in text mode,
    preventing ``Missing $ inserted`` when the same key appears both inside
    ``$p=\\VAR{mcnemar_p}$`` (math) and in prose ``McNemar $p=$\\VAR{mcnemar_p}``
    (text mode after the closing ``$``).
    """
    if value <= 0.0:
        # Numerical underflow / exact zero — IEEE 754 double precision is
        # ~1e-308; any p-value sufficiently small to round to literal 0.0
        # is far below the 1e-12 threshold typically reported in journals.
        return "\\ensuremath{<10^{-12}}"
    if value < 1e-4:
        # Scientific: e.g. 3.2e-05 → \ensuremath{3.2\times 10^{-5}}.
        mantissa, exp = f"{value:.1e}".split("e")
        return f"\\ensuremath{{{mantissa}\\times 10^{{{int(exp)}}}}}"
    return f"{value:.4f}"


def _has_multi_seed(metrics: dict[str, Any], base_key: str) -> bool:
    seeds = metrics.get("seeds_used")
    return (
        f"{base_key}_std" in metrics
        and isinstance(seeds, list)
        and len(seeds) >= 2
    )


def _format_value(key: str, value: Any, metrics: dict[str, Any]) -> str:
    # Learning-rate keys (``lr``, ``lr_encoder``, ``lr_decoder``) must
    # render in scientific notation; otherwise small LRs like ``5e-5``
    # round to ``0.0001`` under the generic ``{:.4f}`` formatter
    # (paper_corrections.md item 9).
    if (key == "lr" or key.startswith("lr_")) and isinstance(value, int | float):
        return _format_lr(float(value))
    # P-values render with scientific notation below 1e-4 — round-to-4
    # turns ``p=3e-5`` into the misleading ``0.0000`` (review item S5).
    if key == "mcnemar_p" and isinstance(value, int | float):
        return _format_pvalue(float(value))
    if key == "seeds_used" and isinstance(value, list):
        ids = ", ".join(str(s) for s in value)
        return f"{len(value)} seeds ({ids})" if value else "0 seeds"
    if key in _MEAN_STD_KEYS and _has_multi_seed(metrics, key):
        mean = float(metrics[f"{key}_mean"])
        std = float(metrics[f"{key}_std"])
        return f"{mean:.4f} \\ensuremath{{\\pm}} {std:.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, TexSource):
        # Pre-escaped LaTeX: emit raw, do NOT double-escape.
        return str(value)
    if isinstance(value, str):
        return _latex_escape_text(value)
    # Defensive: ints/bools/lists not handled above are unlikely to contain
    # LaTeX-active chars, but escaping uniformly is harmless and guards
    # against future metric types that stringify to e.g. "model_v2".
    return _latex_escape_text(str(value))


def inject_results(template: str, metrics: dict[str, Any]) -> str:
    """Replace \\VAR{key} placeholders with formatted metric values.

    Two-pass: first apply the typed formatter DSL (``:pct1``, ``:ms``,
    ``:usd``, …) via :mod:`report.inject_format`, then the legacy
    plain-``\\VAR{key}`` substitution.  Directives that the formatter
    could not resolve (unknown directive, or missing key) are left
    intact so the plain substitution step either formats or counts
    them in the unresolved-VAR audit.
    """
    from report.inject_format import apply_formatters
    result = apply_formatters(template, metrics)
    for key, value in metrics.items():
        placeholder = f"\\VAR{{{key}}}"
        result = result.replace(placeholder, _format_value(key, value, metrics))
    # Backstop: any \VAR{...} that was NOT in the metrics dict becomes ---.
    # Prevents half-rendered \VAR{rulebased_f1_company} tokens leaking into
    # the PDF when a newer results.tex adds placeholders the orchestrator
    # hasn't learned to emit yet.  Every unresolved key is logged at WARNING
    # so the "no placeholders after a successful run" contract is auditable
    # — see :func:`collect_unresolved` for the JSON side-channel that the
    # paper stage writes to ``metrics/unresolved_vars.json``.
    unresolved = re.findall(r"\\VAR\{([^}]+)\}", result)
    if unresolved:
        import logging

        from report.missing import filter_blockers
        blockers = filter_blockers(unresolved)
        log = logging.getLogger("kaggle2")
        if blockers:
            log.error(
                "inject_results: %d UNRESOLVED \\VAR{} keys are not on the "
                "intentional-missing allow-list — these will render as a "
                "red \\MissingCell{} marker and FAIL `make check_artefacts`. "
                "First 5: %s",
                len(blockers), blockers[:5],
            )
        if len(unresolved) - len(blockers) > 0:
            log.info(
                "inject_results: %d unresolved keys are on the allow-list "
                "(intentionally not measured on this profile).",
                len(unresolved) - len(blockers),
            )
    # Replace each unresolved \VAR{key} with \MissingCell{key} — a typed,
    # red marker that survives compile and is counted by the build gate.
    # The previous "blanket --- em-dash" backstop is gone (silent failure
    # mode that produced the v1–v3 em-dash regression cycle).
    from report.missing import render_missing_cell
    def _to_missing(match: re.Match[str]) -> str:
        return render_missing_cell(match.group(1))
    result = re.sub(r"\\VAR\{([^}]+)\}", _to_missing, result)
    return result


def collect_unresolved(template: str, metrics: dict[str, Any]) -> list[str]:
    """Enumerate every ``\\VAR{key}`` in ``template`` not in ``metrics``.

    Called from :mod:`stages.paper` just before writing the filled
    ``.tex``; the returned list is serialised to
    ``metrics/unresolved_vars.json`` so reviewers can audit which
    placeholders did NOT resolve to a real value.  Empty list on a
    fully-populated run is the "no placeholders" guarantee.

    Directive-format keys (``key:directive``) are intentionally excluded:
    they are resolved by :func:`report.inject_format.apply_formatters`
    which runs before the plain-key substitution, so counting them here
    produces false positives in the unresolved-VAR audit.
    """
    used = set(re.findall(r"\\VAR\{([^}]+)\}", template))
    plain_keys = {k for k in used if ":" not in k}
    return sorted(plain_keys - set(metrics.keys()))
