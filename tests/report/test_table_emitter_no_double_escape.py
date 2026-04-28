"""Regression: ``inject_tables`` outputs must NOT be re-escaped by
:func:`report.inject._format_value`.

Audit A (PR #114): six tables (tab:headline_f1, tab:extended,
tab:latency, tab:training, tab:competitors, tab:env) rendered as
literal LaTeX-escaped string literals in ``paper_filled.pdf``
(``\\textbackslash{}begin\\{tabular\\}…``) because the emitter output
went through the same string-escape path as scalar ``\\VAR{}`` values.

Fix: every emitted block is wrapped in :class:`report.inject.TexSource`
so ``_format_value`` emits the raw LaTeX verbatim.  This test pins the
contract by asserting (a) every emitted block starts with the literal
``\\begin{tabular}`` (i.e. *not* ``\\textbackslash{}begin``), (b) no
escaped sequence such as ``\\textbackslash{}`` appears, and (c) round-
tripping the block through ``inject_results`` returns the same raw
LaTeX rather than a doubly-escaped form.
"""
from __future__ import annotations

from report.inject import TexSource, inject_results
from report.inject_tables import inject_tables

# A populated metrics fixture so every emitter resolves real values
# instead of \MissingCell{} markers (which themselves contain \\ and
# would mask a regression).
_FIXTURE: dict[str, object] = {
    "test_set_kind": "internal_63",
    "donut_f1": 0.78, "pipeline_f1": 0.80, "gtocr_rulebased_f1": 0.65,
    **{f"donut_f1_{f}": 0.8 for f in ("company", "date", "address", "total")},
    **{f"pipeline_f1_{f}": 0.82 for f in ("company", "date", "address", "total")},
    **{f"rulebased_f1_{f}": 0.7 for f in ("company", "date", "address", "total")},
    **{f"donut_precision_{f}": 0.8 for f in ("company", "date", "address", "total")},
    **{f"donut_recall_{f}": 0.8 for f in ("company", "date", "address", "total")},
    **{f"donut_em_{f}": 0.7 for f in ("company", "date", "address", "total")},
    **{f"pipeline_precision_{f}": 0.8 for f in ("company", "date", "address", "total")},
    **{f"pipeline_recall_{f}": 0.8 for f in ("company", "date", "address", "total")},
    **{f"pipeline_em_{f}": 0.7 for f in ("company", "date", "address", "total")},
    "donut_latency_mean": 120.0, "pipeline_latency_mean": 250.0,
    "git_sha": "abc1234", "torch_version": "2.1.0", "seed": 42,
    "donut_epochs": 10, "pipeline_epochs": 10,
}

_TARGET_KEYS = (
    "table_headline_f1", "table_extended", "table_latency",
    "table_training", "table_env",
)


def test_emitted_blocks_are_raw_latex() -> None:
    blocks = inject_tables(dict(_FIXTURE))
    for key in _TARGET_KEYS:
        block = blocks[key]
        assert block.startswith("\\begin{tabular}"), (
            f"{key} should start with literal \\begin{{tabular}}, got: {block[:60]!r}"
        )
        # No re-escaping: the string ``\textbackslash{}`` is the
        # signature of the bug — _latex_escape_text turning each ``\``
        # into ``\textbackslash{}``.
        assert "\\textbackslash{}" not in block, (
            f"{key} contains \\textbackslash{{}} — emitter output was re-escaped"
        )


def test_emitted_blocks_are_tex_source_marked() -> None:
    """Without the ``TexSource`` marker every block would be re-escaped
    by :func:`report.inject._format_value` (str branch).
    """
    blocks = inject_tables(dict(_FIXTURE))
    for key in _TARGET_KEYS:
        assert isinstance(blocks[key], TexSource), (
            f"{key} must be a TexSource so inject_results bypasses "
            "_latex_escape_text; otherwise the table renders as escaped "
            "literal text in paper_filled.pdf (Bug A)."
        )


def test_inject_results_does_not_double_escape_table_values() -> None:
    """End-to-end: inject_tables() values survive inject_results() raw."""
    metrics = dict(_FIXTURE)
    for k, v in inject_tables(metrics).items():
        metrics[k] = v
    template = "\\VAR{table_headline_f1}\n%end"
    out = inject_results(template, metrics)
    assert "\\begin{tabular}" in out
    assert "\\textbackslash{}begin" not in out
    assert "\\textbackslash{}\\{tabular" not in out
