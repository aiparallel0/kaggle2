"""Audit A1 regression: zero rulebased rows on canonical_347 in inject_tables.

On a canonical SROIE Task-3 build the rule-based / GT-OCR-rule arms
cannot run (no GT bounding boxes), so the headline-F1, per-field, and
ablation tables emitted by ``report.inject_tables`` must drop the
rule-based row entirely — never render it with ``\\MissingCell{}``
cells (Audit A1).
"""
from __future__ import annotations

from report.inject_tables import inject_tables, render_f1_table


def test_render_f1_table_drops_rulebased_on_canonical() -> None:
    """The Rule-based column header and rows are absent on canonical_347."""
    out = render_f1_table({
        "test_set_kind": "canonical_347",
        "donut_f1": 0.82, "pipeline_f1": 0.81,
        "donut_f1_company": 0.88, "pipeline_f1_company": 0.87,
    })
    assert "Rule-based" not in out, "Rule-based column must be dropped on canonical_347"
    assert "rulebased" not in out.lower()
    # Two-system table → ``lcc`` column spec, never ``lccc``.
    assert "{lcc}" in out
    assert "{lccc}" not in out


def test_inject_tables_canonical_no_rulebased_keys() -> None:
    """No emitted tabular references rulebased_/gtocr_rulebased_ on canonical."""
    blocks = inject_tables({
        "test_set_kind": "canonical_347",
        "donut_f1": 0.82, "pipeline_f1": 0.81,
    })
    for key, block in blocks.items():
        assert "rulebased_" not in block.lower(), (
            f"{key}: rulebased_ key leaked into emitted tabular on canonical"
        )
        # bare "Rule-based" header should also be absent
        assert "Rule-based" not in block, (
            f"{key}: Rule-based label leaked into emitted tabular on canonical"
        )


def test_render_f1_table_keeps_rulebased_on_basic() -> None:
    """Non-canonical (basic 500/63/63) splits keep the rule-based column."""
    out = render_f1_table({
        "donut_f1": 0.82, "pipeline_f1": 0.81,
        "rulebased_f1_company": 0.5,
    })
    assert "Rule-based" in out
    assert "{lccc}" in out
