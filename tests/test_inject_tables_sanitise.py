"""Audit A2 regression: every emitted ``table_*`` block survives sanitisation.

Verifies the three invariants enforced by
``report.inject_tables._sanitise_tabular``:

* Balanced ``{`` / ``}`` braces.
* Every ``\\cite{}`` resolves against ``report/references.bib``.
* No stray ``\\VAR{}`` placeholders.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from report.inject_tables import (
    _bib_keys,
    _sanitise_tabular,
    inject_tables,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "canonical_347_n1.json"


def _emitted_blocks() -> dict[str, str]:
    metrics: dict[str, object] = json.loads(_FIXTURE.read_text())
    return inject_tables(metrics)


def test_emitted_tables_balanced_braces() -> None:
    """Every non-empty emitted block has balanced (escaped-aware) braces."""
    for key, block in _emitted_blocks().items():
        if not block:
            continue
        stripped = re.sub(r"\\[{}]", "", block)
        assert stripped.count("{") == stripped.count("}"), (
            f"{key}: unbalanced braces {stripped.count('{')}/{stripped.count('}')}"
        )


def test_emitted_tables_no_stray_var() -> None:
    """No ``\\VAR{...}`` survives in any emitted tabular."""
    for key, block in _emitted_blocks().items():
        assert "\\VAR{" not in block, f"{key} carries stray \\VAR{{}}"


def test_emitted_tables_cites_resolve_against_bib() -> None:
    """Every ``\\cite{key}`` in an emitted block exists in references.bib."""
    bib = _bib_keys()
    if not bib:
        pytest.skip("references.bib not located")
    for key, block in _emitted_blocks().items():
        cites = re.findall(r"\\cite\{([^}]+)\}", block)
        for group in cites:
            for k in group.split(","):
                k = k.strip()
                assert k in bib, f"{key}: \\cite{{{k}}} not in references.bib"


def test_sanitiser_raises_on_unbalanced_braces() -> None:
    from core.errors import EvalError
    with pytest.raises(EvalError, match="unbalanced braces"):
        _sanitise_tabular("\\begin{tabular}{l} a & b \\\\ \\end{tabular} {")


def test_sanitiser_raises_on_stray_var() -> None:
    from core.errors import EvalError
    with pytest.raises(EvalError, match="stray"):
        _sanitise_tabular("\\begin{tabular}{l} \\VAR{leak} \\\\ \\end{tabular}")


def test_sanitiser_raises_on_unresolved_cite() -> None:
    from core.errors import EvalError
    with pytest.raises(EvalError, match="cite"):
        _sanitise_tabular(
            "\\begin{tabular}{l} \\cite{this_key_does_not_exist_xyz} \\\\ \\end{tabular}"
        )


def test_sanitiser_passes_empty() -> None:
    assert _sanitise_tabular("") == ""
