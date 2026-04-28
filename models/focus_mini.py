"""PR-A / T-E — Mini assigner (graph-attention rename).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: canonical home of the small (~120K-param) opt-in assigner that
    was historically named ``gat_assigner`` (graph-attention network
    over a kNN graph of OCR regions).  PR-A/T-E renames this to
    ``mini_assigner`` (option-b in the plan) since the practical
    difference from the main MLP+cross-attn assigner is the parameter
    budget, not the graph topology — at ``k = N`` the kNN graph is
    fully dense and the model degenerates to a plain transformer
    encoder.  GAT citation is dropped from ``references.bib``.

Public API — ``mini_assign(feats, config) -> FieldAssignment`` —
matches the legacy 2-in/1-out contract.  The legacy
:mod:`models.gat_assigner` module remains importable as a thin
re-export so callers do not break in PR-A; PR-C will drop the
legacy alias.
"""
from __future__ import annotations

from models.focus_gat import (
    AssignerInput,
    FieldAssignment,
)
from models.focus_gat import (
    gat_assign as mini_assign,
)

__all__ = ["AssignerInput", "FieldAssignment", "mini_assign"]
