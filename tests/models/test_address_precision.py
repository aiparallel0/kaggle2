"""PR-ADDR-PREC-2 — per-receipt address-precision regression.

Loads :file:`tests/fixtures/address_mismatch_subset.json` (10
representative SROIE-Task-3 receipts whose FOCUS-A span head currently
over-predicts the address by a header / footer line at one or both
ends) and asserts that the inference-side fix chain — ``shrink_addr_span``
+ ``normalize_address_focus`` — produces a normalised prediction whose
token-level *precision* against the GT is >= 0.85 on every receipt.

The test is CPU-only and torch-free: it exercises
:func:`models.focus_addr_penalty.shrink_addr_span` (the post-span shrink
applied at the focus_pipeline level) and
:func:`models.postprocess_address.normalize_address_focus` (the
end-of-pipeline string normaliser).  The boundary penalty applied
inside :meth:`AttentionAssigner.address_span` lives upstream and is
covered by ``tests/models/test_focus_address_span.py``; the assertion
here is purely on the post-shrink + normalise contract because that's
where the precision win lands.

A token-precision floor of 0.85 was chosen so a one-token slack (e.g.
the postcode-bearing tail line carrying a mid-line junk fragment) does
not flake the test on a future receipt-text refresh; the canonical
347-image eval target is 0.90 macro.
"""
from __future__ import annotations

import json
from pathlib import Path

from models.focus_addr_penalty import shrink_addr_span
from models.postprocess_address import normalize_address_focus

_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "address_mismatch_subset.json"
)
_PRECISION_FLOOR = 0.85


def _token_precision(pred: str, gt: str) -> float:
    """Multiset token precision: |pred ∩ gt| / |pred|.

    Both strings are run through :func:`normalize_address_focus` at the
    call site so casing / punctuation / leading-trailing trim drift is
    already squashed before we tokenise.  Multiset semantics (``min``
    of per-token counts) so a duplicated junk token in ``pred`` is
    correctly penalised.
    """
    p_tokens = pred.split()
    g_tokens = gt.split()
    if not p_tokens:
        # An empty prediction can't have wrong tokens; treat as 1.0 so
        # the recall side (covered by other tests) is the only thing
        # that fails on a degenerate empty span.
        return 1.0
    g_counts: dict[str, int] = {}
    for t in g_tokens:
        g_counts[t] = g_counts.get(t, 0) + 1
    overlap = 0
    for t in p_tokens:
        if g_counts.get(t, 0) > 0:
            overlap += 1
            g_counts[t] -= 1
    return overlap / len(p_tokens)


def _predicted_value(rec: dict[str, object]) -> str:
    """Apply the PR-ADDR-PREC-2 post-shrink + normaliser to ``rec``.

    Mirrors what :func:`models.focus_pipeline._assign_learned_with_attn`
    does after :meth:`AttentionAssigner.address_span` returns: shrink
    the span endpoints inwards via :func:`shrink_addr_span`, join the
    surviving lines with a single space, then run the symmetric
    normaliser.  Returns the empty string when the span collapses.
    """
    texts = list(rec["texts"])  # type: ignore[arg-type]
    i_pred, j_pred = rec["addr_pred"]  # type: ignore[misc]
    si, sj = shrink_addr_span((int(i_pred), int(j_pred)), texts)
    if sj < si:
        return ""
    span = " ".join(t.strip() for t in texts[si : sj + 1] if t.strip())
    return normalize_address_focus(span)


def test_per_receipt_token_precision_meets_floor() -> None:
    """Every fixture receipt's normalised prediction precision >= 0.85."""
    payload = json.loads(_FIXTURE.read_text())
    receipts = payload["receipts"]
    assert len(receipts) == 10, "fixture must hold the 10 canonical misses"
    failures: list[tuple[str, float, str]] = []
    for rec in receipts:
        pred = _predicted_value(rec)
        gt_norm = normalize_address_focus(rec["gt"])
        prec = _token_precision(pred, gt_norm)
        if prec < _PRECISION_FLOOR:
            failures.append((rec["id"], prec, pred))
    assert not failures, (
        f"per-receipt precision below {_PRECISION_FLOOR}: {failures}"
    )


def test_macro_token_precision_meets_paper_target() -> None:
    """Macro-averaged precision over the 10 misses >= 0.90 (paper target)."""
    payload = json.loads(_FIXTURE.read_text())
    precisions: list[float] = []
    for rec in payload["receipts"]:
        pred = _predicted_value(rec)
        gt_norm = normalize_address_focus(rec["gt"])
        precisions.append(_token_precision(pred, gt_norm))
    macro = sum(precisions) / len(precisions)
    assert macro >= 0.90, f"macro token-precision {macro:.3f} < 0.90"


def test_shrink_no_op_on_clean_span() -> None:
    """A pre-clean span (no boundary lines at the ends) is unchanged."""
    texts = [
        "NO 12 JALAN MAJU 5",
        "TAMAN MAJU",
        "47000 SUNGAI BULOH",
    ]
    assert shrink_addr_span((0, 2), texts) == (0, 2)


def test_shrink_collapses_to_empty_when_all_boundary() -> None:
    """An interval consisting only of header / footer lines collapses."""
    texts = ["TAX INVOICE", "GST NO: 000504664064", "TOTAL RM 12.50"]
    si, sj = shrink_addr_span((0, 2), texts)
    assert si == 0 and sj == -1
