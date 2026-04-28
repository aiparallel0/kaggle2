"""Shared per-field normaliser map + ``normalize_bundle`` helper.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: single source of truth for the symmetric ``(preds, receipts)``
    pre-metric normalisation.  Each eval arm (DONUT, pipeline,
    rule-based) calls :func:`normalize_bundle` with its own per-field
    normaliser map so the bundle the headline F1/NED/EM scorer
    (:func:`core.metrics.compute_metrics`) sees is bit-identical to
    the bundle the extended-metrics producer
    (:func:`core.extended_metrics.summarise_extended`) sees.  Before
    PR #111 each arm rolled its own ``_nt`` and the producer-side
    ``stages.eval._build_bundles`` mixed normalised preds with raw
    gold, so per-field precision / recall / bootstrap CI collapsed
    while headline F1 stayed correct (PR #110 follow-up).

Two maps are exposed so the DONUT and pipeline arms keep their
existing — and slightly different — address normalisation: the
pipeline composes the legacy postcode-repair pass with the FOCUS
punctuation/casefold pass (Bug 18); the DONUT arm uses the legacy
pass alone, matching the headline F1 reported in PR #110.  The
remaining three fields (``total`` / ``date`` / ``company``) are
identical across arms.
"""
from __future__ import annotations

from collections.abc import Callable

from core.types import Field, Prediction, Receipt
from models.donut_eval import normalize_total
from models.normalize import (
    normalize_address,
    normalize_company,
    normalize_date,
)
from models.postprocess_address import normalize_address_focus

__all__ = [
    "FIELD_NORMALISERS_DONUT",
    "FIELD_NORMALISERS_PIPELINE",
    "normalize_bundle",
    "normalize_fields",
]


def _normalize_address_pipeline(value: str) -> str:
    """Compose legacy postcode-repair with FOCUS punctuation pass (Bug 18)."""
    return normalize_address_focus(normalize_address(value))


FIELD_NORMALISERS_PIPELINE: dict[str, Callable[[str], str]] = {
    "total": normalize_total,
    "date": normalize_date,
    "company": normalize_company,
    "address": _normalize_address_pipeline,
}

FIELD_NORMALISERS_DONUT: dict[str, Callable[[str], str]] = {
    "total": normalize_total,
    "date": normalize_date,
    "company": normalize_company,
    "address": normalize_address,
}


def _identity(s: str) -> str:
    return s


def normalize_fields(
    fields: list[Field],
    normalisers: dict[str, Callable[[str], str]] = FIELD_NORMALISERS_PIPELINE,
) -> list[Field]:
    """Apply ``normalisers`` to every ``Field.value``."""
    return [Field(
        name=f.name,
        value=normalisers.get(f.name.lower(), _identity)(f.value),
    ) for f in fields]


def normalize_bundle(
    preds: list[Prediction], receipts: list[Receipt],
    normalisers: dict[str, Callable[[str], str]] = FIELD_NORMALISERS_PIPELINE,
) -> tuple[list[Prediction], list[Receipt]]:
    """Return ``(preds', receipts')`` with both sides field-normalised."""
    n_preds = [Prediction(receipt_id=p.receipt_id,
                          fields=normalize_fields(p.fields, normalisers))
               for p in preds]
    n_recs = [Receipt(image_path=r.image_path,
                      fields=normalize_fields(r.fields, normalisers))
              for r in receipts]
    return n_preds, n_recs
