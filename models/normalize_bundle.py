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

Audit-fix (research integrity): both DONUT and pipeline arms now use
**the same** address/company normaliser (the FOCUS punctuation +
casefold pass, applied symmetrically to pred and GT).  Earlier the
DONUT arm used a strict-only normaliser while the pipeline arm got
the punctuation-tolerant pass — that asymmetry could inflate
pipeline F1 by ~0.01–0.03 absolute on receipts where SROIE GT and
pred differ only in trailing punctuation (``"SDN BHD."`` vs ``"SDN
BHD"``).  Routing both arms through the same map is required for an
honest DONUT-vs-pipeline ΔF1 comparison.
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
from models.postprocess_company import normalize_company_focus

__all__ = [
    "FIELD_NORMALISERS_DONUT",
    "FIELD_NORMALISERS_PIPELINE",
    "normalize_bundle",
    "normalize_fields",
]


def _normalize_address_pipeline(value: str) -> str:
    """Compose legacy postcode-repair with FOCUS punctuation pass (Bug 18)."""
    return normalize_address_focus(normalize_address(value))


def _normalize_company_pipeline(value: str) -> str:
    """Compose legacy ``normalize_company`` with FOCUS-C casefold + punct
    pass.  Mirrors :func:`_normalize_address_pipeline` so pred and GT
    reduce to the same casefolded token set on receipts where the OCR /
    GT differ only in trailing punctuation (``"SDN. BHD."`` vs
    ``"SDN BHD"``).  Applied symmetrically to both arms of the
    pipeline-eval bundle by :func:`normalize_bundle`.
    """
    return normalize_company_focus(normalize_company(value))


FIELD_NORMALISERS_PIPELINE: dict[str, Callable[[str], str]] = {
    "total": normalize_total,
    "date": normalize_date,
    "company": _normalize_company_pipeline,
    "address": _normalize_address_pipeline,
}

# Audit-fix: DONUT arm now shares the pipeline's address/company
# normalisers so pred-vs-GT punctuation tolerance is the same on both
# sides of the headline ΔF1 comparison.  ``total`` and ``date`` were
# already shared.  This drops the pre-fix asymmetry that gave the
# pipeline a free ~0.01–0.03 F1 head-start on receipts where SROIE GT
# trailing punctuation differed from OCR.  Applied symmetrically to
# pred and GT via :func:`normalize_bundle` so a punctuation-tolerant
# normaliser is *not* a leak — it canonicalises both sides equally.
FIELD_NORMALISERS_DONUT: dict[str, Callable[[str], str]] = dict(
    FIELD_NORMALISERS_PIPELINE,
)


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
