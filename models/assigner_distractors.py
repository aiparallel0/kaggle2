"""Distractor regexes for address/total fields (Bug 18, factored out of assigner_loss).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: per-region boolean masks flagging SROIE boilerplate that the MIL
    pos-mass loss cannot penalise.  Used as the tie-breaker inside the
    CTKR top-K selector in :mod:`models.assigner_loss`.  No torch dep.
"""
from __future__ import annotations

import re

# has_invoice_no | has_brn | has_gst | has_table_no |
# has_cashier | has_phone | is_tax_invoice_header
_ADDR_DISTRACTOR_RE = re.compile(
    r"\b(?:INV(?:OICE)?\s*NO|TAX\s*INVOICE|CASH(?:IER)?|BRN|"
    r"GST(?:\s*NO)?|TABLE\s*(?:NO)?\b|TABLE\s*\d|"
    r"TEL(?:EPHONE|\.|:)?|FAX|PHONE|H/?P|MOBILE|"
    r"ROC\s*NO|REG(?:ISTRATION)?\s*NO|CO\s*NO)\b",
    re.IGNORECASE,
)
# is_subtotal | is_tax | is_change | is_rounding | is_cash_paid
_TOTAL_DISTRACTOR_RE = re.compile(
    r"\b(?:SUB[\s\-]?TOTAL|SUBTOTAL|"
    r"TAX|GST|SST|VAT|"
    r"CHANGE|KEMBALIAN|"
    r"ROUND(?:ING|ED)?|"
    r"CASH(?:\s+TENDERED|\s+PAID)?|TENDERED|PAID|TENDER)\b",
    re.IGNORECASE,
)


def address_distractor_mask(texts: list[str]) -> list[bool]:
    """Per-region True iff the line looks like an address-field distractor."""
    return [bool(_ADDR_DISTRACTOR_RE.search(t or "")) for t in texts]


def total_distractor_mask(texts: list[str]) -> list[bool]:
    """Per-region True iff the line looks like a total-field distractor."""
    return [bool(_TOTAL_DISTRACTOR_RE.search(t or "")) for t in texts]


def field_distractor_mask(field_name: str, texts: list[str]) -> list[bool]:
    """Dispatch to the per-field distractor regex; all-False for unannotated fields."""
    f = field_name.lower()
    if f == "address":
        return address_distractor_mask(texts)
    if f == "total":
        return total_distractor_mask(texts)
    return [False] * len(texts)


__all__ = [
    "address_distractor_mask",
    "field_distractor_mask",
    "total_distractor_mask",
]
