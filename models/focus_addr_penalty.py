"""FOCUS-A address-span boundary penalty + post-shrink (inference-side).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: shared helpers for lifting ``pipeline_address_precision`` without
    retraining the FOCUS-A span head.  The trained ``_AddressSpanHead``
    occasionally argmaxes a span whose first / last region is pure
    receipt header / footer (company boilerplate, ``TAX INVOICE``,
    ``INV NO …``, ``CASH RECEIPT #``, ``DOC NO``, ``TIME 10:47``,
    ``TABLE #11``, GST / ROC, phone-digit runs).  The recall the head
    delivered (~0.95) is fine; only the boundaries leak.

    Two inference-time levers — neither retrains the assigner:

    * :func:`boundary_prior_vec` returns ``1.0`` for any region whose
      text matches :func:`models.consensus._is_addr_boundary` (which
      already encodes the money / date / phone / GST / company-header
      / invoice-cashier transition / postcode-with-contamination
      classes).  Callers add ``-λ · (prior[i] + prior[j])`` to the
      ``(N, N)`` ``score`` matrix before argmax so a header / footer
      cell is repelled twice (once via its row, once via its column).

    * :func:`shrink_addr_span` walks the predicted ``[i, j]`` interval
      inwards from both ends, dropping any boundary line.  The argmax
      is occasionally still off by one even after the penalty (a
      borderline header has ``prior=0`` because :func:`_is_addr_boundary`
      doesn't match it but a mildly stricter check via the
      :data:`_LEADING_STRIP_RE` / :data:`_TRAILING_STRIP_RE` pair does);
      a deterministic post-shrink halts that residual leak.

    All checks are receipt-text-only — no bbox geometry needed — so the
    helpers stay torch-free and importable from CPU-only CI.
"""
from __future__ import annotations

import re

from models.consensus import _is_addr_boundary

# A standalone tightening regex pair used by :func:`shrink_addr_span`.
# ``_is_addr_boundary`` already covers the money / date / phone / GST
# / SDN BHD / receipt-metadata cases; these add the residual patterns
# that occasionally slip through the broader classifier — most often
# the ``CASH RECEIPT # CS00068955`` style ID where a digit-bearing
# token survives because it isn't *itself* a money / date / phone
# match, only its neighbour is.
_LEADING_STRIP_RE = re.compile(
    r"^\s*("
    r"tax\s+invoice"             # TAX INVOICE / TAX INVOICE NO
    r"|invoice"                  # standalone INVOICE header
    r"|simplified\s+tax"         # simplified tax invoice
    r"|cash\s+(?:sale|bill|receipt)"
    r"|(?:welcome|thank\s+you)"
    r")\b",
    re.IGNORECASE,
)
_TRAILING_STRIP_RE = re.compile(
    r"\b("
    r"inv(?:oice)?\s*(?:no|number|#)"  # INV NO, INVOICE NUMBER, INV #
    r"|doc\s*(?:no|#)"                  # DOC NO, DOC #
    r"|cash(?:ier|\s+receipt|\s+sale|\s+bill)?"  # CASH, CASHIER, CASH RECEIPT
    r"|receipt\s*(?:no|#)?"             # RECEIPT, RECEIPT NO, RECEIPT #
    r"|date\s*[:#]"                     # DATE: / DATE#
    r"|time\b"                          # TIME 10:47
    r"|table\b"                         # TABLE #11
    r"|cover\b|waiter\b|counter\b"      # restaurant footer fields
    r"|gst\s*(?:no|#)?"                 # GST NO
    r"|sst\s*(?:no|#)?|vat\s*(?:no|#)?"  # SST NO / VAT NO
    r"|roc\s*(?:no|#)?"                 # ROC NO
    r"|tel(?:ephone)?\b|fax\b|phone\b"  # contact-info footer
    r"|order\s*(?:no|#)?"               # ORDER NO
    r")\b",
    re.IGNORECASE,
)
# Phone-shaped digit runs (≥9 contiguous digits, possibly hyphenated)
# — any line carrying one is a footer marker.
_PHONE_RUN_RE = re.compile(r"\d[\d\s\-]{8,}\d")


def _is_strip_line(text: str) -> bool:
    """True iff ``text`` should be excised from the address span ends.

    Combines the broad :func:`_is_addr_boundary` classifier with the
    tighter leading / trailing receipt-metadata regexes above.  Used by
    :func:`shrink_addr_span` to walk the predicted span inward from
    both ends until a non-boundary line is found.
    """
    s = text.strip()
    if not s:
        return True
    if _is_addr_boundary(s):
        return True
    if _LEADING_STRIP_RE.search(s):
        return True
    if _TRAILING_STRIP_RE.search(s):
        return True
    return bool(_PHONE_RUN_RE.search(s))


def boundary_prior_vec(texts: list[str]) -> list[float]:
    """Per-region boundary indicator (``1.0`` for boundary, ``0.0`` else).

    Wraps :func:`_is_addr_boundary` so the FOCUS-A inference path can
    add an additive penalty to the ``score`` matrix without re-deriving
    the boundary classes.  Returned as a plain ``list[float]`` so the
    caller can convert to whatever tensor type their backend uses.
    """
    return [1.0 if _is_addr_boundary(t.strip()) else 0.0 for t in texts]


def shrink_addr_span(
    span: tuple[int, int], texts: list[str],
) -> tuple[int, int]:
    """Trim header / footer lines from a predicted ``[i, j]`` span.

    ``span`` is the ``(i, j)`` interval emitted by
    :meth:`AttentionAssigner.address_span` (kept as a 2-tuple to honour
    the 2-in/1-out contract in ``AGENTS.md``).  Walks ``i`` forward
    while ``texts[i]`` is a strip-line and ``j`` backward while
    ``texts[j]`` is a strip-line.  Halts as soon as a non-boundary
    line is found at either end.  Returns ``(0, -1)`` — the canonical
    "empty span" sentinel used by :class:`core.types.AddrPred` — when
    the entire interval collapses (so the caller can route to the
    legacy fallback chain).

    The shrink is deterministic and idempotent.  Numeric-only /
    postcode-bearing lines are NOT boundaries (per
    :func:`_is_addr_boundary`'s postcode exemption) so Malaysian
    addresses ending in ``40000 SHAH ALAM`` are preserved.
    """
    i, j = span
    n = len(texts)
    if i < 0 or j < 0 or j < i or j >= n:
        return (0, -1)
    while i <= j and _is_strip_line(texts[i]):
        i += 1
    if i > j:
        return (0, -1)
    while j >= i and _is_strip_line(texts[j]):
        j -= 1
    if j < i:
        return (0, -1)
    return (i, j)


__all__ = [
    "boundary_prior_vec",
    "shrink_addr_span",
]
