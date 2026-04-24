"""Seven-category miss classification for per-sample error analysis.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: classify each wrong-or-missing prediction into one of seven
    disjoint categories so the paper's Table II and the stacked-bar
    figure in :mod:`report.figures_errors` can decompose *why* a model
    loses points:

    * ``missed_detection`` — pipeline: YOLO saw no box for the field.
    * ``ocr_error``        — pipeline: YOLO found a box, TrOCR produced wrong text.
    * ``assigner_error``   — pipeline: right text exists somewhere but wrong field.
    * ``hallucination``    — prediction invented content absent from input.
    * ``partial``          — gold substring of pred or vice-versa.
    * ``wrong_span``       — partial token-overlap, neither substring.
    * ``wrong_normalization`` — ``total``/``date`` normalisation mismatch.
    * ``postprocess_error`` — otherwise-correct text, broken by postprocessing.

    Never raises — a record with no ground truth still maps to a
    sensible bucket so the figure emitter always has data.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

ErrorCategory = Literal[
    "hallucination",
    "partial",
    "wrong_span",
    "wrong_normalization",
    "missed_detection",
    "ocr_error",
    "assigner_error",
    "postprocess_error",
    "correct",
]

_DATE_RE = re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")
_TOTAL_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _same_date(gold: str, pred: str) -> bool:
    """Normalise date strings and compare — covers 2020/01/05 vs 05-01-2020."""
    g = _DATE_RE.search(gold or "")
    p = _DATE_RE.search(pred or "")
    if not (g and p):
        return False
    gparts = sorted(re.split(r"[-/]", g.group()))
    pparts = sorted(re.split(r"[-/]", p.group()))
    return gparts == pparts


def _same_total(gold: str, pred: str) -> bool:
    """Normalise currency strings and compare — covers 12.34 vs 12,34 vs $12.34."""
    g = _TOTAL_RE.search((gold or "").replace(",", "."))
    p = _TOTAL_RE.search((pred or "").replace(",", "."))
    if not (g and p):
        return False
    try:
        return abs(float(g.group()) - float(p.group())) < 0.005
    except ValueError:
        return False


def classify_miss(
    field: str,
    gold: str,
    pred: str,
    *,
    had_detection: bool | None = None,
    text_present_in_ocr: bool | None = None,
) -> ErrorCategory:
    """Assign a single (field, gold, pred) triple to one of eight categories.

    ``had_detection`` / ``text_present_in_ocr`` are optional pipeline-
    specific signals.  When ``None`` the classifier uses text-only
    heuristics (useful for DONUT error-type decomposition).
    """
    g = (gold or "").strip().lower()
    p = (pred or "").strip().lower()
    if g == p:
        return "correct"
    if had_detection is False:
        return "missed_detection"
    if not p:
        # Empty prediction with a non-empty gold — for pipeline this is
        # usually OCR failure; for DONUT it's a decoder miss.
        return "ocr_error" if had_detection else "hallucination"
    if not g:
        return "hallucination"
    if field == "date" and _same_date(g, p):
        return "wrong_normalization"
    if field == "total" and _same_total(g, p):
        return "wrong_normalization"
    if g in p or p in g:
        return "partial"
    # Assigner error: right text lives in the OCR stream but landed on
    # the wrong field.  text_present_in_ocr is only supplied for the
    # pipeline arm.
    if text_present_in_ocr:
        return "assigner_error"
    g_tokens = set(g.split())
    p_tokens = set(p.split())
    if g_tokens & p_tokens:
        return "wrong_span"
    # Pure-postprocess mismatch vs fully invented text.  Any non-alnum
    # delta with otherwise equal alnum content = postprocess_error.
    if re.sub(r"\W", "", g) == re.sub(r"\W", "", p):
        return "postprocess_error"
    return "hallucination"


def error_breakdown(
    records: Iterable[tuple[str, str, str, bool | None, bool | None]],
) -> dict[str, dict[str, int]]:
    """Aggregate per-field × per-category counts for the paper's stacked bars.

    Each record is ``(field, gold, pred, had_detection, text_in_ocr)``.
    Returns ``{field: {category: count}}``.
    """
    out: dict[str, dict[str, int]] = {}
    for field, gold, pred, had_det, text_in in records:
        cat = classify_miss(field, gold, pred,
                            had_detection=had_det, text_present_in_ocr=text_in)
        out.setdefault(field, {}).setdefault(cat, 0)
        out[field][cat] += 1
    return out
