"""Day-1 oracle ceiling for the FOCUS span-cohesion address head.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: gating diagnostic invoked by ``stages/eval.py`` when
    ``--oracle-address`` is set.  Tier A measures the
    span-concatenation ceiling on receipts whose box files exist
    (training-archive subset) — the upper bound assuming perfect
    line-to-field labels and gold-quality text.  Tier B measures
    the realistic ceiling on canonical-347 by re-running YOLO+TrOCR
    detection and choosing the contiguous span (i*, j*) of detected
    lines whose joined text best matches the gold address string
    under token-set F1.  The decision keys ``proceed_focus`` /
    ``borderline_ocr_bound`` / ``abort_ocr_bottleneck`` follow the
    brief's Day-1 gate (>=0.95/>=0.85, [0.85, 0.95), <0.85).
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from core.errors import EvalError
from core.types import DataSplit, ExpConfig, Receipt
from data.sroie_crops import _parse_box_file
from models.pipeline_detect import _detect_and_read, _fallback_full_image

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    import torch

log = logging.getLogger("kaggle2")

L_MAX = 8

# Decision-gate thresholds — the brief's Day-1 spec.
_DECIDE_PROCEED_F1 = 0.95
_DECIDE_PROCEED_EM = 0.85
_DECIDE_ABORT_F1 = 0.85
# P1 pre-registered prediction (oracle ceiling threshold).
_P1_F1 = 0.90
_P1_EM = 0.70


def _token_prf1(gold: str, pred: str) -> tuple[float, float, float, bool]:
    """Token-set precision, recall, F1, plus EM (mirrors core.metrics.token_f1)."""
    g = set(gold.lower().split())
    p = set(pred.lower().split())
    em = gold.lower() == pred.lower()
    if not g and not p:
        return 1.0, 1.0, 1.0, em
    if not g or not p:
        return 0.0, 0.0, 0.0, em
    common = g & p
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, em


def _gold_address(rec: Receipt) -> str:
    """Return the gold address string for ``rec`` (or '')."""
    return next(
        (f.value for f in rec.fields if f.name.lower() == "address"), "",
    )


def _addr_text_in_order(rec: Receipt, fields: list[str]) -> tuple[str, int]:
    """Concatenated text of address-labeled box-file lines for ``rec``.

    Returns ``(joined_text, n_lines)``.  ``n_lines == 0`` means either
    no box file exists (caller distinguishes via path check) or the
    box file exists but no line was labeled ``address``.
    """
    crops = _parse_box_file(rec, fields)
    addr = [c for c in crops if c.field_label == "address"]
    if not addr:
        return "", 0
    return " ".join(c.text for c in addr), len(addr)


def _tier_a_clean(
    receipts: list[Receipt], fields: list[str],
) -> dict[str, Any]:
    """Span-concatenation ceiling on receipts whose box files exist."""
    n_skipped_no_box = 0
    n_skipped_no_label = 0
    span_lengths: list[int] = []
    p_sum = r_sum = f1_sum = em_sum = 0.0
    n = 0
    for rec in receipts:
        box_path = (
            rec.image_path.parent.parent
            / "box"
            / (rec.image_path.stem + ".txt")
        )
        if not box_path.exists():
            n_skipped_no_box += 1
            continue
        pred_text, n_lines = _addr_text_in_order(rec, fields)
        if n_lines == 0:
            n_skipped_no_label += 1
            continue
        gold = _gold_address(rec)
        prec, rec_v, f1, em = _token_prf1(gold, pred_text)
        p_sum += prec
        r_sum += rec_v
        f1_sum += f1
        em_sum += 1.0 if em else 0.0
        span_lengths.append(n_lines)
        n += 1
    return {
        "source": "training_box_files",
        "n_receipts": n,
        "n_skipped_no_box": n_skipped_no_box,
        "n_skipped_no_addr_label": n_skipped_no_label,
        "f1": round(f1_sum / n, 4) if n else 0.0,
        "em": round(em_sum / n, 4) if n else 0.0,
        "p": round(p_sum / n, 4) if n else 0.0,
        "r": round(r_sum / n, 4) if n else 0.0,
        "mean_span_length": (
            round(sum(span_lengths) / len(span_lengths), 3)
            if span_lengths else 0.0
        ),
        "max_span_length": max(span_lengths) if span_lengths else 0,
    }


def _best_span(texts: list[str], gold: str) -> tuple[int, int, float]:
    """Argmax-F1 contiguous span over detected lines, capped at ``L_MAX``.

    Returns ``(i, j, f1)``.  ``(0, -1, 0.0)`` when ``texts`` is empty
    or ``gold`` has no tokens.  Tie-break: shortest span first, then
    smallest start index.
    """
    n = len(texts)
    g_tok = set(gold.lower().split())
    if n == 0 or not g_tok:
        return 0, -1, 0.0
    best_i, best_j, best_f1, best_len = 0, 0, -1.0, n + 1
    for i in range(n):
        for j in range(i, min(i + L_MAX, n)):
            span_text = " ".join(texts[i : j + 1])
            s_tok = set(span_text.lower().split())
            if not s_tok:
                continue
            common = g_tok & s_tok
            if not common:
                f1 = 0.0
            else:
                prec = len(common) / len(s_tok)
                rec_v = len(common) / len(g_tok)
                f1 = 2 * prec * rec_v / (prec + rec_v)
            length = j - i + 1
            if f1 > best_f1 or (f1 == best_f1 and length < best_len):
                best_i, best_j, best_f1, best_len = i, j, f1, length
    return best_i, best_j, max(best_f1, 0.0)


def _load_yolo_trocr(config: ExpConfig) -> tuple[Any, Any, Any, int, str]:
    """Load YOLO + TrOCR (skip assigner) for the canonical-347 oracle pass."""
    from models.pipeline_eval import _paths_from_config, _resolve_yolo_img
    paths = _paths_from_config(config)
    if not Path(paths.yolo).exists():
        raise EvalError(f"YOLO checkpoint not found at {paths.yolo}")
    if not Path(paths.trocr).exists():
        raise EvalError(f"TrOCR checkpoint not found at {paths.trocr}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise EvalError("ultralytics not installed") from exc
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    yolo = YOLO(paths.yolo)
    trocr_proc = TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(paths.trocr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    trocr_model.eval()
    yolo_img = _resolve_yolo_img(paths, config)
    return yolo, trocr_proc, trocr_model, yolo_img, device


def _tier_b_canonical_heuristic(
    test: list[Receipt], config: ExpConfig,
) -> dict[str, Any]:
    """Heuristic span-ceiling on canonical-347 using YOLO+TrOCR detected lines."""
    yolo, trocr_proc, trocr_model, yolo_img, device = _load_yolo_trocr(config)
    span_lengths: list[int] = []
    n_zero_match = 0
    p_sum = r_sum = f1_sum = em_sum = 0.0
    n = 0
    with torch.no_grad():
        for rec in test:
            try:
                img = Image.open(rec.image_path).convert("RGB")
                texts, _, _ = _detect_and_read(
                    yolo, trocr_proc, trocr_model, img,
                    str(rec.image_path), config, yolo_img, device,
                )
                if not texts:
                    texts, _, _ = _fallback_full_image(
                        trocr_proc, trocr_model, img, config, device,
                    )
            except (OSError, RuntimeError, ValueError):
                log.exception(
                    "oracle: receipt %s detection failed", rec.image_path.stem,
                )
                texts = []
            gold = _gold_address(rec)
            i_star, j_star, span_f1 = _best_span(texts, gold)
            if j_star >= i_star and texts:
                pred_text = " ".join(texts[i_star : j_star + 1])
                length = j_star - i_star + 1
            else:
                pred_text = ""
                length = 0
            prec, rec_v, f1, em = _token_prf1(gold, pred_text)
            p_sum += prec
            r_sum += rec_v
            f1_sum += f1
            em_sum += 1.0 if em else 0.0
            if length:
                span_lengths.append(length)
            if span_f1 == 0.0:
                n_zero_match += 1
            n += 1
    return {
        "source": "canonical_347_yolo_trocr",
        "L_max": L_MAX,
        "n_receipts": n,
        "n_zero_match": n_zero_match,
        "f1": round(f1_sum / n, 4) if n else 0.0,
        "em": round(em_sum / n, 4) if n else 0.0,
        "p": round(p_sum / n, 4) if n else 0.0,
        "r": round(r_sum / n, 4) if n else 0.0,
        "mean_span_length": (
            round(sum(span_lengths) / len(span_lengths), 3)
            if span_lengths else 0.0
        ),
        "max_span_length": max(span_lengths) if span_lengths else 0,
    }


def _decide(tier_b: dict[str, Any]) -> tuple[str, bool]:
    """Map Tier B numbers onto the brief's Day-1 decision branch + P1 flag."""
    f1 = float(tier_b["f1"])
    em = float(tier_b["em"])
    if f1 >= _DECIDE_PROCEED_F1 and em >= _DECIDE_PROCEED_EM:
        decision = "proceed_focus"
    elif f1 >= _DECIDE_ABORT_F1:
        decision = "borderline_ocr_bound"
    else:
        decision = "abort_ocr_bottleneck"
    p1_passes = (f1 >= _P1_F1) and (em >= _P1_EM)
    return decision, p1_passes


def compute_oracle_address(
    data: DataSplit, config: ExpConfig,
) -> dict[str, Any]:
    """Run Tier A + Tier B oracle and return the JSON-serialisable sidecar."""
    log.info(
        "Day-1 oracle: tier A on training+val box-file receipts (n=%d)...",
        len(data.train) + len(data.val),
    )
    tier_a = _tier_a_clean(
        list(data.train) + list(data.val), list(config.fields),
    )
    log.info(
        "Tier A: n=%d f1=%.4f em=%.4f no_box=%d no_label=%d",
        tier_a["n_receipts"], tier_a["f1"], tier_a["em"],
        tier_a["n_skipped_no_box"], tier_a["n_skipped_no_addr_label"],
    )
    log.info(
        "Day-1 oracle: tier B on canonical test (n=%d)...", len(data.test),
    )
    tier_b = _tier_b_canonical_heuristic(list(data.test), config)
    log.info(
        "Tier B: n=%d f1=%.4f em=%.4f zero_match=%d",
        tier_b["n_receipts"], tier_b["f1"], tier_b["em"],
        tier_b["n_zero_match"],
    )
    decision, p1_passes = _decide(tier_b)
    return {
        "run_id": Path(config.output_dir).name,
        "produced_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "tier_a_clean": tier_a,
        "tier_b_canonical_heuristic": tier_b,
        "p1_thresholds": {"f1": _P1_F1, "em": _P1_EM},
        "decision_thresholds": {
            "proceed_focus": "tier_b f1 >= 0.95 AND em >= 0.85",
            "borderline":    "0.85 <= tier_b f1 < 0.95",
            "abort_ocr":     "tier_b f1 < 0.85",
        },
        "decision": decision,
        "p1_passes": p1_passes,
    }
