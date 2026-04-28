"""Oracle ceiling arms: LLM zero-shot (P4) and FOCUS span-cohesion (Day-1)."""
from __future__ import annotations

import base64
import contextlib
import datetime as _dt
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from core.errors import EvalError
from core.types import DataSplit, ExpConfig, Field, Prediction, Receipt
from data.sroie_crops import _parse_box_file
from models.detect import _detect_and_read, _fallback_full_image

with contextlib.suppress(ImportError):
    import torch
if TYPE_CHECKING:
    import torch
log, L_MAX = logging.getLogger("kaggle2"), 8
_DECIDE_PROCEED_F1, _DECIDE_PROCEED_EM, _DECIDE_ABORT_F1, _P1_F1, _P1_EM = 0.95, 0.85, 0.85, 0.90, 0.70
_PROMPT = ('Extract the following four fields from this SROIE-format receipt: company '
    '(merchant name), date (as printed), address (full street address), total (final '
    'amount paid, numeric only). Respond with ONLY a JSON object with keys '
    '"company","date","address","total".')
_EMPTY: dict[str, str] = {"company": "", "date": "", "address": "", "total": ""}

def _token_prf1(gold: str, pred: str) -> tuple[float, float, float, bool]:
    """Token-set precision, recall, F1, plus EM."""
    g, p, em = set(gold.lower().split()), set(pred.lower().split()), gold.lower() == pred.lower()
    if not g and not p:
        return 1.0, 1.0, 1.0, em
    if not g or not p:
        return 0.0, 0.0, 0.0, em
    prec, rec = len(g & p) / len(p), len(g & p) / len(g)
    return prec, rec, (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0), em

def _best_span(texts: list[str], gold: str) -> tuple[int, int, float]:
    """Argmax-F1 contiguous span over detected lines, capped at L_MAX."""
    n, g_tok = len(texts), set(gold.lower().split())
    if n == 0 or not g_tok:
        return 0, -1, 0.0
    best_i, best_j, best_f1, best_len = 0, 0, -1.0, n + 1
    for i in range(n):
        for j in range(i, min(i + L_MAX, n)):
            s_tok = set(" ".join(texts[i:j + 1]).lower().split())
            if not s_tok:
                continue
            common = g_tok & s_tok
            f1, length = (2 * len(common) / (len(s_tok) + len(g_tok)) if common else 0.0), j - i + 1
            if f1 > best_f1 or (f1 == best_f1 and length < best_len):
                best_i, best_j, best_f1, best_len = i, j, f1, length
    return best_i, best_j, max(best_f1, 0.0)

def _call_llm(image_b64: str, mime: str, api: str) -> dict[str, str]:
    """Call Anthropic or OpenAI; raises EvalError on SDK failure."""
    if api == "openai":
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as e:
            raise EvalError("openai SDK not installed") from e
        resp = openai.OpenAI().chat.completions.create(model="gpt-4o", max_tokens=512,
            messages=[{"role": "user", "content": [{"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}]}])
        text = resp.choices[0].message.content or ""
    else:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise EvalError("anthropic SDK not installed") from e
        resp = anthropic.Anthropic().messages.create(model="claude-3-5-sonnet-20241022", max_tokens=512,
            messages=[{"role": "user", "content": [{"type": "image", "source":
                {"type": "base64", "media_type": mime, "data": image_b64}}, {"type": "text", "text": _PROMPT}]}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return dict(_EMPTY)
    try:
        obj: Any = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return dict(_EMPTY)
    return {k: str(obj.get(k, "")) for k in _EMPTY} if isinstance(obj, dict) else dict(_EMPTY)

def run_llm_ceiling(image_path: Path, config: ExpConfig) -> Prediction:
    """Predict a Receipt using the foundation-model arm (cached)."""
    if not config.foundation_enabled:
        return Prediction(receipt_id=image_path.stem, fields=[])
    cache_path, cache = Path(config.foundation_cache_path), {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            cache = {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            pass
    with image_path.open("rb") as fh:
        key = hashlib.sha256(fh.read()).hexdigest()[:16]
    if key not in cache:
        mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        with image_path.open("rb") as fh:
            cache[key] = _call_llm(base64.standard_b64encode(fh.read()).decode("ascii"), mime, config.foundation_api)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, cache_path)
    return Prediction(receipt_id=image_path.stem, fields=[Field(name=k, value=v) for k, v in cache[key].items()])

def _gold_address(rec: Receipt) -> str: return next((f.value for f in rec.fields if f.name.lower() == "address"), "")

def _tier_a_clean(receipts: list[Receipt], fields: list[str]) -> dict[str, Any]:
    """Span-concat ceiling on receipts whose box files exist."""
    n_no_box, n_no_label, span_lens, p_sum, r_sum, f1_sum, em_sum, n = 0, 0, [], 0.0, 0.0, 0.0, 0.0, 0
    for rec in receipts:
        box_path = rec.image_path.parent.parent / "box" / (rec.image_path.stem + ".txt")
        if not box_path.exists():
            n_no_box += 1
            continue
        addr = [c for c in _parse_box_file(rec, fields) if c.field_label == "address"]
        if not addr:
            n_no_label += 1
            continue
        prec, rec_v, f1, em = _token_prf1(_gold_address(rec), " ".join(c.text for c in addr))
        p_sum, r_sum, f1_sum, em_sum = p_sum + prec, r_sum + rec_v, f1_sum + f1, em_sum + (1.0 if em else 0.0)
        span_lens.append(len(addr))
        n += 1
    return {"source": "training_box_files", "n_receipts": n, "n_skipped_no_box": n_no_box,
        "n_skipped_no_addr_label": n_no_label, "f1": round(f1_sum / n, 4) if n else 0.0,
        "em": round(em_sum / n, 4) if n else 0.0, "p": round(p_sum / n, 4) if n else 0.0,
        "r": round(r_sum / n, 4) if n else 0.0, "mean_span_length": round(sum(span_lens) / len(span_lens), 3) if span_lens else 0.0,
        "max_span_length": max(span_lens) if span_lens else 0}

def _load_yolo_trocr(cfg: ExpConfig) -> tuple[Any, Any, Any, int, str]:
    """Load YOLO + TrOCR for canonical-347 oracle pass."""
    from models.eval_pipeline import _paths_from_config, _resolve_yolo_img
    paths = _paths_from_config(cfg)
    if not Path(paths.yolo).exists():
        raise EvalError(f"YOLO checkpoint not found at {paths.yolo}")
    if not Path(paths.trocr).exists():
        raise EvalError(f"TrOCR checkpoint not found at {paths.trocr}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise EvalError("ultralytics not installed") from exc
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    yolo, trocr_proc = YOLO(paths.yolo), TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(paths.trocr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    trocr_model.eval()
    return yolo, trocr_proc, trocr_model, _resolve_yolo_img(paths, cfg), device

def _tier_b_canonical_heuristic(test: list[Receipt], config: ExpConfig) -> dict[str, Any]:
    """Heuristic span-ceiling on canonical-347 using YOLO+TrOCR detected lines."""
    yolo, trocr_proc, trocr_model, yolo_img, device = _load_yolo_trocr(config)
    span_lens, n_zero, p_sum, r_sum, f1_sum, em_sum, n = [], 0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for rec in test:
            try:
                img = Image.open(rec.image_path).convert("RGB")
                texts, _, _ = _detect_and_read(yolo, trocr_proc, trocr_model, img, str(rec.image_path), config, yolo_img, device)
                if not texts:
                    texts, _, _ = _fallback_full_image(trocr_proc, trocr_model, img, config, device)
            except (OSError, RuntimeError, ValueError):
                log.exception("oracle: receipt %s detection failed", rec.image_path.stem)
                texts = []
            gold, (i_s, j_s, span_f1) = _gold_address(rec), _best_span(texts, _gold_address(rec))
            pred_text, length = (" ".join(texts[i_s:j_s + 1]), j_s - i_s + 1) if j_s >= i_s and texts else ("", 0)
            prec, rec_v, f1, em = _token_prf1(gold, pred_text)
            p_sum, r_sum, f1_sum, em_sum = p_sum + prec, r_sum + rec_v, f1_sum + f1, em_sum + (1.0 if em else 0.0)
            if length:
                span_lens.append(length)
            if span_f1 == 0.0:
                n_zero += 1
            n += 1
    return {"source": "canonical_347_yolo_trocr", "L_max": L_MAX, "n_receipts": n, "n_zero_match": n_zero,
        "f1": round(f1_sum / n, 4) if n else 0.0, "em": round(em_sum / n, 4) if n else 0.0,
        "p": round(p_sum / n, 4) if n else 0.0, "r": round(r_sum / n, 4) if n else 0.0,
        "mean_span_length": round(sum(span_lens) / len(span_lens), 3) if span_lens else 0.0, "max_span_length": max(span_lens) if span_lens else 0}

def run_skip_ceiling(data: DataSplit, config: ExpConfig) -> dict[str, Any]:
    """Run Tier A + Tier B oracle and return the JSON-serialisable sidecar."""
    log.info("Day-1 oracle: tier A (n=%d), tier B (n=%d)...", len(data.train) + len(data.val), len(data.test))
    tier_a = _tier_a_clean(list(data.train) + list(data.val), list(config.fields))
    tier_b = _tier_b_canonical_heuristic(list(data.test), config)
    log.info("A: n=%d f1=%.4f em=%.4f; B: n=%d f1=%.4f em=%.4f", tier_a["n_receipts"], tier_a["f1"], tier_a["em"], tier_b["n_receipts"], tier_b["f1"], tier_b["em"])
    f1, em = float(tier_b["f1"]), float(tier_b["em"])
    decision = "proceed_focus" if f1 >= _DECIDE_PROCEED_F1 and em >= _DECIDE_PROCEED_EM else ("borderline_ocr_bound" if f1 >= _DECIDE_ABORT_F1 else "abort_ocr_bottleneck")
    return {"run_id": Path(config.output_dir).name, "produced_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "tier_a_clean": tier_a, "tier_b_canonical_heuristic": tier_b, "p1_thresholds": {"f1": _P1_F1, "em": _P1_EM},
        "decision_thresholds": {"proceed_focus": "tier_b f1 >= 0.95 AND em >= 0.85", "borderline": "0.85 <= tier_b f1 < 0.95", "abort_ocr": "tier_b f1 < 0.85"},
        "decision": decision, "p1_passes": (f1 >= _P1_F1) and (em >= _P1_EM)}
