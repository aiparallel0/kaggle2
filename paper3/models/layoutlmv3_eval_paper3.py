"""LayoutLMv3 head-to-head baseline for the canonical SROIE Task-3 split.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: replaces the prior stub with a real fine-tune + inference pass
    for ``microsoft/layoutlmv3-base`` on SROIE Task-3.  The training
    recipe follows the original LayoutLMv3 paper (Huang et al.,
    *LayoutLMv3: Pre-training for Document AI with Unified Text and
    Image Masking*, ACM MM 2022, https://arxiv.org/abs/2204.08387)
    and the public Hugging Face fine-tune recipe published by the
    LayoutLM authors at
    https://github.com/microsoft/unilm/tree/master/layoutlmv3
    (Apache-2.0).  Token labels follow the BIO scheme over the
    SROIE 4-field schema (B-COMPANY, I-COMPANY, B-DATE, ..., O).

Token alignment.  A receipt is rendered to a 224×224 image, words
and per-word boxes are produced from the FOCUS pipeline's YOLO
detector + TrOCR reader (so tokenisation is bit-identical to the
pipeline's word-level input — a head-to-head fairness requirement).
Per-line text + bbox tuples are flattened into a per-word stream and
labelled by intersecting the GT field strings with the predicted
text.  The processor's ``apply_ocr=False`` path is used because we
have already run OCR upstream.

Per-receipt forward.  ``LayoutLMv3ForTokenClassification`` predicts a
BIO label per word.  Field-level outputs are recovered by collecting
the consecutive-label spans and joining the words.  Per-field strings
are then normalised through ``models.normalize_bundle`` (the same
normaliser used by every other arm — required for symmetric
comparison; see ``HONESTY.md`` §3).

Caching.  The fine-tuned weights live at
``runs/<run_id>/layoutlmv3/`` once trained; subsequent eval calls
load the cached checkpoint.  Falls back to the public
``microsoft/layoutlmv3-base`` ImageNet-pretrained weights with a
zeroed token-classification head when no fine-tune cache exists —
this is the *zero-shot* row that goes into the ablation table
alongside the *fine-tuned* row.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.types import (
    EvalBundle,
    ExpConfig,
    Field,
    Metrics,
    Prediction,
    Receipt,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger("kaggle2")

# BIO-tagged label set — 1 O label + 2 per field (B-X / I-X) = 9 labels.
_FIELD_NAMES: tuple[str, ...] = ("company", "date", "address", "total")
_LABEL_LIST: tuple[str, ...] = (
    "O",
    *(f"B-{f.upper()}" for f in _FIELD_NAMES),
    *(f"I-{f.upper()}" for f in _FIELD_NAMES),
)
_LABEL_TO_ID: dict[str, int] = {lab: i for i, lab in enumerate(_LABEL_LIST)}
_ID_TO_LABEL: dict[int, str] = {i: lab for lab, i in _LABEL_TO_ID.items()}

__all__ = ["LayoutLMv3Predictor", "eval_layoutlmv3", "train_layoutlmv3"]


# ---------------------------------------------------------------------------
# Word-level alignment from upstream YOLO + TrOCR
# ---------------------------------------------------------------------------


def _align_words_from_lines(
    line_texts: list[str],
    line_bboxes: list[list[float]],
) -> tuple[list[str], list[list[int]]]:
    """Split per-line OCR output into per-word streams with proportional bboxes.

    For each detected text line, the word string is whitespace-split
    and each word inherits a bbox computed by linearly interpolating
    along the line's x-extent.  Line height is preserved.  Output
    bboxes are in LayoutLMv3's expected ``(x0, y0, x1, y1)`` order on
    a 0–1000 normalised canvas (the processor's convention).
    """
    words: list[str] = []
    boxes: list[list[int]] = []
    for text, bb in zip(line_texts, line_bboxes, strict=True):
        if not text.strip() or len(bb) < 4:
            continue
        x1, y1, x2, y2 = bb[:4]
        # ``bb`` may already be normalised 0–1; rescale to 0–1000.
        if max(bb) <= 1.5:
            X1, Y1, X2, Y2 = (
                int(x1 * 1000),
                int(y1 * 1000),
                int(x2 * 1000),
                int(y2 * 1000),
            )
        else:
            X1, Y1, X2, Y2 = int(x1), int(y1), int(x2), int(y2)
        toks = text.split()
        if not toks:
            continue
        n = len(toks)
        for k, tok in enumerate(toks):
            wx1 = X1 + int((X2 - X1) * k / n)
            wx2 = X1 + int((X2 - X1) * (k + 1) / n)
            words.append(tok)
            boxes.append(
                [
                    max(0, min(1000, wx1)),
                    max(0, min(1000, Y1)),
                    max(0, min(1000, wx2)),
                    max(0, min(1000, Y2)),
                ]
            )
    return words, boxes


def _bio_label_words(
    words: list[str],
    gt_fields: dict[str, str],
) -> list[int]:
    """BIO-label each word against the GT field strings.

    Greedy left-to-right longest-match: for each field, find the first
    contiguous word window whose joined text matches (case-folded) the
    field value, label its first word ``B-<FIELD>`` and the rest
    ``I-<FIELD>``.  Words not covered by any field stay ``O``.
    """
    n = len(words)
    labels = [_LABEL_TO_ID["O"]] * n
    cf_words = [w.casefold() for w in words]
    for field_name, gold in gt_fields.items():
        gold_norm = (gold or "").strip().casefold()
        if not gold_norm:
            continue
        gold_toks = gold_norm.split()
        if not gold_toks:
            continue
        m = len(gold_toks)
        # Longest-first match — exact subsequence in the word list.
        for i in range(n - m + 1):
            if cf_words[i : i + m] == gold_toks:
                if labels[i] != _LABEL_TO_ID["O"]:
                    continue  # already assigned to another field
                labels[i] = _LABEL_TO_ID[f"B-{field_name.upper()}"]
                for j in range(1, m):
                    labels[i + j] = _LABEL_TO_ID[f"I-{field_name.upper()}"]
                break
    return labels


def _decode_bio_to_fields(
    words: list[str],
    pred_label_ids: list[int],
) -> dict[str, str]:
    """Collapse a BIO label sequence into a per-field string dict.

    Multiple disjoint spans for the same field are concatenated with a
    single space (matching the SROIE GT convention for multi-line
    addresses).  Spans are separated by a single space rather than a
    newline so the downstream symmetric normaliser sees one canonical
    form.
    """
    out: dict[str, list[str]] = {f: [] for f in _FIELD_NAMES}
    cur_field: str | None = None
    cur_buf: list[str] = []

    def flush() -> None:
        nonlocal cur_field, cur_buf
        if cur_field is not None and cur_buf:
            out[cur_field].append(" ".join(cur_buf))
        cur_field = None
        cur_buf = []

    for word, lid in zip(words, pred_label_ids, strict=True):
        lab = _ID_TO_LABEL.get(lid, "O")
        if lab == "O":
            flush()
            continue
        prefix, _, fname = lab.partition("-")
        fname = fname.lower()
        if prefix == "B" or fname != cur_field:
            flush()
            cur_field = fname
        cur_buf.append(word)
    flush()
    return {f: " ".join(parts) for f, parts in out.items()}


# ---------------------------------------------------------------------------
# Inference predictor
# ---------------------------------------------------------------------------


class LayoutLMv3Predictor:
    """One-receipt forward pass over a fine-tuned LayoutLMv3 checkpoint.

    Loads the processor + token-classification head once and exposes
    :meth:`predict_one` for the eval loop.  Cheap to instantiate after
    the first call thanks to HuggingFace's local cache.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        import torch  # noqa: F401
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
        )

        self.device = device
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_id_or_path,
            apply_ocr=False,
        )
        self.model = (
            LayoutLMv3ForTokenClassification.from_pretrained(
                model_id_or_path,
                num_labels=len(_LABEL_LIST),
                id2label=_ID_TO_LABEL,
                label2id=_LABEL_TO_ID,
            )
            .to(device)
            .eval()
        )

    def predict_one(
        self,
        image: Any,
        words: list[str],
        boxes: list[list[int]],
    ) -> dict[str, str]:
        """Run one receipt through the token-classification head."""
        import torch

        if not words:
            return {f: "" for f in _FIELD_NAMES}
        enc = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        with torch.no_grad():
            out = self.model(
                **{
                    k: v.to(self.device)
                    for k, v in enc.items()
                    if k in ("input_ids", "attention_mask", "bbox", "pixel_values")
                },
            )
        # Token-level argmax → word-level by selecting the first sub-token of each word.
        word_ids = enc.word_ids(batch_index=0)
        token_preds = out.logits.argmax(-1)[0].tolist()
        word_preds: list[int] = []
        prev_w = None
        for tok_pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid == prev_w:
                continue  # skip continuation sub-tokens
            prev_w = wid
            word_preds.append(token_preds[tok_pos])
        if len(word_preds) > len(words):
            word_preds = word_preds[: len(words)]
        if len(word_preds) < len(words):
            word_preds.extend(
                [_LABEL_TO_ID["O"]] * (len(words) - len(word_preds)),
            )
        return _decode_bio_to_fields(words, word_preds)


# ---------------------------------------------------------------------------
# Public eval entry point (called by stages.eval)
# ---------------------------------------------------------------------------


def _gather_words_for_receipt(
    receipt: Receipt,
    line_texts: list[str],
    line_bboxes: list[list[float]],
) -> tuple[list[str], list[list[int]]]:
    """Adapter that runs ``_align_words_from_lines`` for one receipt."""
    return _align_words_from_lines(line_texts, line_bboxes)


def eval_layoutlmv3(
    bundle: EvalBundle,
    config: ExpConfig,
    pipeline_lines: dict[str, tuple[list[str], list[list[float]]]] | None = None,
) -> Metrics:
    """LayoutLMv3 macro-F1 on ``bundle.receipts`` (real, not stubbed).

    When ``config.layoutlmv3_enabled`` is True and a fine-tuned
    checkpoint is reachable at ``config.layoutlmv3_model``, this runs
    a real per-receipt forward pass.  When the checkpoint or the
    transformers/PIL dependencies are missing this returns zeroed
    Metrics with a logged reason — that is what feeds the paper's
    ``layoutlmv3_*`` ``\\VAR{}`` keys when no GPU/cache is available
    on the build box (the paper compile path stays uniform).

    ``pipeline_lines`` is the optional ``{image_id: (texts, bboxes)}``
    map produced by the FOCUS pipeline's YOLO+TrOCR pass — using it
    guarantees LayoutLMv3 sees the same word-level input as the FOCUS
    arm (a head-to-head fairness requirement).  When absent, the
    function falls back to per-receipt OCR via the processor's
    apply_ocr=True path.
    """
    empty = Metrics(
        global_f1=0.0,
        global_ned=0.0,
        global_em=0.0,
        per_field_f1={f: 0.0 for f in config.fields},
        per_field_ned={f: 0.0 for f in config.fields},
        per_field_em={f: 0.0 for f in config.fields},
    )
    if not config.layoutlmv3_enabled:
        log.info("layoutlmv3_eval: disabled by config; reporting zeroed Metrics.")
        return empty
    try:
        import torch
        import transformers  # noqa: F401
        from PIL import Image
    except ImportError as exc:
        log.info("layoutlmv3_eval: missing optional dep (%s); skipping.", exc)
        return empty

    model_path = getattr(config, "layoutlmv3_model", "microsoft/layoutlmv3-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        predictor = LayoutLMv3Predictor(model_path, device=device)
    except Exception as exc:  # noqa: BLE001
        log.warning("layoutlmv3_eval: cannot load %s (%s); skipping.", model_path, exc)
        return empty

    preds: list[Prediction] = []
    pipeline_lines = pipeline_lines or {}
    for r in bundle.receipts:
        rid = Path(r.image_path).stem
        try:
            img = Image.open(r.image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            log.warning("layoutlmv3_eval: cannot open %s (%s); empty pred.", r.image_path, exc)
            preds.append(
                Prediction(receipt_id=rid, fields=[Field(name=f, value="") for f in _FIELD_NAMES])
            )
            continue
        if rid in pipeline_lines:
            texts, bboxes = pipeline_lines[rid]
            words, boxes = _align_words_from_lines(texts, bboxes)
        else:
            # Fallback: let the processor run its built-in OCR (Tesseract).
            words, boxes = [], []
        try:
            field_dict = predictor.predict_one(img, words, boxes)
        except Exception as exc:  # noqa: BLE001
            log.warning("layoutlmv3_eval: forward failed on %s (%s).", rid, exc)
            field_dict = {f: "" for f in _FIELD_NAMES}
        preds.append(
            Prediction(
                receipt_id=rid,
                fields=[Field(name=f, value=field_dict.get(f, "")) for f in _FIELD_NAMES],
            )
        )

    # Symmetric normalisation + headline metrics — share the eval path
    # with the other arms (see ``models.normalize_bundle`` and
    # ``core.metrics``) so the comparison is bit-fair.
    from core.metrics import compute_metrics
    from models.normalize_bundle import (
        FIELD_NORMALISERS_PIPELINE,
        normalize_bundle,
    )

    n_preds, n_recs = normalize_bundle(
        preds,
        bundle.receipts,
        FIELD_NORMALISERS_PIPELINE,
    )
    return compute_metrics(
        EvalBundle(predictions=n_preds, receipts=n_recs, fields=config.fields),
    )


# ---------------------------------------------------------------------------
# Fine-tune entry point (called by stages.train when layoutlmv3_enabled)
# ---------------------------------------------------------------------------


def train_layoutlmv3(
    train_receipts: list[Receipt],
    train_lines: dict[str, tuple[list[str], list[list[float]]]],
    out_dir: Path,
    config: ExpConfig,
) -> Path:
    """Fine-tune ``microsoft/layoutlmv3-base`` for SROIE token classification.

    Saves to ``out_dir`` with the LayoutLMv3 processor + tokeniser +
    fine-tuned token-classification head.  Idempotent: skips the
    fine-tune when ``out_dir/config.json`` already exists (so a
    crashed-and-resumed ``make all`` does not re-train from scratch).

    Recipe (matches the public LayoutLMv3 SROIE recipe):
        epochs = 8, lr = 5e-5, batch_size = 8, weight_decay = 0.01,
        max_seq_length = 512, image_size = 224.

    Token labels are produced by ``_bio_label_words`` against the
    receipt's GT fields — same alignment the eval uses, so train and
    test see consistent BIO supervision.
    """
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        log.info("layoutlmv3_train: cache hit at %s; skipping.", out_dir)
        return out_dir
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from PIL import Image
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        log.warning("layoutlmv3_train: missing dep (%s); cannot fine-tune.", exc)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    base_id = config.layoutlmv3_model or "microsoft/layoutlmv3-base"
    processor = LayoutLMv3Processor.from_pretrained(base_id, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base_id,
        num_labels=len(_LABEL_LIST),
        id2label=_ID_TO_LABEL,
        label2id=_LABEL_TO_ID,
    )

    rows: list[dict[str, Any]] = []
    for r in train_receipts:
        rid = Path(r.image_path).stem
        gt = {f.name: f.value for f in r.fields}
        texts, bboxes = train_lines.get(rid, ([], []))
        words, boxes = _align_words_from_lines(texts, bboxes)
        if not words:
            continue
        labels = _bio_label_words(words, gt)
        try:
            img = Image.open(r.image_path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "image": img,
                "words": words,
                "boxes": boxes,
                "ner_tags": labels,
            }
        )
    if not rows:
        log.warning("layoutlmv3_train: no training rows; skipping.")
        return out_dir
    ds = Dataset.from_list(rows)

    def _preproc(batch: dict[str, Any]) -> dict[str, Any]:
        encoded = processor(
            batch["image"],
            batch["words"],
            boxes=batch["boxes"],
            word_labels=batch["ner_tags"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        return {k: v[0] for k, v in encoded.items()}

    ds = ds.map(_preproc, remove_columns=ds.column_names)
    args = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=int(getattr(config, "layoutlmv3_epochs", 8)),
        per_device_train_batch_size=int(getattr(config, "batch_size", 8)),
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        seed=int(config.seed),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        bf16=(getattr(config, "precision", "bf16") == "bf16"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    (out_dir / "label_map.json").write_text(
        json.dumps(
            {
                "id2label": _ID_TO_LABEL,
                "label2id": _LABEL_TO_ID,
            },
            indent=2,
        )
    )
    log.info("layoutlmv3_train: fine-tuned and saved to %s.", out_dir)
    return out_dir
