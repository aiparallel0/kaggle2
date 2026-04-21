"""DONUT Dataset and collator with decoder_input_ids shifting (Bug 2/9 safe).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: wraps SROIE receipts in <s_sroie>...<s_field>value</s_field>...</s_sroie>
    labels for DONUT training.  The collator supplies decoder_input_ids so
    HF Trainer's label-smoothing path does not crash the mBART decoder.
"""
from __future__ import annotations

import random
from typing import Any

from core.types import ExpConfig, Receipt

try:
    import numpy as np
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    _DATASET_BASE: type = torch.utils.data.Dataset
except ImportError:  # lightweight CI — torch/transformers not installed
    _DATASET_BASE = object


def _build_label(receipt: Receipt) -> str:
    """Wrap receipt fields in <s_sroie><s_field>value</s_field></s_sroie>."""
    parts = ["<s_sroie>"]
    for fld in receipt.fields:
        t = fld.name.lower()
        parts.append(f"<s_{t}>{fld.value}</s_{t}>")
    parts.append("</s_sroie>")
    return "".join(parts)


class _SROIEDataset(_DATASET_BASE):  # type: ignore[misc]
    def __init__(
        self, receipts: list[Receipt], processor: DonutProcessor, config: ExpConfig,
    ) -> None:
        self._r, self._p, self._c = receipts, processor, config

    def __len__(self) -> int:
        return len(self._r)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._r[idx]
        from PIL import Image
        img = Image.open(r.image_path).convert("RGB")
        pv = self._p(
            images=img, return_tensors="pt", legacy=False,
            size={"height": self._c.image_size[1], "width": self._c.image_size[0]},
        ).pixel_values.squeeze(0)
        tok = self._p.tokenizer(
            _build_label(r), max_length=self._c.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
            add_special_tokens=False,  # DONUT labels must not include mBART BOS/EOS
        )
        input_ids = tok.input_ids.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self._p.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels}


def _shift_right(labels: torch.Tensor, start_id: int, pad_id: int) -> torch.Tensor:
    """Shift labels right by one to produce decoder_input_ids (Bug 2 fallback)."""
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = start_id
    shifted[shifted == -100] = pad_id
    return shifted


class _DonutCollator:
    """Supplies decoder_input_ids so HF Trainer's label-smoothing works."""

    def __init__(self, model: VisionEncoderDecoderModel) -> None:
        self._model = model

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }
        # Replace -100 with pad before shifting — HF helper does not tolerate -100.
        labels_for_shift = batch["labels"].clone()
        pad_id = self._model.config.pad_token_id
        if pad_id is None:
            raise ValueError(
                "_DonutCollator: model.config.pad_token_id is None. Ensure "
                "train_donut sets it before the Trainer is created.",
            )
        labels_for_shift[labels_for_shift == -100] = pad_id
        if hasattr(self._model, "prepare_decoder_input_ids_from_labels"):
            batch["decoder_input_ids"] = (
                self._model.prepare_decoder_input_ids_from_labels(
                    labels=labels_for_shift,
                )
            )
        else:
            batch["decoder_input_ids"] = _shift_right(
                labels_for_shift,
                self._model.config.decoder_start_token_id,
                self._model.config.pad_token_id,
            )
        return batch


def _seed_worker(_worker_id: int) -> None:
    """Deterministic DataLoader workers (reproducibility guardrail)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
