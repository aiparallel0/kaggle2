"""Load and validate config.json → ExpConfig."""
from __future__ import annotations

import json
from typing import Any

from core.types import ExpConfig


_REQUIRED = [
    "seed", "base_model", "trocr_model", "yolo_model", "image_size",
    "yolo_img_size", "max_length", "trocr_max_len", "epochs_donut",
    "epochs_yolo", "epochs_trocr", "batch_size", "grad_accum", "lr",
    "warmup_steps", "weight_decay", "label_smoothing", "precision",
    "patience", "max_grad_norm", "fields", "new_tokens", "sroie_url",
    "data_dir", "output_dir", "paper_template", "paper_output",
]


def load_config(path: str, defaults: dict[str, Any] | None = None) -> ExpConfig:
    """Load config.json, merge defaults, validate required keys → ExpConfig."""
    raw: dict[str, Any] = {}
    if defaults:
        raw.update(defaults)
    with open(path) as f:
        raw.update(json.load(f))

    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"config.json missing required keys: {missing}")

    if raw["epochs_trocr"] < 5:  # Bug 6 prevention
        raise ValueError(
            f"epochs_trocr={raw['epochs_trocr']} < 5 — "
            "TrOCR will underfit (Bug 6). Set epochs_trocr >= 5."
        )

    known = set(_REQUIRED)
    extra = {k: v for k, v in raw.items() if k not in known}

    img = raw["image_size"]
    return ExpConfig(
        seed=int(raw["seed"]),
        base_model=str(raw["base_model"]),
        trocr_model=str(raw["trocr_model"]),
        yolo_model=str(raw["yolo_model"]),
        image_size=(int(img[0]), int(img[1])),
        yolo_img_size=int(raw["yolo_img_size"]),
        max_length=int(raw["max_length"]),
        trocr_max_len=int(raw["trocr_max_len"]),
        epochs_donut=int(raw["epochs_donut"]),
        epochs_yolo=int(raw["epochs_yolo"]),
        epochs_trocr=int(raw["epochs_trocr"]),
        batch_size=int(raw["batch_size"]),
        grad_accum=int(raw["grad_accum"]),
        lr=float(raw["lr"]),
        warmup_steps=int(raw["warmup_steps"]),
        weight_decay=float(raw["weight_decay"]),
        label_smoothing=float(raw["label_smoothing"]),
        precision=str(raw["precision"]),
        patience=int(raw["patience"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        fields=list(raw["fields"]),
        new_tokens=list(raw["new_tokens"]),
        sroie_url=str(raw["sroie_url"]),
        data_dir=str(raw["data_dir"]),
        output_dir=str(raw["output_dir"]),
        paper_template=str(raw["paper_template"]),
        paper_output=str(raw["paper_output"]),
        extra=extra,
    )
