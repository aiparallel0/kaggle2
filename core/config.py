"""Load and validate config.json → ExpConfig."""
from __future__ import annotations

import json
from typing import Any

from core.errors import TrainError
from core.types import ExpConfig

_REQUIRED = [
    "seed", "base_model", "trocr_model", "yolo_model", "image_size",
    "yolo_img_size", "max_length", "trocr_max_len", "epochs_donut",
    "epochs_yolo", "epochs_trocr", "epochs_assigner", "batch_size",
    "grad_accum", "lr", "lr_decoder", "warmup_steps", "weight_decay",
    "label_smoothing", "precision", "patience", "max_grad_norm",
    "fields", "new_tokens", "sroie_url", "data_dir", "output_dir",
    "paper_template", "paper_output",
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
        raise TrainError(
            f"epochs_trocr={raw['epochs_trocr']} < 5 — "
            "TrOCR will underfit (Bug 6). Set epochs_trocr >= 5.",
        )

    # Bug 4: fp16 without gradient clipping overflows → NaN loss. bf16 is safe
    # because its dynamic range matches fp32. Enforce the invariant at load
    # time so a stale config surfaces the error before any GPU work starts.
    if raw["precision"] == "fp16" and float(raw["max_grad_norm"]) <= 0.0:
        raise TrainError(
            "precision='fp16' requires max_grad_norm > 0 to prevent "
            "loss=NaN from gradient overflow (Bug 4). Set max_grad_norm=1.0 "
            "or switch to precision='bf16' on Ampere+ GPUs.",
        )

    _optional = {
        "yolo_conf", "trocr_max_new_tokens", "max_regions_per_image",
        "warmup_ratio", "lr_scheduler_type", "gradient_checkpointing",
        "num_beams", "expected_f1_warn",
        "skip_donut",
        "assigner_hidden", "assigner_n_layers_level2",
        "emit_hidden", "emit_vocab_size", "emit_max_len", "emit_beam_width",
        "kd_attn_weight", "kd_logits_weight",
    }
    known = set(_REQUIRED) | _optional
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
        epochs_assigner=int(raw["epochs_assigner"]),
        batch_size=int(raw["batch_size"]),
        grad_accum=int(raw["grad_accum"]),
        lr=float(raw["lr"]),
        lr_decoder=float(raw["lr_decoder"]),
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
        yolo_conf=float(raw.get("yolo_conf", 0.25)),
        trocr_max_new_tokens=int(raw.get("trocr_max_new_tokens", 64)),
        max_regions_per_image=int(raw.get("max_regions_per_image", 32)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(raw.get("lr_scheduler_type", "cosine")),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", True)),
        num_beams=int(raw.get("num_beams", 4)),
        expected_f1_warn=float(raw.get("expected_f1_warn", 0.75)),
        skip_donut=bool(raw.get("skip_donut", False)),
        assigner_hidden=int(raw.get("assigner_hidden", 192)),
        assigner_n_layers_level2=int(raw.get("assigner_n_layers_level2", 3)),
        emit_hidden=int(raw.get("emit_hidden", 128)),
        emit_vocab_size=int(raw.get("emit_vocab_size", 259)),
        emit_max_len=int(raw.get("emit_max_len", 96)),
        emit_beam_width=int(raw.get("emit_beam_width", 4)),
        kd_attn_weight=float(raw.get("kd_attn_weight", 0.0)),
        kd_logits_weight=float(raw.get("kd_logits_weight", 0.0)),
        extra=extra,
    )
