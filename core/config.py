"""Load experiment hyperparameters and validate against silent F1-destroying bugs.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: single source of truth for every hyperparameter surfaced in the
    paper (epochs, batch size, differential LR, warmup, label smoothing,
    precision, patience). Encodes two Bug-class guardrails at load time:
    Bug 6 (TrOCR undertrained) and Bug 4 (fp16 NaN without grad clip).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.errors import TrainError
from core.runlayout import derive_paths
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
    """Parse config.json into a typed ExpConfig, enforcing Bug 4 and Bug 6 guardrails."""
    raw: dict[str, Any] = {}
    if defaults:
        raw.update(defaults)
    with open(path) as f:
        raw.update(json.load(f))

    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"config.json missing required keys: {missing}")

    bug_flags = _parse_bug_flags(raw.get("bug_flags"))
    if bug_flags.get("bug_6", True) and raw["epochs_trocr"] < 5:  # Bug 6 prevention
        raise TrainError(
            f"epochs_trocr={raw['epochs_trocr']} < 5 — "
            "TrOCR will underfit (Bug 6). Set epochs_trocr >= 5.",
        )

    # Bug 4: fp16 without gradient clipping overflows → NaN loss. bf16 is safe
    # because its dynamic range matches fp32. Enforce the invariant at load
    # time so a stale config surfaces the error before any GPU work starts.
    if (
        bug_flags.get("bug_4", True)
        and raw["precision"] == "fp16"
        and float(raw["max_grad_norm"]) <= 0.0
    ):
        raise TrainError(
            "precision='fp16' requires max_grad_norm > 0 to prevent "
            "loss=NaN from gradient overflow (Bug 4). Set max_grad_norm=1.0 "
            "or switch to precision='bf16' on Ampere+ GPUs.",
        )

    _optional = {
        "yolo_conf", "trocr_max_new_tokens", "max_regions_per_image",
        "warmup_ratio", "lr_scheduler_type", "gradient_checkpointing",
        "num_beams", "expected_f1_warn", "skip_donut",
        "assigner_hidden", "assigner_n_layers_level2",
        "emit_hidden", "emit_vocab_size", "emit_max_len", "emit_beam_width",
        "kd_attn_weight", "kd_logits_weight",
        "assigner_patience", "assigner_min_delta",
        "weight_decay_assigner", "dropout_assigner", "priors_v2",
        "seeds", "n_trials", "bootstrap_n_iter", "bootstrap_ci_level",
        "address_accept_fraction", "regex_router", "text_pool_learned",
        "total_confidence_threshold",
        "lr_assigner", "warmup_ratio_assigner",
        "bug_flags",  # P1 — 13-bug ablation gating dict
        "rag_enabled", "rag_k",  # P2 — retrieval-augmented DONUT
        "gat_enabled",  # P3 — graph-attention assigner opt-in
        "foundation_enabled", "foundation_api", "foundation_cache_path",  # P4
        "runs_root", "run_id",  # runlayout keys (optional; back-compat).
        # v4 — canonical SROIE / LayoutLMv3 / latency / curation keys.
        "canonical_sroie_enabled", "canonical_sroie_test_path",
        "canonical_sroie_test_url", "canonical_sroie_gt_url",
        "canonical_sroie_mirror_url", "paper_variant",
        "layoutlmv3_enabled", "layoutlmv3_model",
        "measure_latency",
        "qualitative_sample_ids", "fig1_receipt_id",
        "strict_paper",
    }
    known = set(_REQUIRED) | _optional
    extra = {k: v for k, v in raw.items() if k not in known}

    img = raw["image_size"]
    # `seeds` (a list) is authoritative; `seed` is legacy scalar back-compat.
    # `n_trials` defaults to len(seeds) unless explicitly truncated.
    raw_seeds = raw.get("seeds")
    if isinstance(raw_seeds, list) and raw_seeds:
        seeds_list = [int(s) for s in raw_seeds]
    else:
        seeds_list = [int(raw["seed"])]
    raw_n = raw.get("n_trials")
    n_trials = int(raw_n) if raw_n is not None else len(seeds_list)
    if n_trials > len(seeds_list):
        raise TrainError(
            f"n_trials={n_trials} exceeds len(seeds)={len(seeds_list)}; "
            "extend the `seeds` list or reduce `n_trials`.",
        )
    # Paper-variant template selection.  The repo ships:
    #   * ``report/template.tex``           — basic 500/63/63 baseline
    #     study (current default content; serves as the fallback).
    #   * ``report/template_advanced.tex``  — 626-train + 347-canonical-
    #     test variant (DONUT vs YOLO+TrOCR+Attention, no GT-OCR arm).
    #   * ``report/template_basic.tex``     — explicit alias for the
    #     basic variant; falls back to ``template.tex`` when absent.
    # Env override ``KAGGLE2_PAPER_VARIANT`` wins so CLI --paper-variant
    # can flip the choice without editing config.json.
    paper_variant = (
        os.environ.get("KAGGLE2_PAPER_VARIANT")
        or str(raw.get("paper_variant", "advanced"))
    )
    template_path = str(raw["paper_template"])
    candidate = Path(template_path).with_name(f"template_{paper_variant}.tex")
    if candidate.exists():
        template_path = str(candidate)
    # Route output_dir + paper_output through runs_root/run_id layout
    # when configured (back-compat: raw config values survive unchanged).
    output_dir, paper_output = derive_paths(
        str(raw["output_dir"]), str(raw["paper_output"]),
        raw.get("runs_root"), raw.get("run_id"),
        Path(path).resolve().parent,
    )
    return ExpConfig(
        seed=int(raw["seed"]),
        base_model=str(raw["base_model"]), trocr_model=str(raw["trocr_model"]),
        yolo_model=str(raw["yolo_model"]),
        image_size=(int(img[0]), int(img[1])),
        yolo_img_size=int(raw["yolo_img_size"]),
        max_length=int(raw["max_length"]), trocr_max_len=int(raw["trocr_max_len"]),
        epochs_donut=int(raw["epochs_donut"]), epochs_yolo=int(raw["epochs_yolo"]),
        epochs_trocr=int(raw["epochs_trocr"]),
        epochs_assigner=int(raw["epochs_assigner"]),
        batch_size=int(raw["batch_size"]), grad_accum=int(raw["grad_accum"]),
        lr=float(raw["lr"]), lr_decoder=float(raw["lr_decoder"]),
        warmup_steps=int(raw["warmup_steps"]),
        weight_decay=float(raw["weight_decay"]),
        label_smoothing=float(raw["label_smoothing"]),
        precision=str(raw["precision"]), patience=int(raw["patience"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        fields=list(raw["fields"]), new_tokens=list(raw["new_tokens"]),
        sroie_url=str(raw["sroie_url"]), data_dir=str(raw["data_dir"]),
        output_dir=output_dir,
        paper_template=template_path, paper_output=paper_output,
        yolo_conf=float(raw.get("yolo_conf", 0.25)),
        trocr_max_new_tokens=int(raw.get("trocr_max_new_tokens", 64)),
        max_regions_per_image=int(raw.get("max_regions_per_image", 32)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(raw.get("lr_scheduler_type", "cosine")),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", True)),
        num_beams=int(raw.get("num_beams", 4)),
        expected_f1_warn=float(raw.get("expected_f1_warn", 0.75)),
        skip_donut=bool(raw.get("skip_donut", False)),
        assigner_hidden=int(raw.get("assigner_hidden", 384)),
        assigner_n_layers_level2=int(raw.get("assigner_n_layers_level2", 6)),
        emit_hidden=int(raw.get("emit_hidden", 128)),
        emit_vocab_size=int(raw.get("emit_vocab_size", 259)),
        emit_max_len=int(raw.get("emit_max_len", 96)),
        emit_beam_width=int(raw.get("emit_beam_width", 4)),
        kd_attn_weight=float(raw.get("kd_attn_weight", 0.0)),
        kd_logits_weight=float(raw.get("kd_logits_weight", 0.0)),
        assigner_patience=int(raw.get("assigner_patience", 7)),
        assigner_min_delta=float(raw.get("assigner_min_delta", 0.005)),
        weight_decay_assigner=float(raw.get("weight_decay_assigner", 5e-4)),
        dropout_assigner=float(raw.get("dropout_assigner", 0.2)),
        priors_v2=bool(raw.get("priors_v2", True)),
        seeds=seeds_list, n_trials=n_trials,
        bootstrap_n_iter=int(raw.get("bootstrap_n_iter", 1000)),
        bootstrap_ci_level=float(raw.get("bootstrap_ci_level", 0.95)),
        address_accept_fraction=float(raw.get("address_accept_fraction", 0.5)),
        regex_router=bool(raw.get("regex_router", True)),
        text_pool_learned=bool(raw.get("text_pool_learned", False)),
        total_confidence_threshold=float(raw.get("total_confidence_threshold", 0.55)),
        lr_assigner=float(raw.get("lr_assigner", 1e-3)),
        warmup_ratio_assigner=float(raw.get("warmup_ratio_assigner", 0.0)),
        bug_flags=bug_flags,
        rag_enabled=bool(raw.get("rag_enabled", False)),
        rag_k=int(raw.get("rag_k", 3)),
        gat_enabled=bool(raw.get("gat_enabled", False)),
        foundation_enabled=bool(raw.get("foundation_enabled", False)),
        foundation_api=str(raw.get("foundation_api", "anthropic")),
        foundation_cache_path=str(
            raw.get("foundation_cache_path", "./runs/foundation_cache.json")),
        canonical_sroie_enabled=bool(raw.get("canonical_sroie_enabled", False)),
        canonical_sroie_test_path=str(raw.get("canonical_sroie_test_path", "")),
        canonical_sroie_test_url=str(raw.get(
            "canonical_sroie_test_url",
            "https://rrc.cvc.uab.es/downloads/SROIE_test_images_task_3.zip",
        )),
        canonical_sroie_gt_url=str(raw.get(
            "canonical_sroie_gt_url",
            "https://rrc.cvc.uab.es/downloads/SROIE_test_gt_task_3.zip",
        )),
        canonical_sroie_mirror_url=str(raw.get(
            "canonical_sroie_mirror_url",
            "https://doctr-static.mindee.com/models?id=v0.1.1/sroie2019_test.zip&src=0",
        )),
        paper_variant=str(raw.get("paper_variant", "advanced")),
        layoutlmv3_enabled=bool(raw.get("layoutlmv3_enabled", False)),
        layoutlmv3_model=str(raw.get("layoutlmv3_model", "microsoft/layoutlmv3-base")),
        measure_latency=bool(raw.get("measure_latency", False)),
        qualitative_sample_ids=[
            str(x) for x in raw.get("qualitative_sample_ids", []) or []
        ],
        fig1_receipt_id=str(raw.get("fig1_receipt_id", "")),
        strict_paper=bool(raw.get("strict_paper", False)),
        extra=extra,
    )


def _parse_bug_flags(raw: object) -> dict[str, bool]:
    """Coerce ``raw`` into a {bug_1..bug_13: bool} dict; defaults True."""
    out = {f"bug_{i}": True for i in range(1, 14)}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str) and k in out:
                out[k] = bool(v)
    return out
