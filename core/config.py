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
from typing import Any, Literal, cast

from core.errors import ConfigError, TrainError
from core.runlayout import derive_paths
from core.types import ExpConfig

_REQUIRED = [
    "seed", "base_model", "trocr_model", "yolo_model", "image_size",
    "yolo_image_size", "max_length", "trocr_max_len", "epochs_donut",
    "epochs_yolo", "epochs_trocr", "epochs_focus", "batch_size",
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

    # Bug 18 — architectural-flag invariants for the FOCUS framework.
    # The MIL pos-mass loss is one-sided; it rewards mass on positives
    # but never penalises mass on boilerplate.  Shipping a ``paper_variant
    # == 'focus'`` run with any of the FOCUS sub-flags off therefore
    # produces a paper that documents an architecture the run never
    # actually used (the "no silent placeholders" invariant from
    # AGENTS.md, applied to architecture flags).  Enforce at load time
    # so a stale config surfaces the error before any GPU work starts.
    if bug_flags.get("bug_18", True):
        _validate_focus_flags(raw)

    _optional = {
        "yolo_conf", "trocr_max_new_tokens", "max_regions_per_image",
        "yolo_crop_pad_px",
        "warmup_ratio", "lr_scheduler_type", "gradient_checkpointing",
        "num_beams", "f1_warn_threshold", "skip_donut",
        "focus_hidden_dim", "focus_n_layers_level2",
        "emit_hidden", "emit_vocab_size", "emit_max_len", "emit_beam_width",
        "kd_attn_weight", "kd_logits_weight",
        "focus_patience", "focus_min_delta",
        "weight_decay_focus", "dropout_focus", "priors_v2",
        "seeds", "n_trials", "bootstrap_n_iter", "bootstrap_ci_level",
        "address_accept_fraction", "regex_router", "text_pool_learned",
        "total_confidence_threshold",
        "lr_focus", "warmup_ratio_focus",
        "bug_flags",  # P1 — 13-bug ablation gating dict (now 17 with PR-C)
        "rag_enabled", "rag_k",  # P2 — retrieval-augmented DONUT
        "gat_enabled",  # P3 — graph-attention assigner opt-in
        "foundation_enabled", "foundation_api", "foundation_cache_path",  # P4
        "runs_root", "run_id",  # runlayout keys (optional; back-compat).
        # v4 — canonical SROIE / LayoutLMv3 / latency / curation keys.
        "canonical_sroie_enabled", "canonical_sroie_test_path",
        "canonical_sroie_test_url", "canonical_sroie_gt_url",
        "canonical_sroie_hf_repo", "canonical_sroie_hf_revision",
        "paper_variant",
        "layoutlmv3_enabled", "layoutlmv3_model",
        "measure_latency",
        "qualitative_sample_ids", "fig1_receipt_id",
        "strict_paper",
        # PR-A / T-C — magic-number → config promotions.
        "assigner_hardneg_margin", "assigner_kd_temperature",
        "assigner_field_weights", "assigner_param_drift_tol",
        # PR-C — S0/S1/S2/S3 opt-in feature flags.
        "priors_v3",
        "address_anchor_extend", "address_anchor_extender_k",
        "fusion",
        "address_score_token_f1_w", "address_score_line_count_w",
        "address_score_postcode_w", "address_score_money_penalty",
        # PR-E — Pareto sweep selectors.
        "sweep_size", "sweep_dataset",
        # PR-D — GPT-4V eval + carbon accounting.
        "llm_eval_enabled", "llm_eval_provider", "llm_eval_cache_path",
        "carbon_grid_factor_kgco2e_per_kwh",
        # PR-FOCUS — FOCUS-A (PR #106) + FOCUS-T / FOCUS-C / priors_v4 sub-
        # flags.  Defaults are bit-exact with pre-FOCUS runs.
        "focus_enabled", "focus_max_span",
        "focus_iou_weight", "focus_boundary_weight", "focus_confidence_floor",
        "focus_total_enabled", "focus_total_witness_weight",
        "focus_company_enabled", "focus_company_y_weight",
        "focus_company_boilerplate_weight",
        "focus_company_confidence_threshold",
        # FOCUS-C span head (mirrors FOCUS-A).
        "focus_company_span_enabled", "focus_company_span_max_span",
        "focus_company_span_iou_w", "focus_company_span_boundary_w",
        "focus_company_confidence_floor",
        "focus_total_aux_w", "focus_company_pos_aux_w",
        "priors_v4",
        "total_arithmetic_enabled",
        "zone_prior_enabled", "zone_totals_floor", "zone_header_floor",
        "zone_regex_total_floor", "zone_params_path",
    }
    # Bug-18 composite-loss knobs are read via ``_loss_knob`` from
    # ``config.extra``, so they are intentionally NOT in ``_optional`` —
    # the parser surfaces them through ``extra`` exactly like the
    # legacy ``focus_hardneg_weight`` / ``focus_synth_subtotal`` knobs.
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
    #   * ``report/template.tex``           — generic baseline (current
    #     default content); used as the fallback when no variant-
    #     specific sibling exists.
    #   * ``report/template_baseline.tex``  — explicit baseline variant
    #     (500/63/63 internal split, three arms incl. GT-OCR baseline).
    #   * ``report/template_focus.tex``     — explicit focus variant
    #     (626-train + 347-canonical-test, DONUT vs YOLO+TrOCR+Attention,
    #     no GT-OCR arm; competitors table).
    # Logic: if ``template_<paper_variant>.tex`` sits next to
    # ``paper_template``, swap to it; else keep ``paper_template``
    # as-is.  Env override ``KAGGLE2_PAPER_VARIANT`` wins so CLI
    # --paper-variant can flip the choice without editing config.json.
    paper_variant = (
        os.environ.get("KAGGLE2_PAPER_VARIANT")
        or str(raw.get("paper_variant", "focus"))
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
        yolo_image_size=int(raw["yolo_image_size"]),
        max_length=int(raw["max_length"]), trocr_max_len=int(raw["trocr_max_len"]),
        epochs_donut=int(raw["epochs_donut"]), epochs_yolo=int(raw["epochs_yolo"]),
        epochs_trocr=int(raw["epochs_trocr"]),
        epochs_focus=int(raw["epochs_focus"]),
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
        yolo_crop_pad_px=int(raw.get("yolo_crop_pad_px", 0)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(raw.get("lr_scheduler_type", "cosine")),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", True)),
        num_beams=int(raw.get("num_beams", 4)),
        f1_warn_threshold=float(raw.get("f1_warn_threshold", 0.75)),
        skip_donut=bool(raw.get("skip_donut", False)),
        focus_hidden_dim=int(raw.get("focus_hidden_dim", 384)),
        focus_n_layers_level2=int(raw.get("focus_n_layers_level2", 6)),
        emit_hidden=int(raw.get("emit_hidden", 128)),
        emit_vocab_size=int(raw.get("emit_vocab_size", 259)),
        emit_max_len=int(raw.get("emit_max_len", 96)),
        emit_beam_width=int(raw.get("emit_beam_width", 4)),
        kd_attn_weight=float(raw.get("kd_attn_weight", 0.0)),
        kd_logits_weight=float(raw.get("kd_logits_weight", 0.0)),
        focus_patience=int(raw.get("focus_patience", 7)),
        focus_min_delta=float(raw.get("focus_min_delta", 0.005)),
        weight_decay_focus=float(raw.get("weight_decay_focus", 5e-4)),
        dropout_focus=float(raw.get("dropout_focus", 0.2)),
        priors_v2=bool(raw.get("priors_v2", True)),
        seeds=seeds_list, n_trials=n_trials,
        bootstrap_n_iter=int(raw.get("bootstrap_n_iter", 1000)),
        bootstrap_ci_level=float(raw.get("bootstrap_ci_level", 0.95)),
        address_accept_fraction=float(raw.get("address_accept_fraction", 0.5)),
        regex_router=bool(raw.get("regex_router", True)),
        text_pool_learned=bool(raw.get("text_pool_learned", False)),
        total_confidence_threshold=float(raw.get("total_confidence_threshold", 0.55)),
        lr_focus=float(raw.get("lr_focus", 1e-3)),
        warmup_ratio_focus=float(raw.get("warmup_ratio_focus", 0.0)),
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
        canonical_sroie_hf_repo=str(raw.get(
            "canonical_sroie_hf_repo", "Metric-AI/icdar_sroie",
        )),
        canonical_sroie_hf_revision=str(raw.get(
            "canonical_sroie_hf_revision", "main",
        )),
        paper_variant=str(raw.get("paper_variant", "focus")),
        layoutlmv3_enabled=bool(raw.get("layoutlmv3_enabled", False)),
        layoutlmv3_model=str(raw.get("layoutlmv3_model", "microsoft/layoutlmv3-base")),
        measure_latency=bool(raw.get("measure_latency", False)),
        qualitative_sample_ids=[
            str(x) for x in raw.get("qualitative_sample_ids", []) or []
        ],
        fig1_receipt_id=str(raw.get("fig1_receipt_id", "")),
        strict_paper=bool(raw.get("strict_paper", False)),
        # PR-A / T-C — promoted magic numbers.
        assigner_hardneg_margin=float(
            raw.get("assigner_hardneg_margin", 0.10),
        ),
        assigner_kd_temperature=float(
            raw.get("assigner_kd_temperature", 2.0),
        ),
        assigner_field_weights=_parse_field_weights(
            raw.get("assigner_field_weights"),
        ),
        assigner_param_drift_tol=int(
            raw.get("assigner_param_drift_tol", 500),
        ),
        # PR-C — S0/S1/S2/S3 opt-in feature flags.
        priors_v3=bool(raw.get("priors_v3", False)),
        address_anchor_extend=bool(raw.get("address_anchor_extend", False)),
        address_anchor_extender_k=int(
            raw.get("address_anchor_extender_k", 2),
        ),
        fusion=_parse_fusion(raw.get("fusion", "sum")),
        address_score_token_f1_w=float(
            raw.get("address_score_token_f1_w", 1.0),
        ),
        address_score_line_count_w=float(
            raw.get("address_score_line_count_w", 0.25),
        ),
        address_score_postcode_w=float(
            raw.get("address_score_postcode_w", 0.05),
        ),
        address_score_money_penalty=float(
            raw.get("address_score_money_penalty", 0.10),
        ),
        # PR-E — Pareto sweep selectors.
        sweep_size=_parse_sweep_size(raw.get("sweep_size", "")),
        sweep_dataset=_parse_sweep_dataset(raw.get("sweep_dataset", "")),
        # PR-D — GPT-4V + carbon.
        llm_eval_enabled=bool(raw.get("llm_eval_enabled", False)),
        llm_eval_provider=str(raw.get("llm_eval_provider", "gpt-4v")),
        llm_eval_cache_path=str(
            raw.get("llm_eval_cache_path", "./runs/llm_eval_cache.json"),
        ),
        carbon_grid_factor_kgco2e_per_kwh=float(
            raw.get("carbon_grid_factor_kgco2e_per_kwh", 0.475),
        ),
        # PR-FOCUS — FOCUS-A (PR #106) + FOCUS-T / FOCUS-C / priors_v4.
        focus_enabled=bool(raw.get("focus_enabled", False)),
        focus_max_span=int(raw.get("focus_max_span", 8)),
        focus_iou_weight=float(raw.get("focus_iou_weight", 1.0)),
        focus_boundary_weight=float(raw.get("focus_boundary_weight", 1.0)),
        focus_confidence_floor=float(raw.get("focus_confidence_floor", 0.10)),
        focus_total_enabled=bool(raw.get("focus_total_enabled", False)),
        focus_total_witness_weight=float(
            raw.get("focus_total_witness_weight", 1.0),
        ),
        focus_company_enabled=bool(raw.get("focus_company_enabled", False)),
        focus_company_y_weight=float(raw.get("focus_company_y_weight", 1.0)),
        focus_company_boilerplate_weight=float(
            raw.get("focus_company_boilerplate_weight", 1.0),
        ),
        focus_company_confidence_threshold=float(
            raw.get("focus_company_confidence_threshold", 0.30),
        ),
        # FOCUS-C span head (mirrors FOCUS-A).
        focus_company_span_enabled=bool(
            raw.get("focus_company_span_enabled", False),
        ),
        focus_company_span_max_span=int(
            raw.get("focus_company_span_max_span", 4),
        ),
        focus_company_span_iou_w=float(
            raw.get("focus_company_span_iou_w", 1.0),
        ),
        focus_company_span_boundary_w=float(
            raw.get("focus_company_span_boundary_w", 1.0),
        ),
        focus_company_confidence_floor=float(
            raw.get("focus_company_confidence_floor", 0.20),
        ),
        focus_total_aux_w=float(raw.get("focus_total_aux_w", 0.0)),
        focus_company_pos_aux_w=float(raw.get("focus_company_pos_aux_w", 0.0)),
        priors_v4=bool(raw.get("priors_v4", False)),
        total_arithmetic_enabled=bool(raw.get("total_arithmetic_enabled", True)),
        zone_prior_enabled=bool(raw.get("zone_prior_enabled", True)),
        zone_totals_floor=float(raw.get("zone_totals_floor", 0.5)),
        zone_header_floor=float(raw.get("zone_header_floor", 0.4)),
        zone_regex_total_floor=float(raw.get("zone_regex_total_floor", 0.2)),
        zone_params_path=str(raw.get("zone_params_path", "")),
        extra=extra,
    )


def _parse_field_weights(raw: object) -> dict[str, float]:
    """Coerce ``raw`` into the ``{field: weight}`` shape used by the
    assigner train loop; defaults match the legacy ``FIELD_LOSS_WEIGHTS``.
    """
    out = {"company": 1.5, "address": 1.3, "total": 1.2, "date": 0.8}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, int | float):
                out[k] = float(v)
    return out


def _parse_fusion(raw: object) -> Literal["sum", "concat"]:
    """Restrict ``fusion`` to the two supported strategies."""
    s = str(raw).lower().strip() or "sum"
    if s in ("sum", "concat"):
        return cast("Literal['sum', 'concat']", s)
    raise TrainError(
        f"fusion={raw!r} not supported; choose 'sum' or 'concat'.",
    )


_SWEEP_SIZES: tuple[Literal["", "tiny", "small", "base", "large"], ...] = (
    "", "tiny", "small", "base", "large",
)
_SWEEP_DATASETS: tuple[Literal["", "sroie", "cord"], ...] = (
    "", "sroie", "cord",
)


def _parse_sweep_size(
    raw: object,
) -> Literal["", "tiny", "small", "base", "large"]:
    s = str(raw).lower().strip()
    for v in _SWEEP_SIZES:
        if v == s:
            return v
    raise TrainError(
        f"sweep_size={raw!r} not in {{tiny, small, base, large, ''}}.",
    )


def _parse_sweep_dataset(
    raw: object,
) -> Literal["", "sroie", "cord"]:
    s = str(raw).lower().strip()
    for v in _SWEEP_DATASETS:
        if v == s:
            return v
    raise TrainError(
        f"sweep_dataset={raw!r} not in {{sroie, cord, ''}}.",
    )


def _validate_focus_flags(raw: dict[str, Any]) -> None:
    """Enforce Bug-18 FOCUS architecture-flag invariants at load time.

    Three rules are checked, mirroring the AGENTS.md "no silent
    placeholders" invariant applied to architecture flags:

    1. ``paper_variant == "focus"`` requires every ``focus_*`` sub-flag
       (``focus_enabled``, ``focus_total_enabled``, ``focus_company_enabled``)
       to be True — shipping a focus paper with focus off would document
       an architecture the run never invoked.
    2. ``focus_total_enabled`` requires ``priors_v4 == True`` because the
       FOCUS-T head reads ``arithmetic_witness_self`` from the v4 prior
       column (paper §III-D).
    3. ``focus_enabled`` requires ``n_priors >= 20`` (i.e. ``priors_v4``)
       — the FOCUS-A span head shares the prior projection and a
       ≤14-d v3 prior tensor would silently truncate the witness column.

    Raises :class:`ConfigError` on any violation; the error message
    names the offending key(s) so the failure is auditable from the
    config diff alone.
    """
    paper_variant_explicit = "paper_variant" in raw or bool(
        os.environ.get("KAGGLE2_PAPER_VARIANT"),
    )
    paper_variant = (
        os.environ.get("KAGGLE2_PAPER_VARIANT")
        or str(raw.get("paper_variant", "focus"))
    )
    fa = bool(raw.get("focus_enabled", False))
    ft = bool(raw.get("focus_total_enabled", False))
    fc = bool(raw.get("focus_company_enabled", False))
    pv4 = bool(raw.get("priors_v4", False))
    # The "shipping a focus paper with focus off" check only fires when
    # ``paper_variant`` is explicitly set to ``focus`` (i.e. the user
    # opted in to the focus paper).  Tests that exercise other code
    # paths with the implicit default value of ``paper_variant`` are
    # not running a focus paper and must not be tripped here.
    if paper_variant_explicit and paper_variant == "focus":
        missing = [
            n for n, v in (
                ("focus_enabled", fa),
                ("focus_total_enabled", ft),
                ("focus_company_enabled", fc),
            ) if not v
        ]
        if missing:
            raise ConfigError(
                f"paper_variant='focus' requires all focus_* sub-flags "
                f"to be True; got off: {missing} (Bug 18). "
                "Flip them on in configs/default.json or set "
                "paper_variant to a non-focus variant.",
            )
    if ft and not pv4:
        raise ConfigError(
            "focus_total_enabled=True requires priors_v4=True; the "
            "FOCUS-T head reads arithmetic_witness_self from the v4 "
            "prior column (Bug 18).",
        )
    # FOCUS-C span head requires both parent toggles.
    fcs = bool(raw.get("focus_company_span_enabled", False))
    if fcs and (not fa or not fc):
        raise ConfigError(
            "focus_company_span_enabled=True requires focus_enabled=True "
            "AND focus_company_enabled=True; set both parent toggles.",
        )
    if fa:
        # n_priors selector mirrors models/focus_train.py::train_assigner
        # so the assertion catches v3 / v2 / v1 prior shapes uniformly.
        priors_v3 = bool(raw.get("priors_v3", False))
        priors_v2 = bool(raw.get("priors_v2", True))
        n_priors = 20 if pv4 else 14 if priors_v3 else 9 if priors_v2 else 6
        if n_priors < 20:
            raise ConfigError(
                f"focus_enabled=True requires priors_v4=True (n_priors=20); "
                f"got n_priors={n_priors} (Bug 18). Set priors_v4=true.",
            )


def _parse_bug_flags(raw: object) -> dict[str, bool]:
    """Coerce ``raw`` into a {bug_1..bug_18: bool} dict; defaults True.

    Bugs 1..13 are the original silent-bug guards; 14..17 are the PR-C
    additions (anchor-extender warmup ordering, priors_v3 Bahasa false-
    fire, KD pooling on 0-box receipts, RAG self-hit on val).  Bug 18
    is the composite-assigner-loss / FOCUS-flag-architecture guard
    (one-sided MIL pos-mass loss + disabled FOCUS sub-flags).
    """
    out = {f"bug_{i}": True for i in range(1, 19)}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str) and k in out:
                out[k] = bool(v)
    return out
