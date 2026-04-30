"""Typed data structures shared across the kaggle2 pipeline.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: canonical definitions for Receipt, Field, Prediction, Metrics,
    DataSplit (500/63/63 train/val/test), AssignerData (per-receipt
    region groups for the AttentionAssigner), and ExpConfig (the full
    hyperparameter surface documented in Section IV).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict


class AddrPred(TypedDict):
    """One address-span prediction from the FOCUS span-cohesion head.

    ``i`` / ``j`` are the inclusive start / end region indices over the
    detected line list (so the span is ``texts[i:j+1]``).  ``span_text``
    is the joined region text for the predicted span.  ``confidence``
    is the softmax probability of the argmax cell over the masked
    flattened ``score[i, j]`` matrix; gate downstream consumers via
    :attr:`ExpConfig.focus_confidence_floor`.
    """

    i: int
    j: int
    span_text: str
    confidence: float


class TotalPred(TypedDict):
    """One total-line prediction from the FOCUS-T relational head.

    ``i`` is the picked region index (``argmax`` over the per-region
    ``final = score + witness_weight * gate * arithmetic_witness_self``
    logits); ``text`` is the raw region text at index ``i``;
    ``confidence`` is ``softmax(final)[i]``.  Returns ``i = -1`` and
    ``confidence = 0.0`` when the head is invoked on a zero-region
    receipt.
    """

    i: int
    text: str
    confidence: float


class CompanyPred(TypedDict):
    """One merchant-line prediction from the FOCUS-C positional head.

    ``i`` is the picked region index (``argmax`` over
    ``final = score - y_weight * y_norm - boilerplate_weight * boilerplate``);
    ``text`` is the raw region text at index ``i``; ``confidence`` is
    ``softmax(final)[i]``.  The y-normalised prior pulls toward the top
    of the receipt; the boilerplate prior pushes ``"SDN BHD"``-style
    suffix lines down so the head selects the merchant trade name.
    """

    i: int
    text: str
    confidence: float


class CompanySpanPred(TypedDict):
    """Multi-line company-span prediction from FOCUS-C span head.

    Mirrors :class:`AddrPred`: ``i`` / ``j`` are the inclusive start / end
    region indices over the detected line list (so the span is
    ``texts[i:j+1]``); ``span_text`` is the joined region text for the
    predicted span; ``confidence`` is the softmax probability of the
    argmax cell over the masked flattened score matrix.
    """

    i: int
    j: int
    span_text: str
    confidence: float


# ZonePosterior — per-line 3-vector ``(p_header, p_items, p_total)``.
# A list whose length matches the OCR-line count; each row sums to 1.
# Produced by :func:`models.zone_prior.decode_zone_posterior` and
# consumed by the FOCUS-C / FOCUS-T dispatch paths in
# :mod:`models.focus_pipeline` so company and total decisions share a
# single relational prior over the receipt's vertical structure.
ZonePosterior = list[tuple[float, float, float]]
ZONE_HEADER = 0
ZONE_ITEMS = 1
ZONE_TOTAL = 2


@dataclass(frozen=True)
class ZoneConfig:
    """Inference-time knobs for the 3-state receipt-zone HMM.

    The HMM segments each receipt's OCR lines into ``{header, items,
    totals}`` with monotone forward-only transitions.  Emission features
    are derived from values already present in ``priors_v4`` plus four
    boolean keyword indicators built from existing regex tables, so the
    prior costs ~30 floats and runs on CPU at inference.

    * ``enabled``            — master toggle; ``False`` keeps the legacy
                               cross-attn / arithmetic dispatch bit-for-bit.
    * ``totals_zone_floor``  — candidate-money lines with
                               ``p_total < totals_zone_floor`` are dropped
                               from :func:`models.total_arithmetic.
                               total_arithmetic_consensus` enumeration.
    * ``header_zone_floor``  — :meth:`AttentionAssigner.company_pick`
                               picks whose ``p_header`` is below this
                               floor are treated as abstentions so the
                               legacy fallback can fire.
    * ``regex_total_floor``  — the regex-argmax fallback for ``total``
                               drops candidate lines with
                               ``p_total < regex_total_floor``.
    * ``params_path``        — optional JSON file under ``results/`` with
                               EM-fit emission/transition parameters.
                               Empty falls back to hand-tuned defaults
                               which are sufficient on SROIE single-
                               column thermal receipts.
    """

    enabled: bool = True
    totals_zone_floor: float = 0.5
    header_zone_floor: float = 0.4
    regex_total_floor: float = 0.2
    params_path: str = ""


@dataclass
class Field:
    """One of the four SROIE KIE fields (company, date, address, total)."""

    name: str
    value: str


@dataclass
class Receipt:
    """One SROIE receipt: image path + ground-truth field annotations."""

    image_path: Path
    fields: list[Field]


@dataclass
class Prediction:
    """Model output for one receipt (DONUT, pipeline, or rule-based)."""

    receipt_id: str
    fields: list[Field]


@dataclass
class Metrics:
    """Token-F1, NED, EM for one system (paper Table I source)."""

    global_f1: float
    global_ned: float
    global_em: float
    per_field_f1: dict[str, float]
    per_field_ned: dict[str, float]
    per_field_em: dict[str, float]
    per_image_correct: list[bool] = field(default_factory=list)
    # Per-image *macro* F1 (mean of per-field token F1 for the receipt) —
    # the right vector for bootstrap CIs on the headline F1 metric.
    # Defaults to [] for back-compat with legacy constructions that
    # only populate the all-fields-EM vector above.
    per_image_f1: list[float] = field(default_factory=list)


@dataclass
class DataSplit:
    """500/63/63 train/val/test SROIE split (Bug 7: disjoint val/test)."""

    train: list[Receipt]
    val: list[Receipt]
    test: list[Receipt]


@dataclass
class PipelinePaths:
    """Checkpoint paths for YOLOv8, TrOCR, and AttentionAssigner."""

    yolo: str
    trocr: str
    assigner: str


@dataclass
class Crop:
    """One text-line region for TrOCR training or AttentionAssigner input."""

    image_path: Path
    bbox: tuple[float, float, float, float]  # x1 y1 x2 y2 normalised
    text: str = ""
    field_label: str = ""


@dataclass
class AssignerData:
    """Per-receipt region groups for training the AttentionAssigner.

    ``regions`` bundles labeled+distractor crops per receipt so the
    pos-mass NLL loss sees realistic multi-region inputs. ``crops`` is
    kept for TrOCR fine-tuning (labeled crops only).
    """

    trocr_path: str
    crops: list[Crop]
    regions: list[list[Crop]] = field(default_factory=list)


@dataclass
class EvalBundle:
    """Predictions + ground-truth for compute_metrics (2-in/1-out contract)."""

    predictions: list[Prediction]
    receipts: list[Receipt]
    fields: list[str]


@dataclass
class PipelineResult:
    """AttentionAssigner vs. rule-based assignment metrics from the pipeline.

    ``assigner_preds`` / ``rulebased_preds`` hold the per-receipt
    :class:`Prediction` lists in the same order as ``test`` so the
    Change F oracle-patch can surgically replace a regressed hybrid
    field with the rule-based prediction without re-running detection
    or TrOCR.  Default to empty lists for back-compat with legacy
    constructions that only cared about the aggregate metrics.
    """

    assigner: Metrics
    rulebased: Metrics
    assigner_preds: list[Prediction] = field(default_factory=list)
    rulebased_preds: list[Prediction] = field(default_factory=list)
    # Normalised gold receipts (output of :func:`models.normalize_bundle.
    # normalize_bundle` applied to the eval-time test list).  Surfaced so
    # the per-field precision / recall / bootstrap-CI producer in
    # :mod:`stages.eval_producers` can build its ``EvalBundle`` from the
    # *same* ``(preds, receipts)`` pair that ``compute_metrics`` saw —
    # without this, ``summarise_extended`` runs on (normalised pred, raw
    # gold), token-set intersections collapse, and ``F1 > max(P, R)``
    # silently appears in ``extended_metrics.json`` (PR #110 follow-up).
    receipts: list[Receipt] = field(default_factory=list)


@dataclass
class AblationRun:
    """One cell of the 13-bug ablation grid."""

    run_id: str
    bug_id: str  # e.g. "bug_1" or "all_on" or "all_off"
    seed: int
    f1: float
    ned: float
    em: float


@dataclass
class AblationReport:
    """Full 15-cell × N-seed ablation result, written to the run dir."""

    baseline_f1: float  # mean F1 with all bug fixes ON
    runs: list[AblationRun] = field(default_factory=list)
    # Per-bug ΔF1 against the all-on baseline (mean across seeds).
    per_bug_delta: dict[str, float] = field(default_factory=dict)
    # 95% bootstrap CIs on the delta (paired across seeds).
    per_bug_ci_low: dict[str, float] = field(default_factory=dict)
    per_bug_ci_high: dict[str, float] = field(default_factory=dict)
    # 13×13 interaction matrix — cell (i,j) = ΔF1 with BOTH bug_i AND bug_j off.
    # Sparse: populated only for the diagonal by default; off-diagonal cells
    # are filled when ``ablate_bugs(..., include_pairs=True)`` is used.
    interaction: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ExpConfig:
    """Full hyperparameter surface (Section IV of the paper)."""

    seed: int
    base_model: str
    trocr_model: str
    yolo_model: str
    image_size: tuple[int, int]
    yolo_image_size: int
    max_length: int
    trocr_max_len: int
    epochs_donut: int
    epochs_yolo: int
    epochs_trocr: int
    epochs_focus: int
    batch_size: int
    grad_accum: int
    lr: float
    lr_decoder: float
    warmup_steps: int
    weight_decay: float
    label_smoothing: float
    precision: str
    patience: int
    max_grad_norm: float
    fields: list[str]
    new_tokens: list[str]
    sroie_url: str
    data_dir: str
    output_dir: str
    paper_template: str
    paper_output: str
    yolo_conf: float = 0.25
    trocr_max_new_tokens: int = 64
    max_regions_per_image: int = 32
    # YOLO crop padding (pixels) added to every bbox before TrOCR.  Recovers
    # leftmost-digit drops on tight money-line bboxes ("848.00" → "48.00")
    # observed on run 20260430T125211Z. 0 = bit-exact backward-compatible.
    yolo_crop_pad_px: int = 0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    num_beams: int = 4
    f1_warn_threshold: float = 0.75
    skip_donut: bool = False
    focus_hidden_dim: int = 384
    focus_n_layers_level2: int = 6
    emit_hidden: int = 128
    emit_vocab_size: int = 259
    emit_max_len: int = 96
    emit_beam_width: int = 4
    kd_attn_weight: float = 0.0
    kd_logits_weight: float = 0.0
    focus_patience: int = 7
    focus_min_delta: float = 0.005
    weight_decay_focus: float = 5e-4
    dropout_focus: float = 0.2
    priors_v2: bool = True
    seeds: list[int] = field(default_factory=lambda: [42])
    n_trials: int = 1
    bootstrap_n_iter: int = 1000
    bootstrap_ci_level: float = 0.95
    # Change G — grid-searchable knobs for the regex-oracle router (A)
    # and the address-accept band / contiguity gate (B).  Exposed on
    # ExpConfig (and therefore combined_metrics.json) so the paper's
    # Section V table can report the exact thresholds used.
    address_accept_fraction: float = 0.5
    regex_router: bool = True
    # Change D — opt-in learned attention pool over TrOCR encoder tokens.
    # Default False keeps existing checkpoints bit-compatible; fresh
    # trains that set True get the ~800-param pool that preserves
    # SUBTOTAL / CASH sub-word signals the mean-pool erases.
    text_pool_learned: bool = False
    # Fix 3 — confidence-gated ``total`` fallback.  When the assigner's
    # softmax-normalised attention peak for ``total`` is below this
    # threshold, OR the picked line matches a SUBTOTAL keyword, fall
    # back to the rule-based extractor.  0.55 was chosen from the live
    # miss table: all 34/63 ``total`` misses had softmax-max < 0.55,
    # and no true positives had softmax-max < 0.55 in a non-trivial way.
    total_confidence_threshold: float = 0.55
    # Fix 4 — explicit assigner optimiser knobs.  Defaults preserve the
    # hardcoded legacy values (``lr=1e-3``, no warmup) so unchanged
    # training runs remain bit-for-bit reproducible; the regression-fix
    # config sets ``lr_focus=3e-4`` and ``warmup_ratio_focus=0.1``.
    lr_focus: float = 1e-3
    warmup_ratio_focus: float = 0.0
    # PR-A — bug-atlas extension.  Bugs 14-17 are PR-C-era guards:
    #   bug_14 — anchor-extender warmup ordering must precede the head
    #   bug_15 — priors_v3 must NOT fire ``is_distractor`` on Bahasa
    #            ``JUMLAH BESAR`` (a TOTAL synonym, not a distractor)
    #   bug_16 — KD pooling on a 0-box receipt must skip + log, not div0
    #   bug_17 — RAG retrieval must reject self-hits on val (id leak)
    # 13 flags keyed by ``bug_<n>`` (1..17).  Default all-True keeps
    # current behaviour (every guard active).
    bug_flags: dict[str, bool] = field(
        default_factory=lambda: {f"bug_{i}": True for i in range(1, 18)},
    )
    # P2 — Retrieval-augmented DONUT (RA-KIE).  When ``rag_enabled`` is
    # True the Swin encoder is used to index train-set receipts and the
    # top-k nearest neighbours are serialised as <retrieved> tokens
    # prepended to the decoder input.
    rag_enabled: bool = False
    rag_k: int = 3
    # P3 — Graph-attention field assigner (opt-in alternative to the
    # MLP+cross-attn learned assigner in :mod:`models.focus_pipeline`).
    gat_enabled: bool = False
    # P4 — Foundation-model ceiling arm.  Claude/GPT-4V zero-shot
    # inference; cached to ``foundation_cache_path`` (keyed by content
    # hash) for determinism.  Lazy-imported so anthropic/openai are
    # optional dependencies.
    foundation_enabled: bool = False
    foundation_api: str = "anthropic"
    foundation_cache_path: str = "./runs/foundation_cache.json"
    # v4 — Canonical SROIE 347-image test split.  When
    # ``canonical_sroie_enabled`` is True the eval stage runs each
    # trained model on the canonical ICDAR-2019 SROIE test (347
    # receipts) in addition to the 63-image held-out split, and emits
    # ``metrics/canonical_<system>.json`` so Table~IV-bis resolves.
    # ``canonical_sroie_test_path`` is the absolute path to a directory
    # containing ``img/<id>.jpg`` and ``box/<id>.txt`` (or the SROIE
    # task-3 archive layout); ``""`` defers detection to runtime.
    canonical_sroie_enabled: bool = False
    canonical_sroie_test_path: str = ""
    # Real download endpoints + pinned sha256 (see data/sroie_canonical.py).
    # Defaults match the official ICDAR Task-3 page; the HuggingFace mirror
    # (Metric-AI/icdar_sroie) is the identity-verified fallback.
    canonical_sroie_test_url: str = (
        "https://rrc.cvc.uab.es/downloads/SROIE_test_images_task_3.zip"
    )
    canonical_sroie_gt_url: str = (
        "https://rrc.cvc.uab.es/downloads/SROIE_test_gt_task_3.zip"
    )
    canonical_sroie_hf_repo: str = "Metric-AI/icdar_sroie"
    canonical_sroie_hf_revision: str = "main"
    # Two-variant paper bifurcation (see report/template_focus.tex vs
    # report/template_baseline.tex).  ``focus`` (default) is the headline
    # 626-train + 347-test comparison; ``baseline`` is the reduced-scope
    # 500/63/63 + GT-OCR baseline study.  CLI flag: --paper-variant.
    paper_variant: str = "focus"
    # v4 — LayoutLMv3 baseline.  Off by default so the headline run is
    # bit-for-bit reproducible against v3; flip to True on the first
    # GPU box that has the public HF checkpoint cached.
    layoutlmv3_enabled: bool = False
    layoutlmv3_model: str = "microsoft/layoutlmv3-base"
    # v4 — Inference-latency producer.  When True the eval loop times
    # each forward pass and writes ``metrics/latency_<system>.json``
    # so Table~X (the latency table) resolves.  Off by default to keep
    # eval-time deterministic; flip on for the latency-focused profile.
    measure_latency: bool = False
    # v4 — Curated qualitative-sample IDs for Fig.~11 (4-receipt grid:
    # both-correct / DONUT-wins / pipeline-wins / both-fail).  Empty
    # list defers selection to ``report.figures_samples`` which then
    # picks the first 4 IDs that have predictions stored for both
    # systems.  Authors override this list to fix the curated set.
    qualitative_sample_ids: list[str] = field(default_factory=list)
    # v4 — Single representative receipt id for Fig.~1 (architecture
    # diagram with the SAME receipt flowing through both panels).
    # Empty defers to the first qualitative sample ID.
    fig1_receipt_id: str = ""
    # v4 — Strict paper-stage gate.  When True the paper stage raises
    # an EvalError if any unresolved \VAR{} key is not on the
    # ``MISSING_OK`` allow-list.  Default False keeps current behaviour
    # (warn + render \MissingCell{}).  CI gates should flip to True.
    strict_paper: bool = False
    # PR-A / T-C — magic-number → config promotions.  Defaults preserve
    # the legacy hard-coded values so reference runs reproduce bit-for-
    # bit.  Pulled into the assigner train loop and the rule-based
    # teacher KD term so reviewers can grid-search without editing src.
    assigner_hardneg_margin: float = 0.10
    assigner_kd_temperature: float = 2.0
    assigner_field_weights: dict[str, float] = field(
        default_factory=lambda: {
            "company": 1.5, "address": 1.3, "total": 1.2, "date": 0.8,
        },
    )
    # PR-A / T-A1 — checkpoint-introspection assertion tolerance.  When
    # ``stages/eval.py`` introspects the assigner checkpoint at startup
    # it asserts ``abs(introspected_params - declared_params) <
    # assigner_param_drift_tol`` (default 500) so a stale checkpoint /
    # config drift surfaces before any GPU work.
    assigner_param_drift_tol: int = 500
    # PR-C / S2 — opt-in 14-d distractor-aware text priors (priors_v3).
    # Default False so existing 6-/9-d checkpoints load bit-exact;
    # fresh trains flip True alongside an assigner re-train.
    priors_v3: bool = False
    # PR-C / S1 — opt-in anchor-then-extend address head.  When True the
    # ``AttentionAssigner`` exposes an ``address_anchor_head`` +
    # ``address_extender`` pair and the address logit is replaced with
    # ``anchor_score + extender_scores`` over ±k vertical neighbours.
    address_anchor_extend: bool = False
    address_anchor_extender_k: int = 2
    # PR-C / S3 — fusion head selector for ``AttentionAssigner.forward``.
    # ``"sum"`` (default) keeps the legacy additive fusion that is
    # bit-exact with shipped checkpoints; ``"concat"`` activates a
    # ``Linear(2h or 3h, h)`` projection over the concatenated
    # ``[text, bbox, prior]`` tensors.
    fusion: Literal["sum", "concat"] = "sum"
    # PR-C / S0 — address-assembly scoring weights.  Promoted onto
    # ``AssignerPolicy`` (see ``models/focus_pipeline.py``) so PR-E
    # sweeps can grid-search without editing source.  Defaults match
    # the values measured on SROIE (median 3 lines, IQR 2-5; postcode
    # tail bonus 0.05; money-token-inside penalty 0.10).
    address_score_token_f1_w: float = 1.0
    address_score_line_count_w: float = 0.25
    address_score_postcode_w: float = 0.05
    address_score_money_penalty: float = 0.10
    # PR-E — Pareto sweep knobs.  Empty default keeps existing eval flow
    # unchanged; populated only by ``configs/sweep/*.json``.
    sweep_size: Literal["", "tiny", "small", "base", "large"] = ""
    sweep_dataset: Literal["", "sroie", "cord"] = ""
    # PR-D — gated GPT-4V / generic-foundation eval (mirrors the existing
    # ``foundation_*`` knobs but kept distinct so the cache files do not
    # collide).  ``llm_eval_enabled`` is the public switch surfaced in
    # the paper's competitor table; ``llm_eval_cache_path`` keeps GPT-4V
    # answers content-hash-keyed so reruns are deterministic.
    llm_eval_enabled: bool = False
    llm_eval_provider: str = "gpt-4v"
    llm_eval_cache_path: str = "./runs/llm_eval_cache.json"
    # PR-D — carbon emissions accounting (Section "Energy &
    # Emissions").  ``grid_factor`` is the country-specific kgCO2e/kWh
    # default; override via ``ExpConfig.extra["grid_factor"]`` for
    # CI-pinned reruns from a different region.
    carbon_grid_factor_kgco2e_per_kwh: float = 0.475
    # FOCUS address-span head (PR-FOCUS).  Off by default so the headline
    # checkpoints stay bit-exact.  When ``focus_enabled`` is True the
    # ``AttentionAssigner`` instantiates a 3-projection ``_AddressSpanHead``
    # over its post-encoder ``kv`` tensor and the trainer adds the
    # span-IoU + boundary-CE term (weights below) on address-field
    # receipts only.  ``focus_max_span`` caps the contiguous span length
    # (8 covers >99 % of SROIE addresses; see ``_tier_a_clean``).
    # ``focus_confidence_floor`` is the deployment gate consumers apply
    # to ``AddrPred.confidence`` before accepting a predicted span.
    focus_enabled: bool = False
    focus_max_span: int = 8
    focus_iou_weight: float = 1.0
    focus_boundary_weight: float = 1.0
    focus_confidence_floor: float = 0.10
    # FOCUS framework (paper §III-D rewrite).  ``focus_enabled`` remains
    # the master toggle (back-compat: PR #106's FOCUS-A); the four sub-flags
    # below gate FOCUS-T (relational total head), FOCUS-C (positional
    # company head), and the priors_v4 builder.  Defaults preserve baseline
    # bit-for-bit: ``focus_enabled=True`` with all sub-flags False
    # reproduces PR #106 exactly (FOCUS-A only).  FOCUS-D (date) stays on
    # the existing point head — its 0.92 baseline F1 confirms the
    # regex-conforming substring is already near-saturated by token-level
    # cross-attention, so a structural prior would only add bias.
    focus_total_enabled: bool = False
    focus_total_witness_weight: float = 1.0
    focus_company_enabled: bool = False
    focus_company_y_weight: float = 1.0
    focus_company_boilerplate_weight: float = 1.0
    # FOCUS-C inference confidence gate.  Replaces the cross-attn argmax
    # with :meth:`AttentionAssigner.company_pick` (anchor index for the
    # :func:`models.postprocess_company._company_span` greedy assembler)
    # iff ``softmax(final)[i] >= focus_company_confidence_threshold``;
    # below this the legacy argmax pick wins so confident negatives do
    # not regress.  Default 0.40 — bumped from the original 0.30 by the
    # zone-prior PR, which raises *signal* via the relational
    # ``ZonePosterior`` so the threshold can move *up* (precision) rather
    # than down (recall) to address upstream miss modes.
    focus_company_confidence_threshold: float = 0.40
    # FOCUS-C span head (mirrors FOCUS-A).  When ``focus_company_span_enabled``
    # is True the AttentionAssigner instantiates a 3-projection
    # ``_CompanySpanHead`` and the trainer adds the span-IoU + boundary-CE
    # term on company-field receipts.  ``focus_company_span_max_span`` caps
    # the contiguous span length (4 covers >99% of SROIE company names).
    # ``focus_company_confidence_floor`` is the deployment gate consumers
    # apply to ``CompanySpanPred.confidence`` before accepting a span.
    focus_company_span_enabled: bool = False
    focus_company_span_max_span: int = 4
    focus_company_span_iou_w: float = 1.0
    focus_company_span_boundary_w: float = 1.0
    focus_company_confidence_floor: float = 0.20
    # Dedicated auxiliary losses for FOCUS-T and FOCUS-C positional heads.
    # Default 0.0 disables them so existing checkpoints reproduce bit-exact;
    # set to 1.0 in configs/default.json to activate for new training runs.
    focus_total_aux_w: float = 0.0
    focus_company_pos_aux_w: float = 0.0
    priors_v4: bool = False
    # FOCUS-T arithmetic-consensus dispatch.  When True, the ``total``
    # field tries :func:`models.total_arithmetic.total_arithmetic_consensus`
    # ahead of the regex router so receipts whose grand total satisfies
    # ``cash − change`` or ``subtotal + tax + service − discount``
    # commit the consensus value without trusting the (often OCR-
    # corrupted) total line itself.  Default ``True`` because the
    # consensus is conservative — it abstains on under-determined
    # receipts and the caller falls through to the legacy chain.
    total_arithmetic_enabled: bool = True
    # Relational receipt-zone prior (PR — shared zone for company/total).
    # A 3-state monotone HMM (header → items → totals) is decoded by
    # forward–backward over per-line text features and routed into both
    # FOCUS-C (company) and FOCUS-T (total) dispatch paths in
    # :mod:`models.focus_pipeline` so the two fields share a single
    # structural prior.  Defaults preserve the legacy paths bit-for-bit
    # when ``zone_prior_enabled=False``.
    zone_prior_enabled: bool = True
    zone_totals_floor: float = 0.5
    zone_header_floor: float = 0.4
    zone_regex_total_floor: float = 0.2
    zone_params_path: str = ""
    extra: dict[str, object] = field(default_factory=dict)
