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
    yolo_img_size: int
    max_length: int
    trocr_max_len: int
    epochs_donut: int
    epochs_yolo: int
    epochs_trocr: int
    epochs_assigner: int
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
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    num_beams: int = 4
    expected_f1_warn: float = 0.75
    skip_donut: bool = False
    assigner_hidden: int = 384
    assigner_n_layers_level2: int = 6
    emit_hidden: int = 128
    emit_vocab_size: int = 259
    emit_max_len: int = 96
    emit_beam_width: int = 4
    kd_attn_weight: float = 0.0
    kd_logits_weight: float = 0.0
    assigner_patience: int = 7
    assigner_min_delta: float = 0.005
    weight_decay_assigner: float = 5e-4
    dropout_assigner: float = 0.2
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
    # config sets ``lr_assigner=3e-4`` and ``warmup_ratio_assigner=0.1``.
    lr_assigner: float = 1e-3
    warmup_ratio_assigner: float = 0.0
    # P1 — Bug-atlas ablation. Per-bug toggles (True = guard active / bug
    # fixed; False = reintroduce the bug for ablation).  13 flags keyed
    # by ``bug_<n>`` (1..13).  Default all-True keeps current behaviour.
    bug_flags: dict[str, bool] = field(
        default_factory=lambda: {f"bug_{i}": True for i in range(1, 14)},
    )
    # P2 — Retrieval-augmented DONUT (RA-KIE).  When ``rag_enabled`` is
    # True the Swin encoder is used to index train-set receipts and the
    # top-k nearest neighbours are serialised as <retrieved> tokens
    # prepended to the decoder input.
    rag_enabled: bool = False
    rag_k: int = 3
    # P3 — Graph-attention field assigner (opt-in alternative to the
    # MLP+cross-attn learned assigner in :mod:`models.pipeline_assign`).
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
    extra: dict[str, object] = field(default_factory=dict)
