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
    """AttentionAssigner vs. rule-based assignment metrics from the pipeline."""

    assigner: Metrics
    rulebased: Metrics


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
    assigner_hidden: int = 192
    assigner_n_layers_level2: int = 3
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
    extra: dict[str, object] = field(default_factory=dict)
