"""Core dataclasses shared across all modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Field:
    """One ground-truth or predicted KIE field."""

    name: str
    value: str


@dataclass
class Receipt:
    """One SROIE receipt with its image path and ground-truth fields."""

    image_path: Path
    fields: list[Field]


@dataclass
class Prediction:
    """Model prediction for one receipt."""

    receipt_id: str
    fields: list[Field]


@dataclass
class Metrics:
    """Evaluation metrics for one architecture."""

    global_f1: float
    global_ned: float
    global_em: float
    per_field_f1: dict[str, float]
    per_field_ned: dict[str, float]
    per_field_em: dict[str, float]


@dataclass
class DataSplit:
    """Train / val / test receipt lists."""

    train: list[Receipt]
    val: list[Receipt]
    test: list[Receipt]


@dataclass
class PipelinePaths:
    """Filesystem paths for the three pipeline model checkpoints."""

    yolo: str
    trocr: str
    assigner: str


@dataclass
class Crop:
    """One text-region crop extracted by YOLO or SROIE box annotations."""

    image_path: Path
    bbox: tuple[float, float, float, float]  # x1 y1 x2 y2 normalised
    text: str = ""
    field_label: str = ""


@dataclass
class AssignerData:
    """Training payload for the attention-based field assigner.

    ``regions`` groups every Crop per-receipt (labeled + distractors) so the
    assigner trains on realistic multi-region inputs. ``crops`` is retained
    for callers that only need labeled crops.
    """

    trocr_path: str
    crops: list[Crop]
    regions: list[list[Crop]] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Evaluation results for the pipeline with both assignment strategies."""

    assigner: Metrics
    rulebased: Metrics


@dataclass
class ExpConfig:
    """Full experiment configuration loaded from config.json."""

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
    extra: dict[str, object] = field(default_factory=dict)
