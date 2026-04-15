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
    """One text-region crop extracted by YOLO."""

    image_path: Path
    bbox: tuple[float, float, float, float]  # x1 y1 x2 y2 normalised
    text: str = ""


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
    batch_size: int
    grad_accum: int
    lr: float
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
    extra: dict[str, object] = field(default_factory=dict)
