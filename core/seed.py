"""Deterministic seeding for reproducible DONUT and pipeline training.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: ensures bit-for-bit reproducibility on identical hardware by
    seeding stdlib random, numpy, torch, and CUDA RNGs.  The paper's
    multi-seed harness (--seeds) re-invokes this function per seed to
    produce the mean±std F1 reported in Table I.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducible training; call before any GPU work."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
