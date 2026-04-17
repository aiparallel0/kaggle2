"""Deterministic seeding for torch, numpy, random, and CUDA."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed stdlib random, numpy, torch, and set deterministic CUDA flags.

    Must be called before any dataset shuffling, model init, or HF Trainer
    construction so runs are reproducible bit-for-bit on the same hardware.
    """
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
