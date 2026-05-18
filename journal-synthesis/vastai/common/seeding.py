"""Deterministic seeding across stdlib random, numpy and torch.

The stdlib analysis (run_analysis.py) used seed 12345. We keep the same
default so permutation/bootstrap draws are reproducible and comparable.
torch/numpy are imported lazily so this module also imports cleanly in a
no-GPU environment for py_compile / syntax checks.
"""
from __future__ import annotations

import os
import random

DEFAULT_SEED = 12345


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG we touch. Returns the seed for logging.

    Sets PYTHONHASHSEED, stdlib random, numpy, and (if available and a
    CUDA device is present) torch + cuDNN deterministic flags.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # noqa: WPS433

        np.random.seed(seed)
    except Exception:  # numpy genuinely absent: still deterministic for stdlib
        pass
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Determinism over raw speed: required for reproducible decode.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed
