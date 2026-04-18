"""train_trocr must shuffle crops before the train/val split.

The previous ``crops[:split], crops[split:]`` carved validation out of the
*last* 10 % of the input, which (because ``extract_crops`` walks receipts in
filename order) is dominated by receipts whose stems begin with high
hexadecimal prefixes.  That non-representative slice biased
``eval_f1`` by ~0.05 and routinely picked a worse 'best' checkpoint via
``load_best_model_at_end``.

Use a deterministic ``random.Random(seed)`` shuffle so the split is
uncorrelated with filename order but still reproducible across runs.
"""
from __future__ import annotations

import inspect

from models import trocr_train


def test_train_trocr_shuffles_crops_before_split() -> None:
    src = inspect.getsource(trocr_train.train_trocr)
    # We require both a shuffle and a deterministic seed (config.seed),
    # not just ``random.shuffle`` which would re-seed every process.
    assert "Random(config.seed).shuffle" in src, (
        "train_trocr must call random.Random(config.seed).shuffle(...) on "
        "the crops list before computing split = int(len(...) * 0.9). "
        "Without it the val set is filename-ordered, biasing eval_f1."
    )
    # And the split must operate on the shuffled list, not the original.
    assert "shuffled[:split]" in src and "shuffled[split:]" in src, (
        "After shuffling, the slice must be applied to the shuffled list "
        "(not to ``crops``) so the val set actually rotates."
    )
