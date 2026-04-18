"""eval_donut must pin the inference image size to config.image_size.

DonutProcessor stores ``size`` inside its image_processor and
``processor.save_pretrained`` *should* persist it.  In practice transformers
versions occasionally drop the override on reload (when the saved
``preprocessor_config.json`` lacks the ``size`` key, which happened in
4.41–4.45) and silently fall back to the model-card default of
``{"height": 2560, "width": 1920}``.  The encoder then interpolates its
positional embeddings, halving every receipt's effective resolution and
costing ~0.05 absolute F1.

This regression test enforces that ``eval_donut`` passes ``size=`` per
inference call so the active resolution can never silently differ from
training.
"""
from __future__ import annotations

import inspect

from models import donut_eval


def test_eval_donut_passes_explicit_size_to_processor() -> None:
    src = inspect.getsource(donut_eval.eval_donut)
    assert "size_kwargs" in src and "config.image_size" in src, (
        "eval_donut must construct a size= kwarg from config.image_size and "
        "pass it to processor(images=...).  Otherwise the processor falls "
        "back to its persisted default, which can silently be the model-card "
        "size (2560x1920) and degrade F1 by ~0.05."
    )
    # The size kwarg must be passed using HxW key naming that DonutProcessor
    # expects (transformers >= 4.36 uses {"height": H, "width": W}).
    assert '"height": config.image_size[1]' in src
    assert '"width": config.image_size[0]' in src
