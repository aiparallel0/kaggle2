"""PR-D — LayoutLMv3 evaluator smoke test."""
from __future__ import annotations

from pathlib import Path

from conftest import write_min_config


def test_layoutlmv3_disabled_returns_empty_metrics(tmp_path: Path) -> None:
    from core.config import load_config
    from core.types import EvalBundle
    from models.layoutlmv3_eval import eval_layoutlmv3

    cfg = load_config(str(write_min_config(tmp_path, layoutlmv3_enabled=False)))
    bundle = EvalBundle(predictions=[], receipts=[], fields=list(cfg.fields))
    m = eval_layoutlmv3(bundle, cfg)
    assert m.global_f1 == 0.0
    assert set(m.per_field_f1.keys()) == set(cfg.fields)


def test_layoutlmv3_enabled_no_transformers_skips(tmp_path: Path) -> None:
    """Enabled flag + no ``transformers`` → empty metrics, no raise."""
    from core.config import load_config
    from core.types import EvalBundle
    from models.layoutlmv3_eval import eval_layoutlmv3

    cfg = load_config(str(write_min_config(tmp_path, layoutlmv3_enabled=True)))
    bundle = EvalBundle(predictions=[], receipts=[], fields=list(cfg.fields))
    m = eval_layoutlmv3(bundle, cfg)
    assert m.global_f1 == 0.0
