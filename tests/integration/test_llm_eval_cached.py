"""PR-D — LLM eval cache contract."""
from __future__ import annotations

from pathlib import Path

from conftest import write_min_config


def test_llm_eval_disabled_returns_empty(tmp_path: Path) -> None:
    from core.config import load_config
    from core.types import EvalBundle
    from models.llm_eval import eval_llm_zeroshot

    cfg = load_config(str(write_min_config(tmp_path, llm_eval_enabled=False)))
    bundle = EvalBundle(predictions=[], receipts=[], fields=list(cfg.fields))
    m = eval_llm_zeroshot(bundle, cfg)
    assert m.global_f1 == 0.0


def test_llm_eval_no_api_key_returns_empty(
    tmp_path: Path, monkeypatch: object,
) -> None:
    from core.config import load_config
    from core.types import EvalBundle
    from models.llm_eval import eval_llm_zeroshot

    mp = monkeypatch  # type: ignore[assignment]
    mp.delenv("OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    mp.delenv("ANTHROPIC_API_KEY", raising=False)  # type: ignore[attr-defined]
    cfg = load_config(str(write_min_config(
        tmp_path,
        llm_eval_enabled=True,
        llm_eval_cache_path=str(tmp_path / "cache.json"),
    )))
    bundle = EvalBundle(predictions=[], receipts=[], fields=list(cfg.fields))
    m = eval_llm_zeroshot(bundle, cfg)
    assert m.global_f1 == 0.0


def test_llm_eval_content_hash_stable() -> None:
    from models.llm_eval import _content_hash

    p = Path("/nonexistent/abc.png")
    h1 = _content_hash(p)
    h2 = _content_hash(p)
    assert h1 == h2
    assert len(h1) == 64
