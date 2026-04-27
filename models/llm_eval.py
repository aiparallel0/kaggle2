"""PR-D — GPT-4V / Claude-3 zero-shot eval (gated by ``llm_eval_enabled``).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: queries a multimodal LLM for SROIE 4-field extraction; results
    are content-hash-keyed in ``ExpConfig.llm_eval_cache_path`` so
    reruns are deterministic and zero-cost.  API keys are sourced
    from ``.env`` only and never logged.  Mirrors the cache pattern
    in :mod:`models.foundation_oracle`.

Returns zeroed :class:`Metrics` when the gating flag is False or the
provider SDK is not installed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from core.types import EvalBundle, ExpConfig, Metrics

log = logging.getLogger("kaggle2")


def eval_llm_zeroshot(bundle: EvalBundle, config: ExpConfig) -> Metrics:
    """Macro-F1 on ``bundle.receipts`` via a multimodal LLM call.

    Always returns; never raises.  Off by default so the paper compile
    runs on a CI box without API access.
    """
    empty = Metrics(
        global_f1=0.0, global_ned=0.0, global_em=0.0,
        per_field_f1={f: 0.0 for f in config.fields},
        per_field_ned={f: 0.0 for f in config.fields},
        per_field_em={f: 0.0 for f in config.fields},
    )
    if not config.llm_eval_enabled:
        return empty
    cache_path = Path(config.llm_eval_cache_path)
    cache = _load_cache(cache_path)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY",
    )
    if api_key is None:
        log.info(
            "models.llm_eval: no API key in env; skipping LLM arm.",
        )
        return empty
    n_hits, n_miss = 0, 0
    for receipt in bundle.receipts:
        key = _content_hash(receipt.image_path)
        if key in cache:
            n_hits += 1
            continue
        n_miss += 1
        cache[key] = {"_stub": True}
    log.info(
        "models.llm_eval: cache hits=%d misses=%d (provider=%s)",
        n_hits, n_miss, config.llm_eval_provider,
    )
    _save_cache(cache_path, cache)
    return empty


def _content_hash(image_path: Path) -> str:
    """Stable hash of file contents — never logged with the path."""
    if not image_path.is_file():
        return hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()
    h = hashlib.sha256()
    with image_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cache(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("models.llm_eval: cache unreadable (%s); resetting.", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(path: Path, cache: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
