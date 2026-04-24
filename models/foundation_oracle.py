"""Foundation-model ceiling arm (Proposal 4).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: zero-shot Claude Sonnet / GPT-4V inference on SROIE images
    producing a ``Receipt`` prediction.  Results are cached by image
    content-hash to ``config.foundation_cache_path`` so a single API
    call per image is ever made (paper results stay deterministic).

The ``anthropic`` / ``openai`` SDKs are lazy-imported and both listed
as optional dependencies — the module must import cleanly even if
neither is installed.  ``foundation_predict`` is a no-op when
``config.foundation_enabled`` is False, returning an empty Receipt
with ``image_path`` only.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from core.errors import EvalError
from core.types import ExpConfig, Receipt


def _image_hash(path: Path) -> str:
    """Stable content-hash for cache keying (sha256, truncated to 16)."""
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _load_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    """Read the on-disk predictions cache; empty dict if missing/corrupt."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}


def _save_cache(cache_path: Path, cache: dict[str, dict[str, str]]) -> None:
    """Atomic-write the cache dict back to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, cache_path)


_PROMPT = (
    "Extract the following four fields from this SROIE-format receipt: "
    "company (merchant name), date (as printed), address (full street "
    "address), total (final amount paid, numeric only). Respond with "
    'ONLY a JSON object with keys "company","date","address","total".'
)


def _call_anthropic(image_b64: str, mime: str) -> dict[str, str]:
    """Single Claude Sonnet call; raises EvalError on any SDK failure."""
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as e:
        raise EvalError("anthropic SDK not installed") from e
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022", max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                          "media_type": mime, "data": image_b64}},
            {"type": "text", "text": _PROMPT},
        ]}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _parse_json_fields(text)


def _call_openai(image_b64: str, mime: str) -> dict[str, str]:
    """Single GPT-4o call; raises EvalError on any SDK failure."""
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as e:
        raise EvalError("openai SDK not installed") from e
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o", max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]}],
    )
    return _parse_json_fields(resp.choices[0].message.content or "")


def _parse_json_fields(text: str) -> dict[str, str]:
    """Extract the first top-level JSON object from ``text``, fields-only."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {"company": "", "date": "", "address": "", "total": ""}
    try:
        obj: Any = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"company": "", "date": "", "address": "", "total": ""}
    if not isinstance(obj, dict):
        return {"company": "", "date": "", "address": "", "total": ""}
    return {k: str(obj.get(k, "")) for k in ("company", "date", "address", "total")}


def foundation_predict(image_path: Path, config: ExpConfig) -> Receipt:
    """Predict a Receipt using the foundation-model arm (cached).

    Returns an empty Receipt (fields=[]) when the arm is disabled so
    callers can fold the result into the standard metrics pipeline
    without branching.  Any SDK failure raises EvalError with the
    underlying cause chained.
    """
    if not config.foundation_enabled:
        return Receipt(image_path=image_path, fields=[])
    cache_path = Path(config.foundation_cache_path)
    cache = _load_cache(cache_path)
    key = _image_hash(image_path)
    if key in cache:
        fields_dict = cache[key]
    else:
        mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        with image_path.open("rb") as fh:
            b64 = base64.standard_b64encode(fh.read()).decode("ascii")
        if config.foundation_api == "openai":
            fields_dict = _call_openai(b64, mime)
        else:
            fields_dict = _call_anthropic(b64, mime)
        cache[key] = fields_dict
        _save_cache(cache_path, cache)
    from core.types import Field

    fields = [Field(name=k, value=v) for k, v in fields_dict.items()]
    return Receipt(image_path=image_path, fields=fields)
