"""Token-to-JSON normalisation flattening Bug 3 (list) and Bug 8 (wrapper).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: flattens the nested dict/list returned by HuggingFace token2json into
    a {field: value} dict, handling CORD-style multi-page lists (Bug 3) and
    outer <s_sroie> wrappers (Bug 8).  Longest-wins deduplication.
"""
from __future__ import annotations

import re
from typing import Any

# Matches <s_key>value</s_key> pairs in string leaves produced by token2json
# when the model emits a duplicated outer wrapper (e.g. <s_sroie><s_sroie>…).
_TAG_RE = re.compile(r"<s_(\w+)>(.*?)</s_\1>")


def _flatten_token2json(obj: Any) -> dict[str, str]:
    """Collect all string leaves from nested dict/list (Bug 3/8 flattening)."""
    merged: dict[str, str] = {}

    def _merge(key: str, value: str) -> None:
        if key not in merged or len(value) > len(merged[key]):
            merged[key] = value

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict | list):
                for sub_k, sub_v in _flatten_token2json(v).items():
                    _merge(sub_k, sub_v)
            else:
                sv = str(v)
                if "<s_" in sv and "</s_" in sv:
                    for tag_key, tag_val in _TAG_RE.findall(sv):
                        _merge(tag_key, tag_val)
                else:
                    _merge(k, sv)
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict | list):
                for sub_k, sub_v in _flatten_token2json(entry).items():
                    _merge(sub_k, sub_v)
    return merged


def token2json_safe(processor: Any, tokens: str) -> dict[str, str]:
    """Normalize token2json output to flat {field: value} (Bug 3/8 safe)."""
    return _flatten_token2json(processor.token2json(tokens))
