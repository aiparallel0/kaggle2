"""Token-to-JSON normalisation for DONUT output strings (Bug 3 + Bug 8)."""
from __future__ import annotations

import re
from typing import Any

# Matches <s_key>value</s_key> pairs in string leaves produced by token2json
# when the model emits a duplicated outer wrapper (e.g. <s_sroie><s_sroie>…).
_TAG_RE = re.compile(r"<s_(\w+)>(.*?)</s_\1>")


def _flatten_token2json(obj: Any) -> dict[str, str]:
    """Collect all string-valued leaf entries from a nested dict/list tree.

    Recurses into dict values and list elements so wrapper keys (e.g.
    ``"sroie"``) and CORD-style page lists collapse to a single flat
    ``{field: value}`` mapping. On duplicate keys, the longest value wins,
    matching the Bug-3 fix rationale (address lines are usually truncated
    on the first occurrence).
    """
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
    """Normalise ``processor.token2json`` output to a flat ``{field: value}`` dict.

    Two shapes must be flattened:

    * **Bug 3 — list return (CORD multi-page).** ``token2json`` returns
      ``[{...}, {...}]`` when it sees ``<sep/>`` tokens in the output stream.
      Each page may contain the same key; the longest non-empty string wins
      because short strings are almost always truncations.

    * **Bug 8 — outer ``<s_sroie>`` wrapper.** Our training labels wrap every
      receipt in ``<s_sroie>…</s_sroie>`` (this tag is also the
      ``decoder_start_token_id`` / ``eos_token_id``). HuggingFace's
      ``token2json`` parses the wrapper as a root key, returning
      ``{"sroie": {"company": "X", …}}``. Flattening nested dicts collects
      the real field-level entries regardless of wrapper depth, so downstream
      ``compute_metrics`` never sees missing keys.
    """
    return _flatten_token2json(processor.token2json(tokens))
