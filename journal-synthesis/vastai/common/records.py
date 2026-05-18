"""Decode-once shared per-receipt cache.

THE COST BUG this fixes: every experiment script (E1E3, E5, E6, E9,
E10, ...) independently loaded the KIE model and RE-DECODED the SAME
corpora, so a single run paid for the SAME Donut inference ~7x. On a
paid vast.ai GPU that is ~90% wasted spend.

This module provides ONE function, `decode_or_load`, that decodes a
corpus EXACTLY as the scripts already did (same `decode_fields` greedy
pass and same `beam_margin_batch` num_beams=2 pass from
common.pipeline - byte-for-byte the same calls, same batching
semantics, same numerics) and persists the raw per-receipt decode
PRIMITIVES to a cache file:

    results/<corpuslabel>__<ckpthash>.records.jsonl

On a second call with the same corpus + checkpoint the cache is read
back with NO model load and NO GPU touched. Each experiment script
then rebuilds its OWN records / metrics from these primitives with its
ORIGINAL math unchanged (gold/items parsing, subset-sum verdict,
verifier verdicts, thresholds, seeds) - this module deliberately does
NOT compute any experiment metric, only the shared decode.

WHAT IS CACHED PER RECEIPT (the primitives every script derived its
records from, nothing experiment-specific, nothing fabricated):
  receipt_id (bare rid), gold (the raw annotation JSON the script
  loaded), fields (processor.token2json output), softmax_confidence,
  c_seq, beam_margin.

The first JSONL line is a HEADER carrying `computed_on`, `n_records`,
`checkpoint_sha`, corpus label/path, task_prompt and a
`schema_version`. A cache is reused ONLY if the header is present, the
key fields match the request, AND the body line count equals the
header's `n_records` (so a truncated / half-written cache is detected
and rebuilt, never half-used). The checkpoint hash is part of the
filename AND the header so a different checkpoint can never silently
reuse stale decodes.

HEAVY IMPORTS (torch / transformers / PIL) stay lazy: this module
byte-compiles and imports head-less for `python3 -m py_compile`. It
never fabricates: a cache only ever holds real decodes; if no cache
exists and no GPU is present the underlying load_donut raises.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .pipeline import load_donut, decode_fields, beam_margin_batch

# Bump if the cached primitive set changes (forces a rebuild rather
# than silently consuming an incompatible old cache).
SCHEMA_VERSION = 1


def checkpoint_sha(checkpoint: str) -> str:
    """Stable short identifier for a checkpoint string. Used in the
    cache filename AND header so a different checkpoint never silently
    reuses another checkpoint's decodes."""
    return hashlib.sha1(checkpoint.encode("utf-8")).hexdigest()[:12]


def split_corpus_arg(corpus_arg: str) -> Tuple[str, str]:
    """`label=path` -> (label, path), identical to the split every
    script already did with `args.corpus.split('=', 1)`."""
    label, path = corpus_arg.split("=", 1)
    return label, path


def cache_path(results_dir: str, label: str, checkpoint: str) -> str:
    return os.path.join(
        results_dir, f"{label}__{checkpoint_sha(checkpoint)}.records.jsonl")


def _enumerate_receipts(path: str):
    """Canonical receipt enumeration: sorted *.json annotations whose
    matching images/<rid>.png exists. This is EXACTLY the order + skip
    rule the scripts already used (sorted listdir of annotations,
    `.json` only, skip when the png is missing)."""
    anns = os.path.join(path, "annotations")
    imgs_dir = os.path.join(path, "images")
    out = []
    for fn in sorted(os.listdir(anns)):
        if not fn.endswith(".json"):
            continue
        rid = os.path.splitext(fn)[0]
        ip = os.path.join(imgs_dir, rid + ".png")
        if not os.path.exists(ip):
            continue
        out.append((rid, os.path.join(anns, fn), ip))
    return out


def _try_load_cache(path: str, label: str, corpus_path: str,
                    checkpoint: str, task_prompt: str
                    ) -> Optional[List[Dict[str, Any]]]:
    """Return cached primitive records iff the cache exists, its header
    matches the request, and the body is complete (n_records matches).
    Otherwise None (caller will decode + rebuild). NEVER returns a
    partial cache."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    try:
        header = json.loads(lines[0])
    except (ValueError, IndexError):
        return None
    if not isinstance(header, dict) or header.get("_header") is not True:
        return None
    if (header.get("schema_version") != SCHEMA_VERSION
            or header.get("checkpoint_sha") != checkpoint_sha(checkpoint)
            or header.get("corpus_label") != label
            or header.get("corpus_path") != corpus_path
            or header.get("task_prompt") != task_prompt):
        return None
    body = lines[1:]
    if header.get("n_records") != len(body):
        # truncated / half-written -> rebuild, never half-use
        return None
    recs: List[Dict[str, Any]] = []
    for ln in body:
        try:
            recs.append(json.loads(ln))
        except ValueError:
            return None  # corrupt body line -> rebuild
    if len(recs) != header.get("n_records"):
        return None
    return recs


def _write_cache(path: str, label: str, corpus_path: str,
                 checkpoint: str, task_prompt: str,
                 recs: List[Dict[str, Any]]) -> None:
    import datetime
    import socket
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = {
        "_header": True,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha": checkpoint_sha(checkpoint),
        "corpus_label": label,
        "corpus_path": corpus_path,
        "task_prompt": task_prompt,
        "n_records": len(recs),
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)  # atomic: a reader never sees a half file


def decode_or_load(corpus_arg: str, checkpoint: str, task_prompt: str,
                   batch: int, results_dir: Optional[str] = None
                   ) -> List[Dict[str, Any]]:
    """Decode-or-load the shared per-receipt primitives for one corpus.

    Returns a list (in canonical receipt order) of dicts:
      {"receipt_id": <bare rid>, "gold": <raw annotation JSON>,
       "fields": <decode_fields token2json dict>,
       "softmax_confidence": float, "c_seq": float,
       "beam_margin": float|None}

    If a complete, key-matching cache exists it is returned WITHOUT
    importing torch / loading the model / touching the GPU. Otherwise
    the model is loaded ONCE, the corpus is decoded with the EXACT same
    `decode_fields` + `beam_margin_batch` calls (same beam settings,
    fp16, task prompt) the scripts used, the primitives are cached, and
    returned.

    Every script then applies its OWN unchanged math to these
    primitives (gold/items parsing, subset-sum verdict, verifier
    verdicts, thresholds). beam_margin is always computed (E5/E6/E10
    need it; E1E3/E9 simply ignore it and still emit beam_margin=None
    exactly as before) so the SAME cache serves every consumer.
    """
    label, path = split_corpus_arg(corpus_arg)
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results")
        results_dir = os.path.abspath(results_dir)
    cpath = cache_path(results_dir, label, checkpoint)

    cached = _try_load_cache(cpath, label, path, checkpoint, task_prompt)
    if cached is not None:
        return cached

    # ---- cache miss: decode ONCE (same calls as the scripts) ----------
    from PIL import Image

    processor, model = load_donut(checkpoint)
    dec_one = processor.tokenizer(
        task_prompt, add_special_tokens=False,
        return_tensors="pt").input_ids
    pad = processor.tokenizer.pad_token_id

    receipts = _enumerate_receipts(path)
    recs: List[Dict[str, Any]] = []
    for i in range(0, len(receipts), batch):
        chunk = receipts[i:i + batch]
        imgs, rids, golds = [], [], []
        for rid, ann_path, ip in chunk:
            with open(ann_path) as f:
                golds.append(json.load(f))
            imgs.append(Image.open(ip).convert("RGB"))
            rids.append(rid)
        if not imgs:
            continue
        decoded = decode_fields(imgs, processor, model, task_prompt)
        bm = beam_margin_batch(imgs, processor, model, dec_one, pad)
        for j, rid in enumerate(rids):
            fields, sm, cs = decoded[j]
            margin = bm[j]["margin"] if j < len(bm) else None
            recs.append({
                "receipt_id": rid,
                "gold": golds[j],
                "fields": fields if isinstance(fields, dict) else {},
                "softmax_confidence": sm,
                "c_seq": cs,
                "beam_margin": margin,
            })

    _write_cache(cpath, label, path, checkpoint, task_prompt, recs)
    return recs
