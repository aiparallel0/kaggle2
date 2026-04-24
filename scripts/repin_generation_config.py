#!/usr/bin/env python3
"""Repair a stale ``generation_config.json`` on a trained checkpoint (Bug 9).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: post-hoc recovery for runs whose ``generation_config.json`` has the
    wrong ``decoder_start_token_id`` (Bug 9) — the weights are fine, only
    the JSON sidecar is stale, so we rewrite it from the on-disk tokenizer
    and skip retraining.  See ``docs/bug9_postmortem.md``.

Usage:
    python scripts/repin_generation_config.py results/trocr
    python scripts/repin_generation_config.py results/donut --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_DONUT_START = "<s_sroie>"
_DONUT_EOS = "</s_sroie>"
_MANAGED_KEYS = (
    "decoder_start_token_id", "eos_token_id", "pad_token_id",
    "bos_token_id", "forced_bos_token_id", "forced_eos_token_id",
)


def _load_tokenizer(ckpt_dir: Path) -> Any:
    """Load the tokenizer bundled with the checkpoint (DONUT or TrOCR)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required to resolve token IDs; "
            "install it or pass --decoder-start-token-id/--eos-token-id/"
            "--pad-token-id manually.",
        ) from exc
    return AutoTokenizer.from_pretrained(str(ckpt_dir))


def _detect_kind(tokenizer: Any) -> str:
    """Return ``"donut"`` if SROIE special tokens are present, else ``"trocr"``."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    return "donut" if _DONUT_START in vocab and _DONUT_EOS in vocab else "trocr"


def _ids_from_tokenizer(tokenizer: Any, kind: str) -> dict[str, int]:
    """Resolve Bug-9 IDs from the tokenizer for the detected kind."""
    if kind == "donut":
        start = tokenizer.convert_tokens_to_ids([_DONUT_START])[0]
        eos = tokenizer.convert_tokens_to_ids([_DONUT_EOS])[0]
    elif kind == "trocr":
        start, eos = tokenizer.cls_token_id, tokenizer.sep_token_id
    else:
        raise SystemExit(f"Unknown --kind {kind!r}; expected 'donut' or 'trocr'.")
    pad = tokenizer.pad_token_id
    triples = [("decoder_start_token_id", start), ("eos_token_id", eos),
               ("pad_token_id", pad)]
    for name, val in triples:
        if val is None:
            raise SystemExit(
                f"Tokenizer did not provide {name} for kind={kind!r}; "
                "pass it explicitly via the matching CLI flag.",
            )
    return {name: int(val) for name, val in triples}


def _diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    lines = []
    for k in _MANAGED_KEYS:
        b, a = before.get(k, "<missing>"), after.get(k)
        marker = " " if b == a else "*"
        lines.append(f"  {marker} {k}: {b!r} -> {a!r}")
    return "\n".join(lines)


def _backup(path: Path) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%SZ")
    dest = path.with_suffix(path.suffix + f".bak-{stamp}")
    shutil.copy2(path, dest)
    return dest


def _write_and_verify(gc_path: Path, data: dict[str, Any], ids: dict[str, int]) -> None:
    data = dict(data)
    data.update(ids)
    data["bos_token_id"] = ids["decoder_start_token_id"]
    data["forced_bos_token_id"] = None
    data["forced_eos_token_id"] = None
    gc_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    rt = json.loads(gc_path.read_text())
    for k, want in ids.items():
        if rt.get(k) != want:
            raise SystemExit(
                f"Round-trip verification failed: {k}={rt.get(k)!r} (want {want!r})",
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path,
                   help="Path to checkpoint dir (e.g. results/trocr).")
    p.add_argument("--kind", choices=("auto", "donut", "trocr"), default="auto")
    p.add_argument("--decoder-start-token-id", type=int, default=None)
    p.add_argument("--eos-token-id", type=int, default=None)
    p.add_argument("--pad-token-id", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Print diff and exit without writing.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip timestamped backup of generation_config.json.")
    args = p.parse_args(argv)

    ckpt: Path = args.checkpoint
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")
    gc_path = ckpt / "generation_config.json"
    if not gc_path.is_file():
        raise SystemExit(f"generation_config.json not found in {ckpt}")

    before: dict[str, Any] = json.loads(gc_path.read_text())

    need_tokenizer = (args.decoder_start_token_id is None
                      or args.eos_token_id is None
                      or args.pad_token_id is None
                      or args.kind == "auto")
    if need_tokenizer:
        tokenizer = _load_tokenizer(ckpt)
        kind = _detect_kind(tokenizer) if args.kind == "auto" else args.kind
        ids = _ids_from_tokenizer(tokenizer, kind)
    else:
        kind = args.kind
        ids = {}
    for name, val in (("decoder_start_token_id", args.decoder_start_token_id),
                      ("eos_token_id", args.eos_token_id),
                      ("pad_token_id", args.pad_token_id)):
        if val is not None:
            ids[name] = int(val)

    after = {**before, **ids,
             "bos_token_id": ids["decoder_start_token_id"],
             "forced_bos_token_id": None, "forced_eos_token_id": None}

    print(f"[repin] checkpoint={ckpt}  kind={kind}")
    print("[repin] generation_config.json diff (before -> after):")
    print(_diff(before, after))

    if args.dry_run:
        print("[repin] --dry-run; no file written.")
        return 0
    if not args.no_backup:
        backup = _backup(gc_path)
        print(f"[repin] backup: {backup}")
    _write_and_verify(gc_path, before, ids)
    print("[repin] wrote and round-trip-verified generation_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
