"""Bug-9 guard: re-pin generation_config after load_best_model_at_end reload.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: shared helper used by donut_train.py and trocr_train.py to ensure the
    persisted generation_config.json always reflects the SROIE-specific token
    IDs, not the stale mBART defaults restored by load_best_model_at_end.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.dist_util import barrier, is_rank_zero
from core.errors import TrainError

__all__ = ["_persist_generation_config"]


def _persist_generation_config(
    model: Any,
    out_dir: str,
    start_id: int,
    eos_id: int,
    pad_id: int,
) -> None:
    """Re-pin Bug-9 token IDs to model config and gc; persist and assert.

    Call this BEFORE trainer.save_model(out_dir) so the first on-disk write
    is already correct.  After save_model, call
    ``model.generation_config.save_pretrained(out_dir)`` as a belt-and-braces
    second write in case save_model re-serialises an in-memory snapshot.

    Raises TrainError if the written generation_config.json has wrong IDs.

    Under torchrun-launched DDP every rank reaches this helper.  We pin
    the IDs on the in-memory ``model.config`` / ``model.generation_config``
    on every rank (cheap, idempotent), but only rank 0 performs the disk
    write and read-back verification.  Non-zero ranks block on a barrier
    so they don't return before rank 0's bytes are flushed.
    """
    model.config.decoder_start_token_id = start_id
    model.config.eos_token_id = eos_id
    model.config.pad_token_id = pad_id
    gc = model.generation_config
    gc.decoder_start_token_id = start_id
    gc.eos_token_id = eos_id
    gc.pad_token_id = pad_id
    gc.bos_token_id = start_id
    gc.forced_bos_token_id = None  # Bug 9: mBART default leaks otherwise
    gc.forced_eos_token_id = None
    if not is_rank_zero():
        # Wait for rank 0 to finish writing and verifying.
        barrier()
        return
    # Atomic write: HuggingFace ``save_pretrained`` truncates the file
    # in place, so any rank that races this code path (e.g. a buggy
    # rank gate on a future PyTorch / accelerate release) would observe
    # an empty ``generation_config.json`` mid-write and crash with
    # ``json.JSONDecodeError: Expecting value: line 1 column 1``.  Belt-
    # and-braces: write to a sibling temp dir, then atomically rename
    # the file into place after HF's writer has flushed and closed it.
    out_p = Path(out_dir)
    tmp_p = out_p / ".gen_config_tmp"
    tmp_p.mkdir(parents=True, exist_ok=True)
    gc.save_pretrained(str(tmp_p))
    src = tmp_p / "generation_config.json"
    dst = out_p / "generation_config.json"
    os.replace(src, dst)
    # ``save_pretrained`` may also drop a ``generation_config.json``
    # marker file or similar siblings; sweep them across so the temp
    # dir is empty before removal.
    for leftover in tmp_p.iterdir():
        os.replace(leftover, out_p / leftover.name)
    tmp_p.rmdir()
    data: dict[str, object] = json.loads(dst.read_text())
    if (
        data.get("decoder_start_token_id") != start_id
        or data.get("eos_token_id") != eos_id
        or data.get("pad_token_id") != pad_id
    ):
        raise TrainError(
            "Bug-9 guard: generation_config.json mismatch after re-pin: "
            f"decoder_start_token_id={data.get('decoder_start_token_id')!r}"
            f" (want {start_id!r}), "
            f"eos_token_id={data.get('eos_token_id')!r} (want {eos_id!r}), "
            f"pad_token_id={data.get('pad_token_id')!r} (want {pad_id!r})",
        )
    barrier()
