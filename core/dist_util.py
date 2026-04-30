"""DDP rank detection used by training-side write paths.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: small, dependency-free helpers that let any post-training
    save / verify code be safely shared between single-process and
    torchrun-launched DDP invocations.  Only rank 0 should write
    artefacts; other ranks must wait on a barrier or no-op.

Symptoms of getting this wrong (observed on the 8× RTX 5090 sweep,
PR #133): every rank reaches the post-train clean-up, opens-then-
truncates the same JSON file simultaneously, and a non-rank-0 rank
reads the file back while rank 0 is mid-write — raising
``json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)``
and aborting the entire training step.
"""
from __future__ import annotations

import os

__all__ = ["barrier", "is_rank_zero"]


def is_rank_zero() -> bool:
    """True on rank 0 (or single-process / non-DDP training).

    Detection prefers the ``RANK`` / ``LOCAL_RANK`` env vars that
    torchrun sets authoritatively per worker process — these are
    stable for the entire process lifetime and unaffected by HF
    Trainer / accelerate teardown.  Only when neither env var is
    present do we fall back to ``torch.distributed`` (covering
    bespoke launchers that ``init_process_group`` without setting
    ``RANK``).  Finally returns ``True`` when no signal is present
    so single-process runs preserve their existing behaviour.

    The env-var-first ordering matters: empirical 8× RTX 5090 swarm
    runs (PR #133 followup) showed ``dist.get_rank()`` returning 0
    on every rank during post-train cleanup even though
    ``dist.is_initialized()`` was still True — accelerate appears
    to leave the default process group in a degraded state once
    ``Trainer.train()`` returns.  The torchrun env vars are the
    only signal that stays correct end-to-end.
    """
    for var in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(var)
        if v is not None and v != "":
            try:
                return int(v) == 0
            except ValueError:
                continue
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return bool(dist.get_rank() == 0)
    except Exception:  # noqa: BLE001
        pass
    return True


def barrier() -> None:
    """Best-effort distributed barrier; no-op outside DDP."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:  # noqa: BLE001
        pass
