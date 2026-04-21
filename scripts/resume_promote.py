"""Promote the best TrOCR checkpoint into results/trocr/ after a crash.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: os.rename-based promotion avoids disk-doubling (critical when
    disk is already full from the ENOSPC crash).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from core.types import ExpConfig
from scripts.resume_cleanup import checkpoint_step, rmtree_quiet

log = logging.getLogger("resume")

# Training-state sidecars that are not needed for inference. Deleting them
# before the os.rename into trocr_dir/ keeps the promotion small.
_TRAINING_STATE_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "training_args.bin",
    "trainer_state.json",
}


def find_best_checkpoint(trocr_dir: Path) -> Path:
    """Find best epoch's checkpoint-<step>/ via trainer_state.json pointer."""
    checkpoints = sorted(
        [p for p in trocr_dir.glob("checkpoint-*") if p.is_dir()],
        key=checkpoint_step,
    )
    if not checkpoints:
        raise SystemExit(
            f"No checkpoint-* directories under {trocr_dir} — nothing to "
            "recover. Was train_trocr ever started, or was the TrOCR output "
            "directory already cleaned?",
        )
    for cp in reversed(checkpoints):
        state_path = cp / "trainer_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            continue
        bmc = state.get("best_model_checkpoint")
        if not bmc:
            continue
        name = Path(bmc).name
        match = next((c for c in checkpoints if c.name == name), None)
        if match is not None:
            log.info("best_model_checkpoint=%s (from %s)", name, state_path)
            return match
        log.warning(
            "trainer_state.json points at %s but no such dir exists; "
            "falling back to highest-numbered checkpoint.", bmc,
        )
        break
    fallback = checkpoints[-1]
    log.warning("Using fallback best-checkpoint = %s", fallback.name)
    return fallback


def _already_promoted(trocr_dir: Path) -> bool:
    has_weights = any(
        (trocr_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    has_processor = any(
        (trocr_dir / name).exists()
        for name in ("tokenizer.json", "vocab.json", "preprocessor_config.json")
    )
    return has_weights and has_processor


def promote_trocr(config: ExpConfig) -> Path:
    """Promote best TrOCR checkpoint into results/trocr/ and resave processor."""
    trocr_dir = Path(config.output_dir) / "trocr"
    if not trocr_dir.is_dir():
        raise SystemExit(f"Expected {trocr_dir} to exist — nothing to recover.")
    if _already_promoted(trocr_dir):
        log.info("%s already looks promoted; skipping.", trocr_dir)
        for cp in trocr_dir.glob("checkpoint-*"):
            if cp.is_dir():
                rmtree_quiet(cp)
        return trocr_dir

    best = find_best_checkpoint(trocr_dir)
    # Delete non-best checkpoints FIRST so we have headroom to resave the
    # processor afterwards. Critical at 100 % disk use.
    freed = 0
    for cp in trocr_dir.glob("checkpoint-*"):
        if cp.is_dir() and cp.resolve() != best.resolve():
            freed += rmtree_quiet(cp)
    for name in _TRAINING_STATE_FILES:
        p = best / name
        if p.is_file():
            freed += p.stat().st_size
            p.unlink()
    log.info("Freed ~%.1f MiB from non-best TrOCR checkpoints.",
             freed / (1024 * 1024))

    # Rename is constant-time on the same filesystem and doesn't double disk.
    for item in best.iterdir():
        dst = trocr_dir / item.name
        if dst.exists():
            if dst.is_file() or dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        os.rename(item, dst)
    best.rmdir()
    log.info("Promoted best checkpoint files into %s", trocr_dir)

    # Re-save processor (tokenizer + image processor) — what the failed
    # ``processor.save_pretrained(out_dir)`` line would have done. Reloading
    # ``config.trocr_model`` gives a byte-identical processor since we
    # never mutated it during training.
    from transformers import TrOCRProcessor
    processor = TrOCRProcessor.from_pretrained(config.trocr_model)
    processor.save_pretrained(trocr_dir)
    log.info("Wrote processor artifacts to %s", trocr_dir)
    return trocr_dir
