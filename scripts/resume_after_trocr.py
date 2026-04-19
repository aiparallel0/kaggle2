"""Disk-full recovery for a run that crashed at ``trainer.save_model`` in
``train_trocr``.

The TrOCR ``Seq2SeqTrainingArguments`` uses ``save_strategy="epoch"`` with
``load_best_model_at_end=True``, which means each epoch wrote a
``results/trocr/checkpoint-<step>/`` directory before the final
``trainer.save_model(out_dir)`` attempted to serialise the restored
best-checkpoint on top of ``results/trocr/`` and hit ENOSPC.  Everything we
need is still on disk — the best epoch's weights plus ``trainer_state.json``
identifying it — so this script:

  1. Frees disk: per-epoch DONUT checkpoints, the YOLO staging image mirror,
     pip/apt caches, non-best TrOCR checkpoints.
  2. Identifies the best TrOCR checkpoint via ``trainer_state.json``'s
     ``best_model_checkpoint`` pointer (falls back to the highest-numbered
     checkpoint if absent).
  3. ``os.rename``-moves the best checkpoint's files up into
     ``results/trocr/`` so no data is copied (stays on the same filesystem,
     no temporary doubling of disk usage — critical when disk is already
     full).
  4. Re-downloads and saves the TrOCRProcessor (tokenizer + image processor)
     that would have been written by the failed
     ``processor.save_pretrained(out_dir)`` call.
  5. Runs the remaining training steps that ``_stage_train`` never reached:
     ``train_assigner`` and ``_write_pipeline_meta``.

Idempotent — safe to rerun if a later step fails.  Intended entry point is
``scripts/resume_after_trocr.sh`` which layers an apt/pip cache purge and
invokes ``main.py --stage eval`` + ``--stage paper`` after this script
completes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.seed import seed_everything  # noqa: E402
from core.types import AssignerData, ExpConfig  # noqa: E402
from data.sroie import (  # noqa: E402
    download_sroie,
    extract_crops,
    extract_receipt_regions,
    load_or_create_split,
)
from models.assigner_train import train_assigner  # noqa: E402

log = logging.getLogger("resume")

# File names written by Seq2SeqTrainer inside each checkpoint dir that we
# can safely drop once the best epoch has been promoted — they're training
# state for resumption, not model weights.  Keeping them around after the
# run is a pure disk-cost with no upside.
_TRAINING_STATE_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "training_args.bin",
    "trainer_state.json",
}


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _find_best_checkpoint(trocr_dir: Path) -> Path:
    """Return the best epoch's ``checkpoint-<step>/`` under *trocr_dir*.

    Prefers the pointer written by HuggingFace
    (``trainer_state.json::best_model_checkpoint``) and falls back to the
    highest-numbered checkpoint when the state file is missing or the
    pointer does not match any on-disk directory (e.g. partial disk-full
    write).
    """
    checkpoints = sorted(
        [p for p in trocr_dir.glob("checkpoint-*") if p.is_dir()],
        key=_checkpoint_step,
    )
    if not checkpoints:
        raise SystemExit(
            f"No checkpoint-* directories under {trocr_dir} — nothing to "
            "recover.  Was train_trocr ever started, or was the TrOCR "
            "output directory already cleaned?",
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


def _rmtree_quiet(path: Path) -> int:
    """``shutil.rmtree`` that returns bytes freed and ignores ENOENT."""
    if not path.exists():
        return 0
    try:
        freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        freed = 0
    shutil.rmtree(path, ignore_errors=True)
    return freed


def _cleanup_disk(config: ExpConfig) -> None:
    root = Path(config.output_dir)
    freed = 0
    # DONUT per-epoch checkpoints — only the outer directory (best restored
    # via load_best_model_at_end) is needed for eval.
    donut_dir = root / "donut"
    if donut_dir.is_dir():
        for cp in donut_dir.glob("checkpoint-*"):
            if cp.is_dir():
                freed += _rmtree_quiet(cp)
    # YOLO staging: a full copy of every SROIE training image next to YOLO's
    # label .txt files.  Regenerated deterministically from the split on the
    # next train run, so safe to nuke.
    freed += _rmtree_quiet(root / "yolo_data")
    # pip + apt caches.  Tolerate failure — we're root on vast.ai but the
    # script should still make progress on stricter setups.
    subprocess.run(
        ["pip", "cache", "purge"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["apt-get", "clean"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log.info("Freed ~%.1f MiB from DONUT checkpoints + yolo_data/.",
             freed / (1024 * 1024))


def _promote_trocr(config: ExpConfig) -> Path:
    """Promote best TrOCR checkpoint into ``results/trocr/`` and resave processor."""
    trocr_dir = Path(config.output_dir) / "trocr"
    if not trocr_dir.is_dir():
        raise SystemExit(f"Expected {trocr_dir} to exist — nothing to recover.")
    # Already promoted?  Both weights and processor artifacts present ->
    # skip.  We detect by presence of model.safetensors (or pytorch_model.bin)
    # AND a tokenizer artifact at the top level.
    has_weights = any(
        (trocr_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    has_processor = any(
        (trocr_dir / name).exists()
        for name in ("tokenizer.json", "vocab.json", "preprocessor_config.json")
    )
    if has_weights and has_processor:
        log.info("%s already looks promoted; skipping.", trocr_dir)
        # Still remove any stragglers.
        for cp in trocr_dir.glob("checkpoint-*"):
            if cp.is_dir():
                _rmtree_quiet(cp)
        return trocr_dir

    best = _find_best_checkpoint(trocr_dir)

    # Delete the non-best checkpoints FIRST so we have disk headroom to
    # (re)save the processor afterwards.  Critical when we're already at
    # 100 % disk use — copying the best checkpoint would ENOSPC again.
    freed = 0
    for cp in trocr_dir.glob("checkpoint-*"):
        if cp.is_dir() and cp.resolve() != best.resolve():
            freed += _rmtree_quiet(cp)
    # Also drop training-state sidecars inside the best checkpoint before
    # moving it up — we don't need them for inference.
    for name in _TRAINING_STATE_FILES:
        p = best / name
        if p.is_file():
            freed += p.stat().st_size
            p.unlink()
    log.info("Freed ~%.1f MiB from non-best TrOCR checkpoints.",
             freed / (1024 * 1024))

    # Move (rename) each file/dir from best/ into trocr_dir/.  rename is
    # constant-time on the same filesystem and does not double disk usage.
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

    # Re-save the processor (tokenizer + image processor).  This is what
    # the failed ``processor.save_pretrained(out_dir)`` line would have
    # done.  Reloading ``config.trocr_model`` from the hub/cache gives a
    # byte-identical processor since we never mutated it during training.
    from transformers import TrOCRProcessor

    processor = TrOCRProcessor.from_pretrained(config.trocr_model)
    processor.save_pretrained(trocr_dir)
    log.info("Wrote processor artifacts to %s", trocr_dir)
    return trocr_dir


def _remaining_train_steps(config: ExpConfig) -> None:
    """Run the train-stage steps that crashed short of: assigner + meta."""
    data_path = download_sroie(config)
    split_cache = Path(config.output_dir) / "split.json"
    data = load_or_create_split(data_path, config.seed, split_cache)
    log.info("Split cache %s → %d train / %d val / %d test",
             split_cache, len(data.train), len(data.val), len(data.test))
    crops = extract_crops(data.train, config.fields)
    regions = extract_receipt_regions(data.train, config.fields)
    if not crops:
        raise SystemExit(
            "No labeled SROIE crops after resume — check data/ cache integrity.",
        )
    log.info("%d crops / %d region-groups", len(crops), len(regions))

    assigner_marker = Path(config.output_dir) / "assigner.pt"
    if assigner_marker.exists():
        log.info("%s already exists — skipping train_assigner.", assigner_marker)
    else:
        trocr_path = str(Path(config.output_dir) / "trocr")
        out = train_assigner(
            config,
            AssignerData(trocr_path=trocr_path, crops=crops, regions=regions),
        )
        log.info("Assigner → %s", out)

    meta_path = Path(config.output_dir) / "pipeline_meta.json"
    meta_path.write_text(json.dumps({"yolo_img_size": config.yolo_img_size}))
    log.info("pipeline_meta.json → %s", meta_path)


def _print_disk(prefix: str) -> None:
    try:
        usage = shutil.disk_usage(".")
        log.info("%s disk free=%.1f GiB / total=%.1f GiB",
                 prefix,
                 usage.free / (1024 ** 3),
                 usage.total / (1024 ** 3))
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--skip-cleanup", action="store_true",
        help="Skip disk cleanup (DONUT checkpoints, yolo_data, caches).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    seed_everything(config.seed)

    _print_disk("before cleanup:")
    if not args.skip_cleanup:
        _cleanup_disk(config)
    _promote_trocr(config)
    _print_disk("after promotion:")
    _remaining_train_steps(config)
    _print_disk("after resume:")


if __name__ == "__main__":
    main()
