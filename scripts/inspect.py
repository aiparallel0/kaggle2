"""Unified inspection CLI for a trained kaggle2 run.

Three subcommands share a single codebase so downstream analysis can
mix and match: ``diagnose`` and ``parity`` run locally against the
current checkpoints, ``fetch`` pulls a finished run off a remote host.

Subcommands
-----------

* ``diagnose`` — per-receipt dump of YOLO boxes, TrOCR transcripts,
  learned + rule-based field assignments, and aggregate pipeline
  health signals (average boxes/receipt, TrOCR empty rate, fallback
  rate, per-field F1 on the inspected subset). Use this when pipeline
  F1 is unexpectedly low and you want to see what the model actually
  produced for the first N receipts.

* ``parity`` — run the exact function ``main.py --stage eval`` runs
  (``models.pipeline_eval.eval_pipeline``) on the full test split and
  print the aggregate F1 numbers. This is the source of truth for
  "what would main.py print right now", minus the ~23-minute DONUT
  stage. Use this to confirm a reported-low F1 is reproducible.

* ``fetch`` — rsync the relevant artefacts (results JSONs, YOLO run
  logs, TrOCR/DONUT config + tokenizer, assigner checkpoint, paper
  PDF, recent stdout/stderr logs) from a vast.ai (or any SSH-reachable)
  host to a timestamped directory on the local machine. Heavy weights
  (500 MB + 1.4 GB) are excluded unless ``--with-weights`` is passed.

Each subcommand prints its own ``--help`` and can be invoked as::

    python scripts/inspect.py diagnose --n 20
    python scripts/inspect.py parity
    python scripts/inspect.py fetch --ssh-cmd "ssh -p 12345 root@ssh4.vast.ai"
    python scripts/inspect.py fetch -p 12345 root@ssh4.vast.ai --with-weights
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


if TYPE_CHECKING:  # avoid importing torch just to render --help
    pass


log = logging.getLogger("inspect")


# ===========================================================================
# diagnose
# ===========================================================================

def _count_sroie_gt_boxes(image_path: Path) -> int:
    """SROIE provides per-line box annotations in box/<stem>.txt. Count them.

    If YOLO's eval-time detection count is << this number, the detector
    collapsed at the configured imgsz and every downstream component
    inherits a starved region list.
    """
    box_path = image_path.parent.parent / "box" / (image_path.stem + ".txt")
    if not box_path.exists():
        return 0
    n = 0
    for line in box_path.read_text(errors="replace").splitlines():
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            [int(p) for p in parts[:8]]
        except ValueError:
            continue
        n += 1
    return n


def _load_pipeline_models(
    paths: Any, n_fields: int,
) -> tuple[Any, Any, Any, Any, str]:
    """Load YOLO, TrOCR, assigner — shared between diagnose and parity."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from ultralytics import YOLO

    from models.attention_assign import load_assigner

    yolo = YOLO(paths.yolo)
    trocr_proc = TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(paths.trocr)
    assigner = load_assigner(paths.assigner, n_fields=n_fields)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    assigner = assigner.to(device)
    trocr_model.eval()
    assigner.eval()
    return yolo, trocr_proc, trocr_model, assigner, device


def _run_diagnose(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    from core.config import load_config
    from core.metrics import compute_metrics
    from core.types import Field, PipelinePaths, Prediction
    from data.sroie import download_sroie, load_or_create_split
    from models.attention_assign import text_priors
    from models.pipeline_eval import _assign_learned, _is_usable_region
    from models.rule_based import rule_based_assign

    config = load_config(args.config)
    data_path = download_sroie(config)
    split_cache = Path(config.output_dir) / "split.json"
    data = load_or_create_split(data_path, config.seed, split_cache)
    paths = PipelinePaths(
        yolo=str(Path(config.output_dir) / "yolo" / "run" / "weights" / "best.pt"),
        trocr=str(Path(config.output_dir) / "trocr"),
        assigner=str(Path(config.output_dir) / "assigner.pt"),
    )
    yolo, trocr_proc, trocr_model, assigner, device = _load_pipeline_models(
        paths, len(config.fields),
    )
    meta_path = Path(config.output_dir) / "pipeline_meta.json"
    yolo_img = config.yolo_img_size
    if meta_path.exists():
        yolo_img = int(json.loads(meta_path.read_text()).get("yolo_img_size", yolo_img))
    log.info("yolo_img=%d conf=%.2f max_regions=%d",
             yolo_img, config.yolo_conf, config.max_regions_per_image)

    split_name = args.split
    receipts = getattr(data, split_name)
    n = min(args.n, len(receipts))
    subset_receipts = receipts[:n]
    out_receipts: list[dict[str, Any]] = []
    total_boxes = 0
    total_gt_boxes = 0
    total_usable = 0
    total_reads = 0
    fallback_count = 0
    all_empty_assign = 0
    preds_rule: list[Prediction] = []
    preds_assign: list[Prediction] = []

    with torch.no_grad():
        for rec in subset_receipts:
            img = Image.open(rec.image_path).convert("RGB")
            results = yolo.predict(
                str(rec.image_path), imgsz=yolo_img, conf=config.yolo_conf, verbose=False,
            )
            raw_boxes = (
                [b[:4] for b in results[0].boxes.xyxyn.cpu().tolist()]
                if results[0].boxes else []
            )
            raw_boxes.sort(key=lambda b: b[1])  # top-to-bottom
            gt_box_count = _count_sroie_gt_boxes(rec.image_path)
            total_boxes += len(raw_boxes)
            total_gt_boxes += gt_box_count
            used_fallback = False
            regions: list[dict[str, Any]] = []
            texts: list[str] = []
            feats: list[Any] = []
            bboxes: list[list[float]] = []
            w, h = img.width, img.height
            for box in raw_boxes[: config.max_regions_per_image]:
                x1, y1, x2, y2 = box
                crop = img.crop(
                    (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                )
                if crop.width < 1 or crop.height < 1:
                    continue
                pv = trocr_proc(images=crop, return_tensors="pt").pixel_values.to(device)
                enc = trocr_model.encoder(pv).last_hidden_state
                out = trocr_model.generate(pv, max_new_tokens=config.trocr_max_new_tokens)
                text = trocr_proc.batch_decode(out, skip_special_tokens=True)[0]
                total_reads += 1
                usable = _is_usable_region(text)
                regions.append({
                    "bbox": [round(c, 4) for c in [x1, y1, x2, y2]],
                    "text": text,
                    "usable": usable,
                })
                if not usable:
                    continue
                total_usable += 1
                texts.append(text)
                feats.append(enc.mean(dim=1))
                bboxes.append([x1, y1, x2, y2])
            if not texts:
                fallback_count += 1
                used_fallback = True
                pv = trocr_proc(images=img, return_tensors="pt").pixel_values.to(device)
                enc = trocr_model.encoder(pv).last_hidden_state
                full_out = trocr_model.generate(pv, max_new_tokens=config.trocr_max_new_tokens)
                full_text = trocr_proc.batch_decode(full_out, skip_special_tokens=True)[0]
                texts = [full_text]
                feats = [enc.mean(dim=1)]
                bboxes = [[0.0, 0.0, 1.0, 1.0]]
                regions.append({"bbox": [0.0, 0.0, 1.0, 1.0], "text": full_text,
                                "_fallback": True, "usable": True})
            learned = _assign_learned(assigner, texts, feats, bboxes,
                                      config.fields, device)
            rule = rule_based_assign(texts, bboxes)
            if not learned:
                all_empty_assign += 1
            gt = {f.name.lower(): f.value for f in rec.fields}
            rid = rec.image_path.stem
            preds_assign.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in learned.items()],
            ))
            preds_rule.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in rule.items()],
            ))
            # Optionally embed text_priors for the first 3 regions so the user
            # can eyeball whether the priors look well-calibrated.
            for r in regions[:3]:
                if r.get("usable"):
                    r["priors"] = [round(p, 3) for p in text_priors(r["text"])]
            out_receipts.append({
                "receipt_id": rid,
                "yolo_n_boxes": len(raw_boxes),
                "n_usable": len(texts),
                "sroie_gt_n_boxes": gt_box_count,
                "fallback_used": used_fallback,
                "regions": regions,
                "assigner": learned,
                "rulebased": rule,
                "gt": gt,
            })
            log.info("%s: yolo=%d usable=%d gt_lines=%d learned=%s rule=%s",
                     rid, len(raw_boxes), len(texts), gt_box_count,
                     list(learned.keys()), list(rule.keys()))

    m_rule = compute_metrics(preds_rule, subset_receipts, config.fields)
    m_assign = compute_metrics(preds_assign, subset_receipts, config.fields)
    summary = {
        "n_receipts": n,
        "split": split_name,
        "avg_boxes_per_receipt": round(total_boxes / n, 2) if n else 0.0,
        "avg_sroie_gt_boxes_per_receipt": round(total_gt_boxes / n, 2) if n else 0.0,
        "fallback_rate": round(fallback_count / n, 3) if n else 0.0,
        "usable_region_rate": (
            round(total_usable / total_reads, 3) if total_reads else 0.0
        ),
        "all_fields_empty_rate": round(all_empty_assign / n, 3) if n else 0.0,
        "yolo_img": yolo_img,
        "yolo_conf": config.yolo_conf,
        "max_regions": config.max_regions_per_image,
        "subset_assigner_f1": round(m_assign.global_f1, 4),
        "subset_rulebased_f1": round(m_rule.global_f1, 4),
        "subset_assigner_per_field_f1": {
            k: round(v, 4) for k, v in m_assign.per_field_f1.items()
        },
        "subset_rulebased_per_field_f1": {
            k: round(v, 4) for k, v in m_rule.per_field_f1.items()
        },
    }
    report = {"summary": summary, "receipts": out_receipts}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Wrote %s", out_path)
    print(json.dumps(summary, indent=2))


# ===========================================================================
# parity
# ===========================================================================

def _run_parity(args: argparse.Namespace) -> None:
    from core.config import load_config
    from core.types import PipelinePaths
    from data.sroie import download_sroie, load_or_create_split
    from models.pipeline_eval import eval_pipeline

    config = load_config(args.config)
    data_path = download_sroie(config)
    split_cache = Path(config.output_dir) / "split.json"
    data = load_or_create_split(data_path, config.seed, split_cache)
    receipts = getattr(data, args.split)
    paths = PipelinePaths(
        yolo=os.path.join(config.output_dir, "yolo", "run", "weights", "best.pt"),
        trocr=os.path.join(config.output_dir, "trocr"),
        assigner=os.path.join(config.output_dir, "assigner.pt"),
    )
    log.info("Running eval_pipeline on %d %s receipts...", len(receipts), args.split)
    pm = eval_pipeline(paths, receipts, config)
    report = {
        "split": args.split,
        "assigner_global_f1": pm.assigner.global_f1,
        "assigner_per_field_f1": pm.assigner.per_field_f1,
        "rulebased_global_f1": pm.rulebased.global_f1,
        "rulebased_per_field_f1": pm.rulebased.per_field_f1,
    }
    log.info("RESULT (via eval_pipeline — same function main.py calls):")
    print(json.dumps(report, indent=2))


# ===========================================================================
# fetch
# ===========================================================================

_FETCH_INCLUDES = [
    "results/",
    "results/*.json",
    "results/yolo/",
    "results/yolo/run/",
    "results/yolo/run/**",
    "results/donut/",
    "results/donut/*.json",
    "results/donut/generation_config.json",
    "results/donut/config.json",
    "results/donut/preprocessor_config.json",
    "results/donut/tokenizer*",
    "results/donut/special_tokens_map.json",
    "results/donut/added_tokens.json",
    "results/trocr/",
    "results/trocr/*.json",
    "results/trocr/preprocessor_config.json",
    "results/trocr/tokenizer*",
    "results/trocr/vocab.json",
    "results/trocr/merges.txt",
    "results/trocr/special_tokens_map.json",
    "results/assigner.pt",
    "report/",
    "report/paper_filled.tex",
    "report/paper_filled.pdf",
]

_FETCH_EXCLUDES_BASE = [
    "results/yolo_data/",           # staging mirror of SROIE images, huge, regen
    "results/yolo/run/weights/last.pt",
    "results/**/checkpoint-*",      # per-epoch checkpoints — training artefacts only
]

_FETCH_WEIGHT_PATTERNS = [
    "results/donut/model.safetensors",
    "results/donut/pytorch_model.bin",
    "results/trocr/model.safetensors",
    "results/trocr/pytorch_model.bin",
    "results/yolo/run/weights/best.pt",
]


def _build_ssh_cmd(args: argparse.Namespace) -> tuple[list[str], str]:
    """Parse --ssh-cmd / (--port, target) → (ssh-binary-args, remote-host)."""
    if args.ssh_cmd:
        parts = shlex.split(args.ssh_cmd)
        if not parts:
            raise SystemExit("--ssh-cmd must not be empty")
        # Last token is the host; everything before is the ssh invocation.
        host = parts[-1]
        ssh_binary = parts[:-1] if len(parts) > 1 else ["ssh"]
        return ssh_binary, host
    if not args.target:
        raise SystemExit(
            "fetch: provide user@host (and optionally -p PORT) or --ssh-cmd '...'",
        )
    ssh_binary = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if args.port:
        ssh_binary += ["-p", str(args.port)]
    return ssh_binary, args.target


def _run_fetch(args: argparse.Namespace) -> None:
    ssh_binary, host = _build_ssh_cmd(args)

    local_root = Path(args.local_root or f"./vastai_dump/"
                      f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    local_root.mkdir(parents=True, exist_ok=True)
    log.info("Remote host:      %s", host)
    log.info("Remote root:      %s", args.remote_root)
    log.info("Local dest:       %s", local_root)
    log.info("With weights:     %s", "yes" if args.with_weights else "no")

    includes = list(_FETCH_INCLUDES)
    excludes = list(_FETCH_EXCLUDES_BASE)
    if args.with_weights:
        includes += _FETCH_WEIGHT_PATTERNS
    else:
        excludes += [
            "results/donut/model.safetensors",
            "results/donut/pytorch_model.bin",
            "results/trocr/model.safetensors",
            "results/trocr/pytorch_model.bin",
            "results/yolo/run/weights/*.pt",
        ]
    excludes.append("*")  # deny everything else

    filter_args: list[str] = []
    for p in includes:
        filter_args += [f"--include={p}"]
    for p in excludes:
        filter_args += [f"--exclude={p}"]

    ssh_cmd_str = shlex.join(ssh_binary)
    remote_spec = f"{host}:{args.remote_root.rstrip('/')}/"
    base_cmd = [
        "rsync", "-avh", "--partial", "--progress",
        "-e", ssh_cmd_str, *filter_args,
        remote_spec, f"{local_root}/",
    ]
    if args.dry_run:
        log.info("Dry run — rsync command:\n  %s", shlex.join(base_cmd + ["--dry-run"]))
        _run_with_retry(base_cmd + ["--dry-run"], retries=0)
        return

    _run_with_retry(base_cmd, retries=args.retries)

    # Loose logs — best-effort, tolerate missing paths.
    log_patterns = [
        f"{args.remote_root}/nohup.out",
        f"{args.remote_root}/*.log",
        f"{args.remote_root}/results/*.log",
    ]
    for pattern in log_patterns:
        cmd = [
            "rsync", "-avh", "--ignore-missing-args",
            "-e", ssh_cmd_str,
            f"{host}:{pattern}", f"{local_root}/logs/",
        ]
        _silent_call(cmd)

    # Remote environment snapshot.
    env_script = (
        "echo == date ==; date -u;"
        "echo == uname ==; uname -a;"
        f"echo == df ==; df -h {args.remote_root} || true;"
        "echo == nvidia-smi ==; nvidia-smi || true;"
        "echo == python ==; python --version 2>&1;"
        "echo == torch ==; python -c "
        "\"import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())\" "
        "2>&1 || true;"
        f"echo == git HEAD ==; (cd {args.remote_root} && git rev-parse HEAD && git status --porcelain);"
        f"echo == ls results ==; ls -la {args.remote_root}/results 2>&1 | head -200;"
    )
    env_cmd = [*ssh_binary, host, "bash", "-lc", env_script]
    env_out = _capture(env_cmd)
    (local_root / "remote_env.txt").write_text(env_out)
    log.info("Wrote remote_env.txt (%d bytes)", len(env_out))
    log.info("Done. Local dump: %s", local_root)


def _run_with_retry(cmd: list[str], retries: int) -> None:
    """Run ``cmd`` with exponential-backoff retries on failure."""
    delay = 2.0
    attempts = 0
    while True:
        attempts += 1
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempts > retries:
                raise SystemExit(
                    f"Command failed after {attempts} attempts: {exc}",
                ) from exc
            log.warning(
                "Attempt %d/%d failed (exit %d) — retrying in %.0fs",
                attempts, retries + 1, exc.returncode, delay,
            )
            time.sleep(delay)
            delay *= 2


def _silent_call(cmd: list[str]) -> None:
    with contextlib.suppress(OSError):
        subprocess.run(cmd, check=False, capture_output=True)


def _capture(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"<error: {exc}>\n"
    return (result.stdout or "") + (result.stderr or "")


# ===========================================================================
# CLI
# ===========================================================================

def _add_diagnose(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "diagnose",
        help="Dump raw pipeline intermediates for the first N receipts.",
        description=(
            "Dump per-receipt YOLO boxes, TrOCR transcriptions, and both "
            "assignment strategies (learned + rule-based) for the first N "
            "test receipts. Writes a JSON report and prints aggregate "
            "health signals (avg boxes/receipt, TrOCR empty rate, "
            "fallback rate, per-field F1 on the inspected subset)."
        ),
    )
    p.add_argument("--config", default="config.json")
    p.add_argument("--n", type=int, default=10,
                   help="Number of receipts to dump (default 10).")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which split to inspect (default test).")
    p.add_argument("--out", default="results/diagnose.json",
                   help="Path to write the JSON report to.")
    p.set_defaults(func=_run_diagnose)


def _add_parity(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "parity",
        help="Run eval_pipeline on the full split and print aggregate F1.",
        description=(
            "Invoke models.pipeline_eval.eval_pipeline on the full split "
            "— the same function main.py --stage eval uses — and print "
            "the global + per-field F1 numbers for both the learned "
            "assigner and the rule-based baseline. Use to confirm a "
            "reported F1 is reproducible without running the 23-minute "
            "DONUT stage."
        ),
    )
    p.add_argument("--config", default="config.json")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which split to evaluate (default test).")
    p.set_defaults(func=_run_parity)


def _add_fetch(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "fetch",
        help="rsync result artefacts from a remote host.",
        description=(
            "Pull the JSON metrics, YOLO logs, TrOCR/DONUT configs, "
            "assigner checkpoint, and paper outputs from a remote "
            "(typically vast.ai) host over SSH. Heavy model weights "
            "(500 MB + 1.4 GB) are excluded unless --with-weights is "
            "passed. Writes to a timestamped directory unless "
            "--local-root is given."
        ),
    )
    p.add_argument("target", nargs="?",
                   help="ssh target, e.g. root@ssh4.vast.ai")
    p.add_argument("-p", "--port", type=int,
                   help="SSH port (convenience; same as 'ssh -p PORT').")
    p.add_argument("--ssh-cmd", dest="ssh_cmd",
                   help="Full SSH invocation incl. host (e.g. "
                        "'ssh -p 12345 root@ssh4.vast.ai'). "
                        "Overrides target + port.")
    p.add_argument("--remote-root", default="/workspace/kaggle2",
                   help="Remote repo path (default /workspace/kaggle2).")
    p.add_argument("--local-root", default=None,
                   help="Local destination (default ./vastai_dump/<ts>/).")
    p.add_argument("--with-weights", action="store_true",
                   help="Include DONUT + TrOCR + YOLO weight blobs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the rsync command without transferring files.")
    p.add_argument("--retries", type=int, default=4,
                   help="Retry count for network-related rsync failures.")
    p.set_defaults(func=_run_fetch)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/inspect.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_diagnose(sub)
    _add_parity(sub)
    _add_fetch(sub)
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
