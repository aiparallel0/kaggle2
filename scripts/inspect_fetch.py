"""fetch sub-command: rsync results from a remote vast.ai host.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: pulls JSON metrics, YOLO logs, and paper outputs from a remote
    training host (typically vast.ai) for local inspection.
"""
from __future__ import annotations

import argparse
import logging
import shlex
from datetime import UTC, datetime
from pathlib import Path

from scripts.inspect_fetch_util import (
    build_ssh_cmd,
    capture,
    remote_env,
    rsync_filters,
    run_with_retry,
    silent_call,
)

log = logging.getLogger("inspect")
_ = capture  # re-exported for historical callers; silence ruff F401


def _run_fetch(args: argparse.Namespace) -> None:
    ssh_binary, host = build_ssh_cmd(args)
    local_root = Path(args.local_root or f"./vastai_dump/"
                      f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    local_root.mkdir(parents=True, exist_ok=True)
    log.info("Remote host:  %s", host)
    log.info("Remote root:  %s", args.remote_root)
    log.info("Local dest:   %s", local_root)
    log.info("With weights: %s", "yes" if args.with_weights else "no")

    ssh_cmd_str = shlex.join(ssh_binary)
    remote_spec = f"{host}:{args.remote_root.rstrip('/')}/"
    base_cmd = [
        "rsync", "-avh", "--partial", "--progress",
        "-e", ssh_cmd_str, *rsync_filters(args.with_weights),
        remote_spec, f"{local_root}/",
    ]
    if args.dry_run:
        log.info("Dry run — rsync command:\n  %s",
                 shlex.join(base_cmd + ["--dry-run"]))
        run_with_retry(base_cmd + ["--dry-run"], retries=0)
        return
    run_with_retry(base_cmd, retries=args.retries)

    # Best-effort log pull; tolerate missing paths.
    for pattern in (f"{args.remote_root}/nohup.out",
                    f"{args.remote_root}/*.log",
                    f"{args.remote_root}/results/*.log"):
        silent_call([
            "rsync", "-avh", "--ignore-missing-args",
            "-e", ssh_cmd_str, f"{host}:{pattern}", f"{local_root}/logs/",
        ])

    env_out = remote_env(ssh_binary, host, args.remote_root)
    (local_root / "remote_env.txt").write_text(env_out)
    log.info("Wrote remote_env.txt (%d bytes)", len(env_out))
    log.info("Done. Local dump: %s", local_root)


def add_fetch(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
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
