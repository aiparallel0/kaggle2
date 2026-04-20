"""Unified inspection CLI for a trained kaggle2 run.

Three subcommands share a single codebase so downstream analysis can mix and
match: ``diagnose`` and ``parity`` run locally against the current
checkpoints, ``fetch`` pulls a finished run off a remote host.

Subcommands
-----------

* ``diagnose`` — per-receipt dump of YOLO boxes, TrOCR transcripts, learned +
  rule-based field assignments, and aggregate pipeline health signals (avg
  boxes/receipt, TrOCR empty rate, fallback rate, per-field F1 on the
  inspected subset).

* ``parity`` — run the exact function ``main.py --stage eval`` runs
  (``models.pipeline_eval.eval_pipeline``) on the full test split and print
  the aggregate F1 numbers.

* ``fetch`` — rsync the relevant artefacts from a vast.ai (or any
  SSH-reachable) host to a timestamped directory on the local machine.

Each subcommand prints its own ``--help`` and can be invoked as::

    python scripts/inspect.py diagnose --n 20
    python scripts/inspect.py parity
    python scripts/inspect.py fetch --ssh-cmd "ssh -p 12345 root@ssh4.vast.ai"
    python scripts/inspect.py fetch -p 12345 root@ssh4.vast.ai --with-weights
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.inspect_diagnose import add_diagnose  # noqa: E402
from scripts.inspect_fetch import add_fetch  # noqa: E402
from scripts.inspect_parity import add_parity  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/inspect.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    add_diagnose(sub)
    add_parity(sub)
    add_fetch(sub)
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
