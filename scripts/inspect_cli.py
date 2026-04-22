"""Unified inspection CLI for a trained kaggle2 run.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: three subcommands (diagnose, parity, fetch) for post-training
    diagnostics of the YOLO+TrOCR+Attention pipeline.  Used to verify
    F1 parity and debug silent F1-destroying bugs.
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
        prog="scripts/inspect_cli.py",
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
