"""Regex weak-labeller: assigns 4-class tags to SROIE box lines.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: future multi-class YOLO head experiment (nc=4 instead of nc=1).
    Currently unused in the paper's binary text-line detector.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Conservative, ASCII-only patterns. SROIE is printed English + Malay; these
# match the overwhelming majority of keyed lines without false positives on
# store names (which contain no ``:`` and no trailing decimal).
_MONEY_RE = re.compile(r"(?:RM|MYR|\$)\s*\d+[.,]\d{2}|\b\d+[.,]\d{2}\b")
_DATE_RE = re.compile(
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{8}\b",
)
_KV_RE = re.compile(r"\S+\s*:\s*\S")

HEADER_BAND = 0.15  # top fraction of the image considered header


def _classify(text: str, cy_norm: float) -> int:
    """Assign 4-class id: header (0) > money (2) > kv (1) > address (3)."""
    if cy_norm < HEADER_BAND:
        return 0
    if _MONEY_RE.search(text):
        return 2
    if _DATE_RE.search(text) or _KV_RE.search(text):
        return 1
    return 3


def _label_receipt(box_path: Path) -> list[int] | None:
    """Return one class id per box line; None if unreadable."""
    try:
        raw_lines = box_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    img_h_hint = 1.0  # SROIE box files are pixel-space; we normalise below.
    # We cannot read image size without Pillow in this hot path. Use the
    # max y across all lines as the receipt height proxy — good enough for
    # the ``cy < 0.15`` header band cutoff since SROIE images are portrait.
    ys: list[float] = []
    parsed: list[tuple[list[int], str]] = []
    for raw in raw_lines:
        parts = raw.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            coords = [int(p) for p in parts[:8]]
        except ValueError:
            continue
        ys.extend(coords[1::2])
        parsed.append((coords, parts[8]))
    if not parsed:
        return []
    img_h_hint = max(ys) if ys else 1.0
    labels: list[int] = []
    for coords, text in parsed:
        cy = (min(coords[1::2]) + max(coords[1::2])) / 2.0
        labels.append(_classify(text, cy / max(img_h_hint, 1.0)))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Weak-label SROIE box files into 4 classes.")
    parser.add_argument("--data", default="data/sroie_cache", help="SROIE cache root")
    parser.add_argument("--out", default="data/sroie/class_labels.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap number of receipts (0 = all). Default Phase-1 target: 200.")
    args = parser.parse_args()

    box_dir = Path(args.data) / "0325updated.task1train(626p)" / "box"
    if not box_dir.exists():
        # Fall back to any dir named 'box' underneath --data.
        candidates = list(Path(args.data).rglob("box"))
        if not candidates:
            raise SystemExit(f"No SROIE box/ dir under {args.data}")
        box_dir = candidates[0]

    receipts = sorted(box_dir.glob("*.txt"))
    if args.limit > 0:
        receipts = receipts[: args.limit]

    out: dict[str, list[int]] = {}
    for box_path in receipts:
        labels = _label_receipt(box_path)
        if labels is None:
            continue
        out[box_path.stem] = labels

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {len(out)} receipts × weak class labels → {out_path}")


if __name__ == "__main__":
    main()
