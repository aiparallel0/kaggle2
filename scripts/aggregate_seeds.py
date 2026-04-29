"""Aggregate per-seed pipeline_metrics.json into mean ± std table.

Run after a multi-seed sweep (local or vast.ai swarm) to produce
the headline mean / std / 95% CI table that ``configs/canonical_5seed.json``
implies but doesn't compute.

Usage:
    # Local sweep (already extracted run dirs)
    python scripts/aggregate_seeds.py runs/sweep-20260429*-seed*/

    # Cloud sweep (tar.zst archives downloaded to ./runs/<sweep>/)
    python scripts/aggregate_seeds.py runs/sweep-20260429*/

The script auto-extracts ``.tar.zst`` archives in-place when it
detects them, so a freshly-rclone'd swarm output works directly.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

FIELD_KEYS = ("company", "date", "address", "total")


def _maybe_extract(p: Path) -> Path:
    """If ``p`` is a ``.tar.zst``, extract next to it and return the dir."""
    if p.suffix == ".zst" and p.name.endswith(".tar.zst"):
        target = p.parent / p.name[: -len(".tar.zst")]
        if not target.exists():
            print(f"Extracting {p.name} → {target}", file=sys.stderr)
            target.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                ["tar", "--use-compress-program=zstd", "-xf", str(p),
                 "-C", str(target)],
            )
        return target
    return p


def _load_metrics(run_dir: Path) -> dict[str, float] | None:
    """Find ``pipeline_metrics.json`` somewhere under ``run_dir``."""
    candidates = [
        run_dir / "pipeline_metrics.json",
        run_dir / "metrics" / "pipeline_metrics.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    # Recursive search as a last resort.
    for p in run_dir.rglob("pipeline_metrics.json"):
        with open(p) as f:
            return json.load(f)
    return None


def _ci_half_width(values: list[float], confidence: float = 0.95) -> float:
    """Normal-approximation half-width.  Returns 0 for n<2."""
    if len(values) < 2:
        return 0.0
    s = statistics.stdev(values)
    z = 1.96 if confidence == 0.95 else 2.576
    return z * s / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Per-seed run directories or .tar.zst archives")
    parser.add_argument("--out", default=None, help="Write aggregated JSON here")
    args = parser.parse_args()

    rows: list[dict[str, float]] = []
    for raw in args.paths:
        path = _maybe_extract(Path(raw))
        if not path.exists():
            print(f"SKIP missing path: {raw}", file=sys.stderr)
            continue
        # If the path itself is the run directory, use it; otherwise
        # search one level deep for the actual run dir.
        if (path / "pipeline_metrics.json").exists() or (path / "metrics").exists():
            run_dirs = [path]
        else:
            run_dirs = [p for p in path.iterdir() if p.is_dir()]
        for d in run_dirs:
            m = _load_metrics(d)
            if m is None:
                print(f"  no metrics in {d}", file=sys.stderr)
                continue
            row = {
                "run": d.name,
                "f1": float(m.get("assigner_f1", 0.0)),
                "ned": float(m.get("assigner_ned", 0.0)),
                "em": float(m.get("assigner_em", 0.0)),
                "rules_f1": float(m.get("rulebased_f1", 0.0)),
            }
            for fk in FIELD_KEYS:
                row[f"f1_{fk}"] = float(
                    m.get("per_field_f1", {}).get(fk, 0.0),
                )
            rows.append(row)

    if not rows:
        print("No metrics found.", file=sys.stderr)
        sys.exit(2)

    # Print human-readable table.
    cols = ["f1", "ned", "em", "rules_f1"] + [f"f1_{fk}" for fk in FIELD_KEYS]
    print(f"\n{'run':<48} " + " ".join(f"{c:>10}" for c in cols))
    print("-" * (48 + 11 * len(cols)))
    for r in rows:
        print(f"{r['run']:<48} " + " ".join(f"{r[c]:>10.4f}" for c in cols))

    # Aggregate.
    agg: dict[str, dict[str, float]] = {}
    for c in cols:
        vals = [r[c] for r in rows]
        agg[c] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "ci95": _ci_half_width(vals, 0.95),
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    print(f"\n{'metric':<10} {'mean':>10} {'std':>10} {'±ci95':>10} "
          f"{'min':>10} {'max':>10} {'n':>4}")
    print("-" * 66)
    for c in cols:
        a = agg[c]
        print(f"{c:<10} {a['mean']:>10.4f} {a['std']:>10.4f} "
              f"{a['ci95']:>10.4f} {a['min']:>10.4f} {a['max']:>10.4f} "
              f"{a['n']:>4}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"per_run": rows, "aggregate": agg}, f, indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
