"""Emit the per-run ``MANIFEST.json`` — the definitive file index.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: walk ``<run_dir>/`` after ``stage_paper`` finishes, compute
    ``sha256``/``size_bytes``/``mtime_utc``/``producer_stage`` for every
    file, and write ``<run_dir>/MANIFEST.json``.  Reviewers use this
    manifest to verify a ``pack_run.sh`` archive round-trip and to spot
    missing artefacts without running the pipeline themselves.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import TypedDict

# Subdirectory → producer stage mapping.  Files whose parent does not
# match any prefix get tagged ``unknown`` (e.g. legacy flat writes).
_STAGE_BY_DIR: dict[str, str] = {
    "metrics": "eval",
    "curves": "train",
    "predictions": "eval",
    "attention": "eval",
    "figures": "paper",
    "paper": "paper",
    "env": "bootstrap",
}


class ManifestEntry(TypedDict):
    """One row in ``MANIFEST.json`` — see module docstring for semantics."""

    relpath: str
    size_bytes: int
    sha256: str
    mtime_utc: str
    producer_stage: str


class Manifest(TypedDict):
    """Top-level ``MANIFEST.json`` schema."""

    schema_version: int
    run_id: str
    generated_utc: str
    file_count: int
    total_bytes: int
    entries: list[ManifestEntry]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _producer_stage(relpath: Path) -> str:
    """Infer the stage that wrote ``relpath`` from its top-level dir."""
    if not relpath.parts:
        return "unknown"
    top = relpath.parts[0]
    return _STAGE_BY_DIR.get(top, "unknown")


def build_manifest(run_dir: Path, run_id: str) -> Manifest:
    """Walk ``run_dir`` and return a fully-populated :class:`Manifest`."""
    entries: list[ManifestEntry] = []
    total = 0
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "MANIFEST.json":
            # Never self-enumerate: an existing MANIFEST from a prior
            # paper pass would poison the sha256 of the new manifest.
            continue
        rel = path.relative_to(run_dir)
        size = path.stat().st_size
        entries.append({
            "relpath": rel.as_posix(),
            "size_bytes": size,
            "sha256": _sha256(path),
            "mtime_utc": _dt.datetime.fromtimestamp(
                path.stat().st_mtime, tz=_dt.UTC,
            ).isoformat(timespec="seconds"),
            "producer_stage": _producer_stage(rel),
        })
        total += size
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "file_count": len(entries),
        "total_bytes": total,
        "entries": entries,
    }


def write_manifest(run_dir: Path, run_id: str) -> Path:
    """Compute + write ``<run_dir>/MANIFEST.json``; return its path."""
    manifest = build_manifest(run_dir, run_id)
    out = run_dir / "MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    return out
