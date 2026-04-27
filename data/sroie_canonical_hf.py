"""HuggingFace Task-3 SROIE materialiser — schema-agnostic, Donut-aware.

Called by :func:`data.sroie_canonical.ensure_canonical_test_set` when the
RRC primary fails.  Downloads ``Metric-AI/icdar_sroie`` (or the repo
configured via ``canonical_sroie_hf_repo``) using
``huggingface_hub.snapshot_download``, picks the split that has exactly
347 rows (the canonical Task-3 test count), parses the Donut-style
``ground_truth`` JSON cell on each row, and writes ``<stem>.jpg`` +
``<stem>.json`` files into ``workdir/hf/{img,entities}/``.

Schema is **content-driven**, not column-name-driven, so the materialiser
survives upstream renames without code changes:

* image cell  — raw bytes, PIL Image, HF ``{"bytes": …}`` struct, path str.
* stem cell   — file_name / id / image_id, else HF image["path"], else
  a deterministic ``f"X{idx:08d}"`` index fallback.  Stem fidelity to RRC
  is best-effort; identity preservation is enforced by
  :data:`data.sroie_canonical._RRC_TASK3_GT_SHA256` (content), not by
  stem matching alone.
* ground_truth cell — JSON string / dict in any of ``ground_truth``,
  ``gt_parse``, ``label``, ``text``; supports the ``{"gt_parse": {...}}``
  and ``{"gt_parses": [{...}]}`` envelopes; falls back to flat columns.

Public entry point
------------------
``try_huggingface(workdir, repo_id, revision)``  — called by
:func:`~data.sroie_canonical.ensure_canonical_test_set`.

Operator sanity script
----------------------
``python -m data.sroie_canonical_hf --peek <repo> <revision>`` prints the
split sizes, column names, and the first row's ``ground_truth`` value so
upstream schema drift can be diagnosed in seconds.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.errors import DataError

if TYPE_CHECKING:
    from datasets import Dataset

log = logging.getLogger("kaggle2")

_TASK3_TEST_COUNT = 347

# ── schema-agnostic column resolution ───────────────────────────────────────
_IMAGE_COLS = ("image", "img", "image_bytes", "receipt_image", "jpg")
_STEM_COLS = ("file_name", "filename", "id", "image_id", "receipt_id", "stem")
_GT_JSON_COLS = ("ground_truth", "gt_parse", "label", "text")
_FIELD_KEYS = ("company", "date", "address", "total")
_FIELD_KEY_VARIANTS: dict[str, tuple[str, ...]] = {
    "company": ("company", "COMPANY", "Company"),
    "date":    ("date",    "DATE",    "Date"),
    "address": ("address", "ADDRESS", "Address"),
    "total":   ("total",   "TOTAL",   "Total"),
}

# Guard against missing huggingface_hub at import time (mypy-strict safe).
try:
    from huggingface_hub import snapshot_download as _snapshot_download
    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _snapshot_download = None  # type: ignore[assignment]  # guarded by _HF_AVAILABLE
    _HF_AVAILABLE = False


def _extract_stem(row: dict[str, object], idx: int) -> str:
    """Derive a file stem from *row*, falling back to ``f"X{idx:08d}"``.

    Tries explicit stem columns first, then HF image-struct ``path``, then
    the deterministic index fallback.  See module docstring on stem fidelity.
    """
    for col in _STEM_COLS:
        v = row.get(col)
        if v and str(v).strip():
            stem = str(v).strip()
            return stem.rsplit(".", 1)[0] if "." in stem else stem
    img = row.get("image")
    if isinstance(img, dict):
        path = img.get("path")
        if isinstance(path, str) and path.strip():
            return Path(path).stem
    return f"X{idx:08d}"


def _extract_image_bytes(row: dict[str, object]) -> bytes | None:
    """Return raw JPEG bytes from the row image cell, or *None* if absent."""
    for col in _IMAGE_COLS:
        v = row.get(col)
        if v is None:
            continue
        # HF Image feature: {"bytes": ..., "path": ...}
        if isinstance(v, dict):
            raw = v.get("bytes")
            if isinstance(raw, bytes | bytearray):
                raw_b = bytes(raw)
                # Fast path: bytes already carry the JPEG SOI marker → return
                # as-is (the most common HF-mirrored receipt case).
                if raw_b[:3] == b"\xff\xd8\xff":
                    return raw_b
                # Non-JPEG payload (e.g. PNG): re-encode via PIL so the
                # ".jpg" filename stays truthful.  When PIL is unavailable
                # or the bytes cannot be decoded, skip the row (per D3:
                # any image failure → skip + count) rather than silently
                # writing a non-JPEG payload to a .jpg file.
                try:
                    import PIL.Image
                    img = PIL.Image.open(io.BytesIO(raw_b))
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG")
                    return buf.getvalue()
                except (ImportError, OSError):
                    log.warning(
                        "canonical-SROIE HF: non-JPEG image bytes could not "
                        "be re-encoded; skipping row.",
                    )
                    return None
        if isinstance(v, bytes | bytearray):
            return bytes(v)
        # PIL Image object (HF auto-decodes when default features are used)
        try:
            import PIL.Image
            if isinstance(v, PIL.Image.Image):
                buf = io.BytesIO()
                v.convert("RGB").save(buf, format="JPEG")
                return buf.getvalue()
        except ImportError:
            pass
        # Path string — rare but possible with some dataset builders
        if isinstance(v, str | Path):
            p = Path(str(v))
            if p.is_file():
                return p.read_bytes()
    return None


def _extract_fields(row: dict[str, object]) -> dict[str, str]:
    """Extract the four KIE fields from a flat-column row (variant casing)."""
    out: dict[str, str] = {}
    for field, variants in _FIELD_KEY_VARIANTS.items():
        for col in variants:
            v = row.get(col)
            if v is not None:
                out[field] = str(v)
                break
        else:
            out[field] = ""
    return out


def _extract_gt(row: dict[str, object]) -> dict[str, str] | None:
    """Return ``{company,date,address,total}`` from a Donut-style row, or None.

    Accepts ``str`` (parse JSON), ``dict`` (use directly), and the
    ``{"gt_parse": {...}}`` / ``{"gt_parses": [{...}]}`` envelopes Donut
    fine-tuners commonly emit.  Falls back to flat-column extraction so
    legacy mirrors keep working.  Returns ``None`` only when the row
    cannot be coerced into all four fields with non-empty string values.
    """
    raw: object = None
    for col in _GT_JSON_COLS:
        if col in row and row[col]:
            raw = row[col]
            break
    obj: object
    if raw is None:
        # No JSON-style cell — try flat-column fallback (legacy schema).
        flat = _extract_fields(row)
        if all(flat.get(k) for k in _FIELD_KEYS):
            return flat
        return None
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        obj = raw
    else:
        return None
    # Donut convention: {"gt_parse": {...}} or {"gt_parses": [{...}]}.
    if isinstance(obj, dict) and "gt_parse" in obj and isinstance(obj["gt_parse"], dict):
        obj = obj["gt_parse"]
    if (
        isinstance(obj, dict)
        and "gt_parses" in obj
        and isinstance(obj["gt_parses"], list)
        and obj["gt_parses"]
        and isinstance(obj["gt_parses"][0], dict)
    ):
        obj = obj["gt_parses"][0]
    if not isinstance(obj, dict):
        return None
    obj_lower = {str(k).lower(): str(v) for k, v in obj.items()}
    out: dict[str, str] = {}
    for canonical, variants in _FIELD_KEY_VARIANTS.items():
        for v in variants:
            if v.lower() in obj_lower and obj_lower[v.lower()]:
                out[canonical] = obj_lower[v.lower()]
                break
        if canonical not in out:
            return None  # missing field — row is unusable
    return out


def _select_canonical_split(snap_path: Path) -> Dataset:
    """Pick the split (test/validation/train) with exactly 347 rows."""
    from datasets import load_dataset
    data_dir = snap_path / "data"
    if not data_dir.is_dir():
        # Some snapshots ship parquet at the repo root.
        data_dir = snap_path
    ds_dict = load_dataset("parquet", data_dir=str(data_dir))
    for split_name in ("test", "validation", "train"):
        if split_name in ds_dict and len(ds_dict[split_name]) == _TASK3_TEST_COUNT:
            log.info(
                "canonical-SROIE HF: using split=%s (%d rows)",
                split_name, len(ds_dict[split_name]),
            )
            return ds_dict[split_name]
    sizes = {k: len(v) for k, v in ds_dict.items()}
    raise DataError(
        f"canonical-SROIE HF: no split with exactly {_TASK3_TEST_COUNT} rows. "
        f"Splits seen: {sizes}",
    )


def _materialize_rows(snap_dir: Path, img_dir: Path, ent_dir: Path) -> int:
    """Read the canonical split from *snap_dir* → write jpg+json. Returns row count."""
    try:
        ds = _select_canonical_split(snap_dir)
    except ImportError:
        log.warning(
            "canonical-SROIE HF: 'datasets' library not installed "
            "(pip install datasets).",
        )
        return 0
    n_extracted = 0
    n_skipped_img = 0
    n_skipped_gt = 0
    bad_gt_samples: list[str] = []
    for idx, row in enumerate(ds):
        row_d: dict[str, object] = dict(row)
        stem = _extract_stem(row_d, idx)
        try:
            img_bytes = _extract_image_bytes(row_d)
        except (OSError, ValueError, KeyError, AttributeError):
            # Per D3: any failure on a single image → skip + count.  Narrow
            # to the exception types image decoding can plausibly raise so
            # programming errors (e.g. wrong row shape) still surface.
            img_bytes = None
        if not img_bytes:
            n_skipped_img += 1
            continue
        fields = _extract_gt(row_d)
        if fields is None:
            n_skipped_gt += 1
            if len(bad_gt_samples) < 3:
                raw_gt = next(
                    (str(row_d[c])[:200] for c in _GT_JSON_COLS if c in row_d), "<absent>",
                )
                bad_gt_samples.append(f"row[{idx}]: {raw_gt!r}")
            continue
        (img_dir / f"{stem}.jpg").write_bytes(img_bytes)
        (ent_dir / f"{stem}.json").write_text(
            json.dumps(fields, ensure_ascii=False), encoding="utf-8",
        )
        n_extracted += 1
    if n_extracted < _TASK3_TEST_COUNT:
        sample_blob = "\n  ".join(bad_gt_samples) or "<none captured>"
        raise DataError(
            f"canonical-SROIE HF: only materialised {n_extracted}/{_TASK3_TEST_COUNT} "
            f"rows (skipped {n_skipped_img} for missing image, {n_skipped_gt} for "
            f"unparseable ground_truth). First unparseable rows:\n  {sample_blob}",
        )
    return n_extracted


def try_huggingface(workdir: Path, repo_id: str, revision: str) -> Path | None:
    """Download *repo_id* canonical split and materialise img+entities dirs.

    Returns ``workdir/hf`` on success, *None* on import/download/parse
    failure.  Re-raises :class:`~core.errors.DataError` on schema
    violations (caller must treat those as hard failures, not retries).
    """
    if not _HF_AVAILABLE:
        log.warning(
            "canonical-SROIE HF mirror skipped: huggingface_hub not installed "
            "(pip install huggingface_hub).",
        )
        return None
    log.info(
        "canonical-SROIE: trying HuggingFace fallback (repo=%s rev=%s)",
        repo_id, revision,
    )
    work_hf = workdir / "hf"
    snap_dir = work_hf / "_snap"
    img_dir = work_hf / "img"
    ent_dir = work_hf / "entities"
    img_dir.mkdir(parents=True, exist_ok=True)
    ent_dir.mkdir(parents=True, exist_ok=True)
    try:
        snap_path = _snapshot_download(
            repo_id=repo_id, revision=revision,
            repo_type="dataset", local_dir=str(snap_dir),
        )
    except Exception as exc:  # noqa: BLE001  # network errors vary across hf_hub versions
        log.warning("canonical-SROIE HF download failed (%s).", exc)
        return None
    try:
        n = _materialize_rows(Path(snap_path), img_dir, ent_dir)
    except DataError:
        raise  # schema errors are hard failures — propagate to caller
    except Exception as exc:  # noqa: BLE001  # parquet/PIL/fs errors are all retryable
        log.warning("canonical-SROIE HF materialise failed (%s).", exc)
        return None
    if n == 0:
        log.warning("canonical-SROIE HF: 0 rows materialised from %s", snap_path)
        return None
    return work_hf


# --------------------------------------------------------------------------- #
# Operator sanity script — diagnose upstream schema drift in 5 seconds.        #
# Usage: python -m data.sroie_canonical_hf --peek <repo_id> <revision>         #
# --------------------------------------------------------------------------- #

def _peek(repo_id: str, revision: str) -> int:  # pragma: no cover
    """Print split sizes, column names, and first row's ground_truth."""
    if not _HF_AVAILABLE:
        print("huggingface_hub not installed; pip install huggingface_hub")
        return 2
    import tempfile

    from datasets import load_dataset
    with tempfile.TemporaryDirectory() as td:
        snap = _snapshot_download(
            repo_id=repo_id, revision=revision,
            repo_type="dataset", local_dir=td,
        )
        snap_p = Path(snap)
        data_dir = snap_p / "data" if (snap_p / "data").is_dir() else snap_p
        ds_dict = load_dataset("parquet", data_dir=str(data_dir))
        print(f"repo={repo_id} revision={revision}")
        for split_name, ds in ds_dict.items():
            print(f"  split={split_name}: rows={len(ds)} columns={list(ds.column_names)}")
            if len(ds) > 0:
                first: dict[str, Any] = dict(ds[0])
                gt: object = next(
                    (first[c] for c in _GT_JSON_COLS if c in first), "<no gt-style col>",
                )
                gt_str = str(gt)
                if len(gt_str) > 400:
                    gt_str = gt_str[:400] + "…"
                print(f"    first ground_truth: {gt_str}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Peek at HF SROIE schema (sizes, columns, first ground_truth).",
    )
    ap.add_argument(
        "--peek", nargs=2, metavar=("REPO", "REVISION"), required=True,
        help="HuggingFace dataset repo id + revision (e.g. Metric-AI/icdar_sroie main).",
    )
    args = ap.parse_args()
    sys.exit(_peek(args.peek[0], args.peek[1]))
