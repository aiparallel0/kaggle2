"""HuggingFace Task-3 SROIE materialiser — schema-agnostic column resolution.

Called by :func:`data.sroie_canonical.ensure_canonical_test_set` when the
RRC primary fails.  Downloads ``Metric-AI/icdar_sroie`` (or the repo
configured via ``canonical_sroie_hf_repo``) using
``huggingface_hub.snapshot_download``, reads parquet test shards, and
writes ``<stem>.jpg`` + ``<stem>.json`` files into
``workdir/hf/{img,entities}/``.

Schema-agnostic: handles multiple plausible column-name variants so the
materialiser survives upstream renames without code changes.
  image cell  — raw bytes, PIL Image, HF ``{"bytes":…}`` struct, or path string.
  stem cell   — id, receipt_id, stem, file_name, …
  field cells — company/COMPANY/Company, date/DATE/Date, etc.

Public entry point
------------------
``try_huggingface(workdir, repo_id, revision)``  — called by
:func:`~data.sroie_canonical.ensure_canonical_test_set`.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from core.errors import DataError

log = logging.getLogger("kaggle2")

# ── schema-agnostic column resolution ───────────────────────────────────────
_IMAGE_COLS = ("image", "img", "image_bytes", "receipt_image", "jpg")
_STEM_COLS = ("id", "receipt_id", "stem", "file_name", "filename", "image_id")
_FIELD_COLS: dict[str, tuple[str, ...]] = {
    "company": ("company", "COMPANY", "Company"),
    "date":    ("date",    "DATE",    "Date"),
    "address": ("address", "ADDRESS", "Address"),
    "total":   ("total",   "TOTAL",   "Total"),
}

# Guard against missing huggingface_hub at import time (mypy-strict safe).
# _HF_AVAILABLE lets try_huggingface short-circuit before touching network.
# Per A9: _snapshot_download = None in the except branch; the type:ignore
# suppresses the None-to-callable assignment that mypy correctly flags.
try:
    from huggingface_hub import snapshot_download as _snapshot_download
    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _snapshot_download = None  # type: ignore[assignment]  # guarded by _HF_AVAILABLE
    _HF_AVAILABLE = False


def _extract_stem(row: dict[str, object], idx: int) -> str:
    """Derive a file stem from *row*, falling back to a zero-padded index."""
    for col in _STEM_COLS:
        v = row.get(col)
        if v and str(v).strip():
            stem = str(v).strip()
            return stem.rsplit(".", 1)[0] if "." in stem else stem
    return f"receipt_{idx:05d}"


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
                return bytes(raw)
        if isinstance(v, bytes | bytearray):
            return bytes(v)
        # PIL Image object (pillow installed on most training hosts)
        try:
            import PIL.Image
            if isinstance(v, PIL.Image.Image):
                buf = io.BytesIO()
                v.save(buf, format="JPEG")
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
    """Extract the four KIE fields, trying variant column spellings."""
    out: dict[str, str] = {}
    for field, variants in _FIELD_COLS.items():
        for col in variants:
            v = row.get(col)
            if v is not None:
                out[field] = str(v)
                break
        else:
            out[field] = ""
    return out


def _check_schema(col_names: list[str]) -> None:
    """Raise DataError if ALL four KIE fields are absent from the dataset schema."""
    lower = {c.lower() for c in col_names}
    missing = [f for f in ("company", "date", "address", "total") if f not in lower]
    if len(missing) == 4:
        raise DataError(
            "canonical-SROIE HF: dataset has none of the required KIE fields "
            f"(company/date/address/total). Observed columns: {col_names[:10]}",
        )


def _materialize_rows(snap_dir: Path, img_dir: Path, ent_dir: Path) -> int:
    """Read parquet test shards from *snap_dir* → write jpg+json. Returns row count."""
    try:
        from datasets import Dataset
    except ImportError:
        log.warning(
            "canonical-SROIE HF: 'datasets' library not installed "
            "(pip install datasets).",
        )
        return 0
    test_pqs = sorted(snap_dir.rglob("*test*.parquet"))
    if not test_pqs:
        test_pqs = sorted(snap_dir.rglob("*.parquet"))
    if not test_pqs:
        log.warning("canonical-SROIE HF: no parquet files found in %s", snap_dir)
        return 0
    ds = Dataset.from_parquet([str(p) for p in test_pqs])
    _check_schema(list(getattr(ds, "column_names", [])))
    n = 0
    for idx, row in enumerate(ds):
        row_d: dict[str, object] = dict(row)
        stem = _extract_stem(row_d, idx)
        img_bytes = _extract_image_bytes(row_d)
        if img_bytes:
            (img_dir / f"{stem}.jpg").write_bytes(img_bytes)
        fields = _extract_fields(row_d)
        (ent_dir / f"{stem}.json").write_text(
            json.dumps(fields, ensure_ascii=False), encoding="utf-8",
        )
        n += 1
    return n


def try_huggingface(workdir: Path, repo_id: str, revision: str) -> Path | None:
    """Download *repo_id* test split and materialise img+entities dirs.

    Returns ``workdir/hf`` on success, *None* on import/download/parse
    failure.  Re-raises :class:`~core.errors.DataError` on schema
    violations (caller must treat those as hard failures, not retries).

    Logs an ``INFO`` line before the network call so firewalled operators
    know which host is being contacted.
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
