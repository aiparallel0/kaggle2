"""Canonical SROIE Task-3 test split downloader (347 images + GT).

Fetches ICDAR-2019 SROIE Task-3 test images + KIE ground truth from
``rrc.cvc.uab.es`` (primary) with the docTR mirror as fallback,
sha256-verifies before extraction, and lays files out under
``<data_dir>/test/{img,entities}/`` — the hierarchical convention
:func:`data.sroie._canonical_test_split` already consumes. Idempotent.
"""
from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from core.errors import DataError
from core.types import ExpConfig

__all__ = ["ensure_canonical_test_set", "_TASK3_TEST_COUNT"]

log = logging.getLogger("kaggle2")

_TASK3_TEST_COUNT = 347

# docTR community mirror sha256 — pinned upstream by Mindee. The primary
# RRC URLs do not publish a stable digest; we accept whatever ships when
# the primary succeeds (count check below catches truncated downloads)
# but reject the docTR fallback unless the digest matches exactly.
_DOCTR_MIRROR_SHA256 = (
    "41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2"
)


def _sha256_file(path: Path) -> str:
    """Stream a file through sha256; return hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, timeout: float = 60.0) -> None:
    """HTTP GET ``url`` → ``dst``; raise DataError on any failure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("canonical-SROIE: GET %s", url)
    req = urllib.request.Request(
        url, headers={"User-Agent": "kaggle2/1.0 (+https://github.com/aiparallel0/kaggle2)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dst.open("wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataError(f"canonical-SROIE download failed: {url} → {exc}") from exc


def _verify_sha256(path: Path, expected: str | None, label: str) -> None:
    """Compare ``path`` digest to ``expected``; raise on mismatch.

    ``expected=None`` skips (used for primary RRC URLs which publish no
    pinned digest); the post-extract count assertion catches truncation.
    """
    if expected is None:
        return
    got = _sha256_file(path)
    if got.lower() != expected.lower():
        raise DataError(
            f"canonical-SROIE {label} sha256 mismatch: "
            f"expected {expected[:16]}…, got {got[:16]}…",
        )


def _extract_zip(zip_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dst)


def _flatten_files(src_root: Path, dst: Path, exts: tuple[str, ...]) -> int:
    """Move every file under ``src_root`` with one of ``exts`` into ``dst``.

    Recursive because ICDAR archives nest one folder deep
    (``task3-test(347p)/<id>.jpg``).  Returns the placed-file count.
    """
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            target = dst / p.name
            if not target.exists():
                p.replace(target)
            n += 1
    return n


# Path-segment pairs that identify SROIE Task-3 entity directories in the
# three known mirror layouts: docTR single-ZIP, RRC dual-ZIP, zzzDavid flat.
_ENTITY_PATH_PARTS: tuple[tuple[str, str], ...] = (
    ("test", "entities"),
    ("test", "key"),
    ("data", "key"),
)


def _is_task3_entity_path(p: Path) -> bool:
    """Return True iff ``p`` lives under a Task-3 entity subdirectory."""
    parts = p.parts
    return any(
        len(parts) >= 3 and parts[-3] == a and parts[-2] == b
        for a, b in _ENTITY_PATH_PARTS
    )


def _flatten_json_entities(src_root: Path, dst: Path) -> int:
    """Move Task-3 JSON entity .txt files from ``src_root`` into ``dst``.

    Path-scoped: only considers files under ``*/test/entities/``,
    ``*/test/key/``, or ``*/data/key/`` — rejects ``task1_2_test/box/``
    and other Task-1/2 directories. Content-scoped: first non-whitespace
    byte must be ``{`` (JSON dict). Returns count of files placed.
    """
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_root.rglob("*.txt"):
        if not p.is_file() or not _is_task3_entity_path(p):
            continue
        try:
            head = p.read_text(errors="ignore").lstrip()[:1]
        except OSError:
            continue
        if head != "{":
            continue
        target = dst / p.name
        if not target.exists():
            p.replace(target)
        n += 1
    return n


def _try_primary(workdir: Path, urls: tuple[str, str]) -> tuple[Path, Path] | None:
    """Try the RRC primary URLs; return (img_zip, gt_zip) or None on failure."""
    img_zip, gt_zip = workdir / "sroie_test_img.zip", workdir / "sroie_test_gt.zip"
    try:
        _download(urls[0], img_zip)
        _download(urls[1], gt_zip)
    except DataError as exc:
        log.warning("canonical-SROIE primary mirror failed (%s); trying fallback.", exc)
        return None
    return img_zip, gt_zip


def _try_mirror(workdir: Path, mirror_url: str) -> Path | None:
    """Try the docTR mirror (single ZIP containing both img + GT)."""
    mirror_zip = workdir / "sroie_test_mirror.zip"
    try:
        _download(mirror_url, mirror_zip)
        _verify_sha256(mirror_zip, _DOCTR_MIRROR_SHA256, "docTR mirror")
    except DataError as exc:
        log.warning("canonical-SROIE docTR mirror failed (%s).", exc)
        return None
    return mirror_zip


def ensure_canonical_test_set(config: ExpConfig, data_path: Path) -> Path:
    """Download + extract the 347-image SROIE Task-3 test set under ``data_path``.

    Produces ``<data_path>/test/img/<id>.jpg`` (×347) and
    ``<data_path>/test/entities/<id>.txt`` (×347, SROIE key:value
    format).  Idempotent: returns immediately when ``test/img/``
    already has ≥347 JPGs.  Raises :class:`DataError` if neither
    mirror is reachable or the post-extract count is wrong.
    """
    img_dir = data_path / "test" / "img"
    ent_dir = data_path / "test" / "entities"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) >= _TASK3_TEST_COUNT:
        return data_path
    primary_urls = (config.canonical_sroie_test_url, config.canonical_sroie_gt_url)
    mirror_url = config.canonical_sroie_mirror_url
    workdir = data_path / "_canonical_dl"
    workdir.mkdir(parents=True, exist_ok=True)
    primary = _try_primary(workdir, primary_urls)
    if primary is not None:
        img_zip, gt_zip = primary
        extract_dir = workdir / "primary"
        _extract_zip(img_zip, extract_dir)
        _extract_zip(gt_zip, extract_dir)
    else:
        mirror_zip = _try_mirror(workdir, mirror_url)
        if mirror_zip is None:
            raise DataError(
                "canonical-SROIE: both primary (rrc.cvc.uab.es) and fallback "
                "(docTR mirror) failed — set canonical_sroie_enabled=false to "
                "use the 500/63/63 internal split, or pre-populate "
                "<data_dir>/test/{img,entities}/.",
            )
        extract_dir = workdir / "mirror"
        _extract_zip(mirror_zip, extract_dir)
    n_img = _flatten_files(extract_dir, img_dir, (".jpg", ".jpeg"))
    n_ent = _flatten_json_entities(extract_dir, ent_dir)
    if n_img < _TASK3_TEST_COUNT or n_ent < _TASK3_TEST_COUNT:
        raise DataError(
            f"canonical-SROIE post-extract count mismatch: {n_img} jpg / "
            f"{n_ent} txt — expected ≥{_TASK3_TEST_COUNT} of each.",
        )
    log.info("canonical-SROIE ready: %d images + %d entity files under %s",
             n_img, n_ent, data_path / "test")
    return data_path
