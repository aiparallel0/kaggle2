"""Canonical SROIE Task-3 test split downloader (347 images + GT).

Per-mirror :class:`MirrorAdapter` pattern: each adapter declares its
archive's image / entity globs, the path-segment marker that scopes
collection to the legitimate Task-3 sub-tree (no Task-1 boxes, no
thumbnails), and the upstream entity content format
(``sroie_kv`` JSON-content key:value text, or pure ``json``).  This
makes image-side and entity-side scoping symmetric per mirror — the
fix for the asymmetry that produced ``360 jpg / 0 txt`` after PR #90.

SSL is hardened with :mod:`certifi` so the RRC primary actually
reaches ``rrc.cvc.uab.es`` on fresh vast.ai containers.  The post-
extract count check is exact (``== 347``), not ``>= 347``.

Identity-preserving pins
------------------------
``_RRC_TASK3_STEMS_SHA256`` and ``_RRC_TASK3_GT_SHA256`` start as the
sentinel ``"PIN_ON_FIRST_RUN"``.  On first successful download, the
observed hashes are written to ``canonical_status.json``.  To lock
them, run::

    python -m data.sroie_canonical --pin-stems /path/to/canonical_status.json

and paste the printed values into the constants below.  Once pinned,
any drift (upstream revision or label change) raises :class:`DataError`.

HuggingFace revision pinning
-----------------------------
Operators: after the first successful download on a non-firewalled host,
pin ``canonical_sroie_hf_revision`` in ``config.json`` to the specific
commit SHA shown in ``runs/<run_id>/env/canonical_status.json`` under
key ``hf_revision_used``.  Until pinned, "main" tracks the live HEAD of
``Metric-AI/icdar_sroie`` which could change silently.
"""
from __future__ import annotations

import hashlib
import json
import logging
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from core.errors import DataError
from core.types import ExpConfig

__all__ = [
    "CanonicalStatus",
    "MirrorAdapter",
    "ensure_canonical_test_set",
    "_TASK3_TEST_COUNT",
    "_RRC_TASK3_STEMS_SHA256",
    "_RRC_TASK3_GT_SHA256",
    "_stems_sha256",
    "_gt_content_sha256",
    "_verify_sha256",
    "_RRC_ADAPTER",
    "_HF_ADAPTER",
]

log = logging.getLogger("kaggle2")

_TASK3_TEST_COUNT = 347

# Sentinel value — replace with the hex digest printed by --pin-stems once
# you have a successful canonical download on a non-firewalled host.
_PIN_SENTINEL = "PIN_ON_FIRST_RUN"

# sha256 over "\n".join(sorted(stems)).encode().
# Detects stem-set drift between upstream revisions.
# Regenerate: python -m data.sroie_canonical --pin-stems <canonical_status.json>
_RRC_TASK3_STEMS_SHA256 = _PIN_SENTINEL  # TODO: pin after first successful run

# sha256 over the normalised GT content (see _gt_content_sha256).
# Detects label drift between upstream revisions.
# Regenerate: python -m data.sroie_canonical --pin-stems <canonical_status.json>
_RRC_TASK3_GT_SHA256 = _PIN_SENTINEL  # TODO: pin after first successful run


@dataclass(frozen=True)
class CanonicalStatus:
    """Outcome of :func:`ensure_canonical_test_set` (also persisted as JSON)."""

    mirror_used: str
    n_img_collected: int
    n_ent_collected: int
    fallback_triggered: bool = False
    stems_sha256: str = ""
    gt_content_sha256: str = ""


@dataclass(frozen=True)
class MirrorAdapter:
    """Per-mirror layout adapter — image+entity globs, scoping, format."""

    name: str
    image_glob: str           # rglob pattern, e.g. "*.jpg"
    entity_glob: str          # rglob pattern, e.g. "*.txt" or "*.json"
    image_path_marker: str    # path segment that MUST appear in image path
    entity_path_marker: str   # path segment that MUST appear in entity path
    entity_format: str        # "sroie_kv" | "json" — upstream content format


# RRC dual-zip: images extract under ``task3-test(347p)/``; GT files
# extract under ``test/entities/`` and contain JSON dicts.
_RRC_ADAPTER = MirrorAdapter(
    name="rrc",
    image_glob="*.jpg",
    entity_glob="*.txt",
    image_path_marker="task3-test(347p)",
    entity_path_marker="entities",
    entity_format="sroie_kv",
)

# HuggingFace mirror (Metric-AI/icdar_sroie): materialised by
# _try_huggingface into workdir/hf/{img,entities}/.
_HF_ADAPTER = MirrorAdapter(
    name="huggingface",
    image_glob="*.jpg",
    entity_glob="*.json",
    image_path_marker="img",
    entity_path_marker="entities",
    entity_format="json",
)


def _ssl_context() -> ssl.SSLContext:
    """Default SSL context with certifi's CA bundle when available.

    Vast.ai containers regularly ship without a system CA bundle for
    ``rrc.cvc.uab.es``, which silently demoted the RRC primary to the
    docTR fallback.  Prefer certifi (vendored by pip-installed
    ``requests`` / ``urllib3``); fall back to the platform default.
    Never disables verification.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover — certifi is a hard requirement
        return ssl.create_default_context()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, timeout: float = 60.0) -> None:
    """HTTP GET ``url`` → ``dst``; raise :class:`DataError` on any failure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("canonical-SROIE: GET %s", url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "kaggle2/1.0 (+https://github.com/aiparallel0/kaggle2)"},
    )
    ctx = _ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp, \
                dst.open("wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        msg = f"canonical-SROIE download failed: {url} → {exc}"
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            msg += (
                " — install/upgrade certifi (`pip install --upgrade certifi`) "
                "or refresh the system CA bundle (`apt install ca-certificates`)."
            )
        raise DataError(msg) from exc


def _verify_sha256(path: Path, expected: str | None, label: str) -> None:
    """Compare ``path`` digest to ``expected``; raise on mismatch.

    ``expected=None`` skips verification (used for unpinned RRC URLs);
    the post-extract exact-count check catches truncation downstream.
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


def _has_marker(p: Path, marker: str) -> bool:
    return not marker or marker in p.parts


def _place_files(
    src_root: Path, dst: Path, adapter: MirrorAdapter, *, is_entity: bool,
) -> int:
    """Move adapter-scoped files from ``src_root`` into ``dst``.

    Path-scoped via ``adapter.{image,entity}_path_marker`` so Task-1
    box files / thumbnail directories cannot leak in.  Entity files
    are additionally content-gated: first non-whitespace byte must be
    ``{`` (JSON dict) — a belt-and-braces guard for layout drift.
    Returns count of files placed.
    """
    glob_pat = adapter.entity_glob if is_entity else adapter.image_glob
    marker = adapter.entity_path_marker if is_entity else adapter.image_path_marker
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_root.rglob(glob_pat):
        if not p.is_file() or not _has_marker(p, marker):
            continue
        if is_entity:
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


def _try_primary(workdir: Path, urls: tuple[str, str]) -> Path | None:
    """RRC dual-zip fetch; return extract dir or None on failure."""
    img_zip, gt_zip = workdir / "rrc_img.zip", workdir / "rrc_gt.zip"
    extract = workdir / "rrc"
    try:
        _download(urls[0], img_zip)
        _download(urls[1], gt_zip)
    except DataError as exc:
        log.warning("canonical-SROIE primary mirror failed (%s); trying HF fallback.", exc)
        return None
    _extract_zip(img_zip, extract)
    _extract_zip(gt_zip, extract)
    return extract


def _stems_sha256(img_dir: Path) -> str:
    """sha256 over ``\\n``.join(sorted stems) — detects test-set identity drift."""
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    return hashlib.sha256("\n".join(stems).encode()).hexdigest()


def _gt_content_sha256(ent_dir: Path) -> str:
    """Mirror-agnostic GT content sha256 over company/date/address/total.

    Handles both ``.json`` (HF) and ``.txt`` (RRC sroie_kv) entity files so
    the hash is comparable across mirrors — both represent the same GT values.
    """
    canon: list[str] = []
    seen: set[str] = set()
    all_files = sorted(
        [*ent_dir.glob("*.json"), *ent_dir.glob("*.txt")],
        key=lambda p: p.stem,
    )
    for p in all_files:
        if p.stem in seen:
            continue
        seen.add(p.stem)
        try:
            text = p.read_text(encoding="utf-8").strip()
            kv: dict[str, str] = {}
            if text.startswith("{"):
                raw = json.loads(text)
                kv = {str(k).lower(): str(v) for k, v in raw.items()}
            else:
                for line in text.splitlines():
                    if ":" not in line:
                        continue
                    k, _, v = line.partition(":")
                    h = k.split(",", 1)[0].strip()
                    if "," in k and h.isdigit():
                        continue
                    kv[k.strip().lower()] = v.strip()
            keep = {k: kv.get(k, "") for k in ("address", "company", "date", "total")}
            canon.append(p.stem + "\x1f" + json.dumps(keep, sort_keys=True, ensure_ascii=False))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return hashlib.sha256("\n".join(canon).encode()).hexdigest()


def ensure_canonical_test_set(
    config: ExpConfig, data_path: Path,
) -> CanonicalStatus:
    """Download + extract the 347-image SROIE Task-3 test set under ``data_path``.

    Produces ``<data_path>/test/img/<id>.jpg`` (×347) and
    ``<data_path>/test/entities/<id>.{txt,json}`` (×347).  Idempotent:
    returns immediately when both directories already hold exactly 347 files.
    Raises :class:`DataError` if neither mirror is reachable, the
    post-extract count is not exactly ``347 / 347``, or a pinned identity
    hash does not match the collected data.
    """
    img_dir = data_path / "test" / "img"
    ent_dir = data_path / "test" / "entities"
    # Tightened idempotent short-circuit: require EXACT counts on BOTH sides.
    # (>= 347 previously allowed a stale 360-jpg run to be treated as cached-OK)
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) == _TASK3_TEST_COUNT:
        n_ent = len(list(ent_dir.iterdir())) if ent_dir.exists() else 0
        if n_ent == _TASK3_TEST_COUNT:
            return CanonicalStatus(
                mirror_used="cached",
                n_img_collected=_TASK3_TEST_COUNT,
                n_ent_collected=n_ent,
            )
    workdir = data_path / "_canonical_dl"
    workdir.mkdir(parents=True, exist_ok=True)
    primary = _try_primary(
        workdir,
        (config.canonical_sroie_test_url, config.canonical_sroie_gt_url),
    )
    if primary is not None:
        adapter, extract_dir = _RRC_ADAPTER, primary
    else:
        from data.sroie_hf import try_huggingface
        hf = try_huggingface(
            workdir,
            repo_id=config.canonical_sroie_hf_repo,
            revision=config.canonical_sroie_hf_revision,
        )
        if hf is None:
            raise DataError(
                "canonical-SROIE: both primary (rrc.cvc.uab.es) and HuggingFace "
                f"fallback (repo={config.canonical_sroie_hf_repo}) failed — set "
                "canonical_sroie_enabled=false to use the 500/63/63 internal split, "
                "or pre-populate <data_dir>/test/{img,entities}/.",
            )
        adapter, extract_dir = _HF_ADAPTER, hf
    n_img = _place_files(extract_dir, img_dir, adapter, is_entity=False)
    n_ent = _place_files(extract_dir, ent_dir, adapter, is_entity=True)
    if n_img != _TASK3_TEST_COUNT or n_ent != _TASK3_TEST_COUNT:
        raise DataError(
            f"canonical-SROIE post-extract count mismatch (mirror={adapter.name}): "
            f"{n_img} img / {n_ent} ent — expected exactly "
            f"{_TASK3_TEST_COUNT}/{_TASK3_TEST_COUNT}.",
        )
    # Compute identity hashes for pin-on-first-run workflow.
    observed_stems_sha = _stems_sha256(img_dir)
    observed_gt_sha = _gt_content_sha256(ent_dir)
    # Enforce pinned values when set (not the placeholder "PIN_ON_FIRST_RUN").
    if (
        _RRC_TASK3_STEMS_SHA256 != _PIN_SENTINEL
        and observed_stems_sha != _RRC_TASK3_STEMS_SHA256
    ):
        raise DataError(
            f"canonical-SROIE stem-set drift (mirror={adapter.name}): "
            f"expected {_RRC_TASK3_STEMS_SHA256[:16]}…, "
            f"got {observed_stems_sha[:16]}… — "
            "an upstream mirror revision changed the test image set. "
            "If intentional, regenerate the pin with "
            "`python -m data.sroie_canonical --pin-stems <canonical_status.json>`.",
        )
    if (
        _RRC_TASK3_GT_SHA256 != _PIN_SENTINEL
        and observed_gt_sha != _RRC_TASK3_GT_SHA256
    ):
        raise DataError(
            f"canonical-SROIE GT content drift (mirror={adapter.name}): "
            f"expected {_RRC_TASK3_GT_SHA256[:16]}…, "
            f"got {observed_gt_sha[:16]}….",
        )
    log.info("canonical-SROIE ready (%s): %d images + %d entity files under %s",
             adapter.name, n_img, n_ent, data_path / "test")
    return CanonicalStatus(
        mirror_used=adapter.name,
        n_img_collected=n_img,
        n_ent_collected=n_ent,
        stems_sha256=observed_stems_sha,
        gt_content_sha256=observed_gt_sha,
    )


# --------------------------------------------------------------------------- #
# Back-compat shims for tests/test_sroie_canonical_gt_format.py.               #
# New code consumes :class:`MirrorAdapter` via ensure_canonical_test_set.      #
# --------------------------------------------------------------------------- #

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
    """Path-scoped + JSON-content-gated entity flatten (legacy Bug 14 gate)."""
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


# --------------------------------------------------------------------------- #
# One-shot pin helper — run after first successful canonical download.          #
# Usage: python -m data.sroie_canonical --pin-stems [path/to/status.json]      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Print canonical identity pins from canonical_status.json.",
    )
    ap.add_argument(
        "--pin-stems", metavar="STATUS_JSON", required=True,
        help="Path to canonical_status.json written by a successful run.",
    )
    args = ap.parse_args()
    payload = json.loads(Path(args.pin_stems).read_text())
    stems_sha = payload.get("stems_sha256", "")
    gt_sha = payload.get("gt_content_sha256", "")
    if not stems_sha or not gt_sha:
        sys.exit(f"missing stems_sha256/gt_content_sha256 in {args.pin_stems}")
    print(f'_RRC_TASK3_STEMS_SHA256 = "{stems_sha}"')
    print(f'_RRC_TASK3_GT_SHA256    = "{gt_sha}"')
