"""CORD-v2 dataset loader (Korean restaurant receipts).

Project: kaggle2 — Document KIE.
Reference: Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR
    Parsing*, NeurIPS Document Intelligence Workshop, 2019
    (https://github.com/clovaai/cord, CC BY 4.0).
HuggingFace mirror: ``naver-clova-ix/cord-v2`` — official Clova-AI release
    with ``train/validation/test`` splits, image bytes + structured
    ``ground_truth`` JSON describing every field
    (``menu``, ``sub_total``, ``total``, ``store``, ``date`` ...).

Role: closes the ``HONESTY.md §2.2`` gap (single-dataset SROIE only) by
    providing a real second corpus.  CORD shares the receipt domain
    with SROIE but differs structurally: Korean restaurant receipts
    rather than Malaysian/Singaporean retail, 30-field schema vs
    4-field, line-itemised menu (each ``menu`` entry has its own
    ``unitprice`` × ``cnt`` = ``price``) which gives FOCUS-Σ Identity 3
    a richer verification surface than SROIE's flat money lines.

Schema mapping.  CORD's ``ground_truth.gt_parse`` is nested under three
top-level keys (``store``, ``menu``, ``total``).  We expose:

    company  ↔ ``store.name``        (merchant trade name)
    date     ↔ ``date.date`` or ``date`` (receipt date)
    total    ↔ ``total.total_price``  (grand total)
    subtotal ↔ ``total.subtotal_price``
    tax      ↔ ``total.tax_price``
    service  ↔ ``total.service_price`` (when present)
    cash     ↔ ``total.cashprice``
    change   ↔ ``total.changeprice``

CORD has no structured ``address`` field — the address is sometimes
embedded in ``store.name`` or omitted entirely.  Cross-dataset eval
therefore uses the ``CROSS_DATASET_FIELDS`` triple
``{company, date, total}``; the FOCUS-Σ ablation row uses
``CORD_FULL_FIELDS`` for richer subset-sum verification.

Caching.  ``datasets.load_dataset`` caches under
``${HF_HOME}/datasets/`` automatically.  Images are materialised to
``cache_dir/cord-<split>/NNNN.png`` so downstream code can open them
by path (matching the SROIE shape).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.types import ExpConfig, Field, Receipt

log = logging.getLogger("kaggle2.data.cord")

# CORD schema → SROIE-comparable field mapping.  ``None`` on the SROIE
# side means CORD has no equivalent (skip in cross-dataset eval).
CORD_TO_SROIE: dict[str, str | None] = {
    "store.name": "company",
    "date.date": "date",
    "date": "date",
    "total.total_price": "total",
    "total.subtotal_price": None,
    "total.tax_price": None,
    "total.service_price": None,
    "total.cashprice": None,
    "total.changeprice": None,
}

CROSS_DATASET_FIELDS: tuple[str, ...] = ("company", "date", "total")
CORD_FULL_FIELDS: tuple[str, ...] = (
    "company", "date", "total",
    "subtotal", "tax", "service", "cash", "change",
)

__all__ = [
    "CORD_FULL_FIELDS",
    "CORD_TO_SROIE",
    "CROSS_DATASET_FIELDS",
    "CordReceipt",
    "cord_to_sroie_receipts",
    "load_cord",
    "load_cord_full_schema",
    "load_cord_split",
]


@dataclass
class CordReceipt:
    """One CORD-v2 receipt: image path + parsed-JSON ground truth.

    ``raw_gt`` is the unwrapped CORD ``ground_truth.gt_parse`` object.
    Use the property accessors for the SROIE-comparable fields and
    :meth:`item_prices` to get the per-line money values FOCUS-Σ
    Identity 3 verifies subset-sums against.
    """

    image_path: Path
    raw_gt: dict[str, Any] = field(default_factory=dict)

    @property
    def store_name(self) -> str:
        return _flatten_first(self.raw_gt, "store.name") or ""

    @property
    def date(self) -> str:
        for key in ("date.date", "date"):
            v = _flatten_first(self.raw_gt, key)
            if v:
                return v
        return ""

    @property
    def total(self) -> str:
        return _flatten_first(self.raw_gt, "total.total_price") or ""

    @property
    def subtotal(self) -> str:
        return _flatten_first(self.raw_gt, "total.subtotal_price") or ""

    @property
    def tax(self) -> str:
        return _flatten_first(self.raw_gt, "total.tax_price") or ""

    @property
    def service(self) -> str:
        return _flatten_first(self.raw_gt, "total.service_price") or ""

    @property
    def cash(self) -> str:
        return _flatten_first(self.raw_gt, "total.cashprice") or ""

    @property
    def change(self) -> str:
        return _flatten_first(self.raw_gt, "total.changeprice") or ""

    def item_prices(self) -> list[float]:
        """Per-item line-total prices from CORD's ``menu`` array.

        Each menu entry has ``price`` (= ``unitprice`` × ``cnt``).
        Used by FOCUS-Σ Identity 3 to verify
        ``Σ(items) + tax + service − discount ≈ total``.  Returns
        parsed floats; entries with unparseable prices are skipped.
        """
        out: list[float] = []
        menu = self.raw_gt.get("menu") or []
        if isinstance(menu, dict):
            menu = [menu]
        for entry in menu:
            if not isinstance(entry, dict):
                continue
            raw_price = entry.get("price") or entry.get("nm_unitprice")
            if not raw_price:
                continue
            try:
                clean = (
                    str(raw_price)
                    .replace(",", "")
                    .replace("원", "")
                    .replace("KRW", "")
                    .replace("$", "")
                    .replace(" ", "")
                    .strip()
                )
                if clean:
                    out.append(float(clean))
            except (TypeError, ValueError):
                continue
        return out


def _flatten_first(obj: Any, dotted: str) -> str | None:
    """Walk a dotted path over a CORD ``ground_truth`` dict.

    Handles both the wrapped form (under ``gt_parse``) and the
    unwrapped form (top-level keys).  Lists fall through with
    first-element semantics so ``menu.0.nm`` works without an
    explicit index.
    """
    if not isinstance(obj, dict):
        return None
    if "gt_parse" in obj and isinstance(obj["gt_parse"], dict):
        obj = obj["gt_parse"]
    parts = dotted.split(".")
    cur: Any = obj
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list):
            if not cur:
                return None
            cur = cur[0].get(p) if isinstance(cur[0], dict) else None
        else:
            return None
        if cur is None:
            return None
    if isinstance(cur, str):
        return cur
    if isinstance(cur, int | float):
        return str(cur)
    return None


def load_cord_split(
    split: Literal["train", "validation", "test"] = "test",
    repo_id: str = "naver-clova-ix/cord-v2",
    cache_dir: str | None = None,
) -> list[CordReceipt]:
    """Load one CORD split as ``list[CordReceipt]``.

    Lazy-imports ``datasets`` and ``PIL`` so the dependencies stay
    optional — callers that don't touch CORD see no overhead.
    Falls back to an empty list (logged warning) when the dataset
    cannot be reached.

    Image handling.  CORD ships images as PIL objects inside the HF
    dataset; this function materialises them to
    ``cache_dir/<split>/NNNN.png`` so downstream tools can open them
    by path (matching the SROIE shape).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        log.info(
            "data.cord.load_cord_split: missing dep (%s); "
            "cross-dataset eval will skip CORD.", exc,
        )
        return []
    cache_root = Path(cache_dir or Path.home() / ".cache" / "kaggle2" / "cord")
    try:
        ds = load_dataset(repo_id, split=split, cache_dir=str(cache_root))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "data.cord.load_cord_split: cannot load %s/%s (%s).",
            repo_id, split, exc,
        )
        return []
    out_dir = cache_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[CordReceipt] = []
    for i, row in enumerate(ds):
        img_path = out_dir / f"{i:04d}.png"
        if not img_path.exists():
            try:
                row["image"].save(img_path, format="PNG")
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "data.cord.load_cord_split: cannot save image %d (%s); skip.",
                    i, exc,
                )
                continue
        gt_raw = row.get("ground_truth")
        if isinstance(gt_raw, str):
            try:
                gt = json.loads(gt_raw)
            except json.JSONDecodeError:
                gt = {}
        elif isinstance(gt_raw, dict):
            gt = gt_raw
        else:
            gt = {}
        receipts.append(CordReceipt(image_path=img_path, raw_gt=gt))
    log.info(
        "data.cord.load_cord_split: loaded %d CORD-%s receipts (cache=%s).",
        len(receipts), split, out_dir,
    )
    return receipts


def cord_to_sroie_receipts(
    cord_rows: list[CordReceipt],
    schema: tuple[str, ...] = CROSS_DATASET_FIELDS,
) -> list[Receipt]:
    """Adapt CordReceipt to the kaggle2 :class:`Receipt` shape.

    Output uses SROIE field names so the FOCUS pipeline, LayoutLMv3
    evaluator, and DONUT decoder share one Receipt schema across both
    datasets.  Fields not in ``schema`` are omitted; fields CORD does
    not populate (e.g. ``address``) appear as empty strings — the
    symmetric normaliser collapses these to "" on both pred and GT
    side, contributing zero to F1 in either direction.
    """
    receipts: list[Receipt] = []
    for cr in cord_rows:
        fields_dict = {
            "company": cr.store_name,
            "date": cr.date,
            "total": cr.total,
            "address": "",
            "subtotal": cr.subtotal,
            "tax": cr.tax,
            "service": cr.service,
            "cash": cr.cash,
            "change": cr.change,
        }
        receipts.append(
            Receipt(
                image_path=cr.image_path,
                fields=[
                    Field(name=f, value=fields_dict.get(f, ""))
                    for f in schema
                ],
            ),
        )
    return receipts


def load_cord_full_schema(
    split: Literal["train", "validation", "test"] = "test",
    cache_dir: str | None = None,
) -> list[Receipt]:
    """Full-CORD-schema as :class:`Receipt`s.

    Includes the SROIE-comparable triple plus CORD-native
    ``subtotal / tax / service / cash / change``.  Used by FOCUS-Σ's
    full 3-identity witness count on receipts where every keyword is
    present (CORD's structured GT guarantees this for a much larger
    fraction of receipts than SROIE).
    """
    return cord_to_sroie_receipts(
        load_cord_split(split, cache_dir=cache_dir),
        schema=CORD_FULL_FIELDS,
    )


# ---------------------------------------------------------------------------
# Backward-compat shim — old call site uses ``load_cord(split, config)``.
# ---------------------------------------------------------------------------

def load_cord(
    split: Literal["train", "validation", "test"], config: ExpConfig,
) -> list[Receipt]:
    """Backward-compat: project CORD onto the 4-field SROIE schema.

    Use :func:`load_cord_split` + :func:`cord_to_sroie_receipts` for
    new code that wants access to the full CORD schema.
    """
    cache_dir = str(Path(config.data_dir) / "cord_cache")
    return cord_to_sroie_receipts(
        load_cord_split(split, cache_dir=cache_dir),
        schema=("company", "address", "date", "total"),
    )
