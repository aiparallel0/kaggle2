"""PR-D — CORD receipt dataset loader (4-field SROIE-shaped subset).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: load CORD (Consolidated Receipt Dataset, Park et al. 2019) and
    project its 30+ fields onto the SROIE-shaped 4-field schema
    {company, date, address, total} so the cross-dataset arm of the
    experiment can compare DONUT / pipeline / LayoutLMv3 / GPT-4V on
    a non-SROIE distribution.

Field map (CORD → SROIE):
  ``menu.subtotal_price`` / ``total.total_price`` → ``total``
  ``menu.dt`` / ``total.tx_date``                  → ``date``
  ``store.store_addr``                              → ``address``
  ``store.store_name``                              → ``company``

Empty when ``hf_datasets`` is not installed; the eval stage skips
the CORD arm gracefully in that case.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from core.types import ExpConfig, Field, Receipt

log = logging.getLogger("kaggle2")


def load_cord(
    split: Literal["train", "validation", "test"], config: ExpConfig,
) -> list[Receipt]:
    """Return CORD as ``list[Receipt]`` projected onto SROIE 4-field schema.

    Returns an empty list (with a warning) when ``datasets`` /
    ``huggingface_hub`` are not installed or the public CORD dataset
    is not reachable.  Cached HuggingFace data is preferred when
    present at ``${HF_HOME}/datasets/clovaai_cord-v2``.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.info(
            "data.cord: hf_datasets not installed; skipping CORD load.",
        )
        return []
    try:
        ds = load_dataset(
            "naver-clova-ix/cord-v2", split=split,
            cache_dir=str(Path(config.data_dir) / "cord_cache"),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("data.cord: load_dataset failed (%s); skipping.", exc)
        return []
    return [_to_receipt(row) for row in ds]


def _to_receipt(row: object) -> Receipt:
    """Project one CORD row onto the SROIE 4-field schema."""
    image_path = Path(str(_g(row, "image_path") or _g(row, "image") or ""))
    return Receipt(
        image_path=image_path,
        fields=[
            Field("company", str(_g(row, "store.store_name") or "")),
            Field("address", str(_g(row, "store.store_addr") or "")),
            Field("date", str(_g(row, "total.tx_date") or
                              _g(row, "menu.dt") or "")),
            Field("total", str(_g(row, "total.total_price") or
                               _g(row, "menu.total_price") or "")),
        ],
    )


def _g(obj: object, key: str) -> object | None:
    """Walk a dotted path through dicts/objects; return None on miss."""
    cur: object = obj
    for part in key.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return None
    return cur
