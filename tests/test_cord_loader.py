"""PR-D — CORD loader smoke test."""
from __future__ import annotations

from pathlib import Path

from conftest import write_min_config


def test_cord_loader_importable() -> None:
    from data.cord import load_cord

    assert callable(load_cord)


def test_cord_loader_empty_without_datasets(tmp_path: Path) -> None:
    """Without HF ``datasets`` the loader returns ``[]`` (warns)."""
    import sys

    from core.config import load_config
    from data.cord import load_cord

    cfg = load_config(str(write_min_config(tmp_path)))
    if "datasets" in sys.modules:
        out = load_cord("test", cfg)
        assert isinstance(out, list)
    else:
        out = load_cord("test", cfg)
        assert out == []


def test_cord_field_projection_via_to_receipt() -> None:
    """The private ``_to_receipt`` projects nested CORD JSON correctly."""
    from data.cord import _to_receipt

    row = {
        "image_path": "/tmp/x.png",
        "store": {
            "store_name": "TEST RESTAURANT",
            "store_addr": "1 ALPHA ST",
        },
        "total": {"total_price": "12.50", "tx_date": "2019-08-01"},
        "menu": {"total_price": "12.50"},
    }
    rec = _to_receipt(row)
    fields = {f.name: f.value for f in rec.fields}
    assert fields["company"] == "TEST RESTAURANT"
    assert fields["address"] == "1 ALPHA ST"
    assert fields["date"] == "2019-08-01"
    assert fields["total"] == "12.50"
