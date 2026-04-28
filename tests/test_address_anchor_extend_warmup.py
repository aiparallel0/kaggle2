"""PR-C / S1 — Bug-14 guard: anchor-extender warmup ordering."""
from __future__ import annotations

from pathlib import Path

from conftest import write_min_config


def test_address_anchor_extend_default_off(tmp_path: Path) -> None:
    from core.config import load_config

    cfg = load_config(str(write_min_config(tmp_path)))
    assert cfg.address_anchor_extend is False
    assert cfg.address_anchor_extender_k == 2


def test_address_anchor_extend_can_enable(tmp_path: Path) -> None:
    from core.config import load_config

    cfg = load_config(str(write_min_config(
        tmp_path,
        address_anchor_extend=True,
        address_anchor_extender_k=3,
    )))
    assert cfg.address_anchor_extend is True
    assert cfg.address_anchor_extender_k == 3


def test_bug_14_flag_present(tmp_path: Path) -> None:
    """Bug-14 must be in the default ``bug_flags`` dict (PR-C atlas)."""
    from core.config import load_config

    cfg = load_config(str(write_min_config(tmp_path)))
    assert "bug_14" in cfg.bug_flags
    assert "bug_15" in cfg.bug_flags
    assert "bug_16" in cfg.bug_flags
    assert "bug_17" in cfg.bug_flags
    assert cfg.bug_flags["bug_14"] is True
