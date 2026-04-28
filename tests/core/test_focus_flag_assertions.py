"""Hard-assertion tests for the FOCUS architecture-flag invariants (Bug 18).

Pins the three rules in :func:`core.config._validate_focus_flags`:

  * ``paper_variant=='focus'`` requires every ``focus_*`` sub-flag True.
  * ``focus_total_enabled`` requires ``priors_v4=True``.
  * ``focus_enabled`` requires ``n_priors >= 20`` (i.e. priors_v4).

These rules turn the AGENTS.md "no silent placeholders" invariant into
a load-time guarantee, so the previous failure mode (shipping a focus
paper while every focus_* flag is False) cannot reproduce.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_min_config

from core.config import load_config
from core.errors import ConfigError


def _focus_overrides(**flags: object) -> dict[str, object]:
    base: dict[str, object] = {
        "focus_enabled": True,
        "focus_total_enabled": True,
        "focus_company_enabled": True,
        "priors_v4": True,
        "priors_v3": True,
        "paper_variant": "focus",
    }
    base.update(flags)
    return base


def test_default_focus_config_loads_clean(tmp_path: Path) -> None:
    """All FOCUS flags True + priors_v4 True must load without error."""
    p = write_min_config(tmp_path, **_focus_overrides())
    cfg = load_config(str(p))
    assert cfg.focus_enabled is True
    assert cfg.focus_total_enabled is True
    assert cfg.focus_company_enabled is True
    assert cfg.priors_v4 is True


def test_focus_paper_with_flags_off_raises(tmp_path: Path) -> None:
    """``paper_variant='focus'`` with any focus_* flag False → ConfigError."""
    p = write_min_config(
        tmp_path, **_focus_overrides(focus_total_enabled=False),
    )
    with pytest.raises(ConfigError, match="paper_variant='focus'"):
        load_config(str(p))


def test_focus_total_without_priors_v4_raises(tmp_path: Path) -> None:
    """FOCUS-T requires the v4 prior column (arithmetic_witness_self)."""
    p = write_min_config(
        tmp_path,
        **_focus_overrides(priors_v4=False, paper_variant="non-focus"),
    )
    with pytest.raises(ConfigError, match="priors_v4=True"):
        load_config(str(p))


def test_focus_enabled_without_priors_v4_raises(tmp_path: Path) -> None:
    """FOCUS-A span head requires n_priors >= 20 (i.e. priors_v4)."""
    p = write_min_config(
        tmp_path,
        focus_enabled=True,
        focus_total_enabled=False,
        focus_company_enabled=False,
        priors_v4=False,
        priors_v3=True,
        paper_variant="non-focus",
    )
    with pytest.raises(ConfigError, match="n_priors=14"):
        load_config(str(p))


def test_disabling_bug_18_skips_validation(tmp_path: Path) -> None:
    """``bug_flags.bug_18 = False`` is the documented escape hatch."""
    p = write_min_config(
        tmp_path,
        focus_enabled=True,
        focus_total_enabled=True,
        focus_company_enabled=False,
        priors_v4=True,
        priors_v3=True,
        paper_variant="focus",
        bug_flags={
            "bug_1": True, "bug_2": True, "bug_3": True, "bug_4": True,
            "bug_5": True, "bug_6": True, "bug_7": True, "bug_8": True,
            "bug_9": True, "bug_10": True, "bug_11": True, "bug_12": True,
            "bug_13": True, "bug_18": False,
        },
    )
    cfg = load_config(str(p))
    assert cfg.bug_flags["bug_18"] is False
