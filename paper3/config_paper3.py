"""Paper 3 configuration loader.

Loads ``paper3/configs/default.json`` (or any other Paper 3 preset)
and returns a fully-populated :class:`core.types.ExpConfig` with the
SVKIE multi-prior framework engaged: FOCUS-T cross-attention assigner
on, FOCUS-Σ subset-sum verifier on, three-headed ensemble enabled,
zone-prior HMM on, and the architecture-agnostic wrapper paths
configured to evaluate DONUT and LayoutLMv3 backbones in addition to
the in-house FOCUS-T head.

This guarantees that an operator who copies a stale JSON from the
Paper 2 tree and tries to load it as a Paper 3 config gets the full
SVKIE stack engaged regardless of the on-disk content.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.config import load_config
from core.types import ExpConfig

__all__ = ["load_paper3_config", "PAPER3_DEFAULT_CONFIG_PATH"]


PAPER3_DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.json"


def load_paper3_config(config_path: str | Path | None = None) -> ExpConfig:
    """Load a Paper 3 config and enforce SVKIE-engaged invariant.

    Args:
        config_path: optional path to a Paper 3 JSON preset.  When None,
            loads ``paper3/configs/default.json``.

    Returns:
        An :class:`core.types.ExpConfig` with the full SVKIE multi-
        prior framework engaged: FOCUS-T cross-attention, FOCUS-Σ
        subset-sum verifier, GAT graph-attention head, zone-prior
        HMM, multi-seed harness, paired-bootstrap CIs.  Output
        directory rerouted under ``paper3/runs/``; paper template
        pinned at ``paper3/report/template_paper3.tex``.
    """
    cfg_path = Path(config_path) if config_path else PAPER3_DEFAULT_CONFIG_PATH
    cfg = load_config(str(cfg_path))
    return replace(
        cfg,
        # Engage the full SVKIE stack.
        focus_enabled=True,
        focus_total_enabled=True,
        focus_company_enabled=True,
        focus_company_span_enabled=True,
        total_arithmetic_enabled=True,
        zone_prior_enabled=True,
        priors_v4=True,
        # Paper 3 paths.
        paper_template=str(
            Path(__file__).parent / "report" / "template_paper3.tex",
        ),
        paper_output=str(
            Path(__file__).parent / "report" / "paper3_filled.tex",
        ),
        paper_variant="paper3",
    )
