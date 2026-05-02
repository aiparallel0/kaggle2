"""Paper 2 configuration loader.

Loads ``paper2/configs/default.json`` (or any other Paper 2 preset)
and returns a fully-populated :class:`core.types.ExpConfig` whose
FOCUS-related flags are forcibly disabled: Paper 2's pipeline is
deliberately non-learned (regex + zone-prior HMM + per-field
postprocess) so any FOCUS or arithmetic-witness flag set in the JSON
is overridden to False before the config is returned.

This guarantees that an operator who copies a stale JSON from the
Paper 3 tree and tries to load it as a Paper 2 config gets a clean
Paper 2 configuration regardless of the on-disk content — the
guarantee is enforced in code, not in documentation.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.config import load_config
from core.types import ExpConfig

__all__ = ["load_paper2_config", "PAPER2_DEFAULT_CONFIG_PATH"]


PAPER2_DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.json"


def load_paper2_config(config_path: str | Path | None = None) -> ExpConfig:
    """Load a Paper 2 config and enforce FOCUS-disabled invariant.

    Args:
        config_path: optional path to a Paper 2 JSON preset.  When None,
            loads ``paper2/configs/default.json``.

    Returns:
        An :class:`core.types.ExpConfig` with every FOCUS, learned-
        assigner, and arithmetic-verifier flag set to False; output
        directory rerouted under ``paper2/runs/``; and paper template
        pinned at ``paper2/report/template_paper2.tex``.  The
        rule-based and zone-prior knobs the pipeline DOES use are
        preserved from the JSON.
    """
    cfg_path = Path(config_path) if config_path else PAPER2_DEFAULT_CONFIG_PATH
    cfg = load_config(str(cfg_path))
    return replace(
        cfg,
        # Disable every learned-assignment / arithmetic-verifier knob.
        focus_enabled=False,
        focus_total_enabled=False,
        focus_company_enabled=False,
        focus_company_span_enabled=False,
        total_arithmetic_enabled=False,
        gat_enabled=False,
        priors_v3=False,
        priors_v4=False,
        rag_enabled=False,
        layoutlmv3_enabled=False,
        cord_eval_enabled=False,
        # Keep the rule-based + zone-prior pieces that Paper 2 uses.
        regex_router=True,
        zone_prior_enabled=True,
        # Paper 2 paths.
        paper_template=str(
            Path(__file__).parent / "report" / "template_paper2.tex",
        ),
        paper_output=str(
            Path(__file__).parent / "report" / "paper2_filled.tex",
        ),
        paper_variant="paper2",
    )
