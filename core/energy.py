"""Energy / carbon accounting — Wh + kgCO2 per training run.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: convert a power-draw time series (sampled by
    :mod:`core.tracking` via ``pynvml``) into Wh, then into kgCO2
    using a per-region grid-intensity table.  The default is the IEA
    world-mix value of 0.233 kgCO2/kWh; pass ``region=`` to override.
    Used by ``figures_cost.py`` to produce the USD-vs-F1-vs-Wh Pareto
    and by ``inject_tables.render_table_latency`` to populate the
    ``\\VAR{energy_wh}`` / ``\\VAR{energy_kgco2}`` placeholders.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

log = logging.getLogger("kaggle2")

# Default grid intensity values (kgCO2 per kWh) for the regions
# kaggle2 is most often run in.  World-mix from the IEA is the default
# fall-back when the caller's region is unknown.
DEFAULT_INTENSITY_KG_PER_KWH: dict[str, float] = {
    "world": 0.233,
    "US": 0.386,
    "US-CA": 0.239,
    "US-TX": 0.408,
    "EU": 0.231,
    "DE": 0.366,
    "FR": 0.056,
    "UK": 0.193,
    "CN": 0.555,
    "IN": 0.713,
    "JP": 0.436,
    "CA": 0.128,
    "AU": 0.656,
}

# Default vast.ai hourly USD rates by GPU model.  Overridable via
# :func:`hourly_usd_for` when the config carries a
# ``vastai_hourly_usd`` dict.
DEFAULT_HOURLY_USD: dict[str, float] = {
    "RTX 4090": 0.40,
    "RTX 3090": 0.25,
    "RTX A6000": 0.60,
    "A100_40G": 1.20,
    "A100_80G": 1.80,
    "H100": 3.00,
}


def integrate_power_wh(
    power_samples_watts: Iterable[float],
    interval_seconds: float,
) -> float:
    """Riemann-sum a power-draw sequence into watt-hours.

    ``power_samples_watts`` is e.g. the ``gpu_power_w`` column from
    ``curves/gpu_util.csv``; ``interval_seconds`` is the sampling
    period.  Returns total energy in Wh.  Non-positive inputs return 0
    so a crashed sampler doesn't poison the reported number.
    """
    if interval_seconds <= 0:
        return 0.0
    total_w_seconds = 0.0
    for w in power_samples_watts:
        try:
            total_w_seconds += float(w) * interval_seconds
        except (TypeError, ValueError):
            continue
    return total_w_seconds / 3600.0


def wh_to_kgco2(
    energy_wh: float,
    region: str = "world",
    intensity_override: dict[str, float] | None = None,
) -> float:
    """Convert ``energy_wh`` (watt-hours) to kgCO2 for ``region``."""
    table = intensity_override or DEFAULT_INTENSITY_KG_PER_KWH
    intensity = table.get(region, DEFAULT_INTENSITY_KG_PER_KWH["world"])
    return (energy_wh / 1000.0) * intensity


def hourly_usd_for(
    gpu_model: str,
    override: dict[str, float] | None = None,
) -> float:
    """Return the USD/hour rate for ``gpu_model``.

    Checks the override dict first (so a config can carry per-run
    spot prices), then the shipped defaults, then falls back to a
    conservative 0.50 USD/hour when the GPU is unrecognised.  Never
    raises — this is reporting, not billing.
    """
    if override is not None:
        for key, value in override.items():
            if key.lower() in gpu_model.lower():
                return float(value)
    for key, value in DEFAULT_HOURLY_USD.items():
        if key.lower() in gpu_model.lower():
            return float(value)
    log.info("energy: unknown GPU model %r — falling back to $0.50/h", gpu_model)
    return 0.50


def wall_seconds_to_usd(
    wall_seconds: float,
    gpu_model: str,
    override: dict[str, float] | None = None,
) -> float:
    """Convert wall-clock seconds into USD using :func:`hourly_usd_for`."""
    if wall_seconds <= 0:
        return 0.0
    return (wall_seconds / 3600.0) * hourly_usd_for(gpu_model, override)
