"""Compute training cost, energy, and CO₂ from GPU telemetry logs.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: integrates ``gpu_power_w`` samples via trapezoidal rule to produce
    the run_hours/energy_kwh/cost_usd/co2_kg columns in Table II of the
    paper's Results section.  Carbon intensity defaults to 0.4 kg CO₂/kWh
    (Strubell et al. 2019).  Pure functions; 2-in/1-out contract.
"""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_INTENSITY_KG_KWH: float = 0.4  # Strubell et al. 2019 world average


def _read_log(log_path: str) -> list[dict[str, object]]:
    """Parse JSONL telemetry into row dicts (empty list if absent/empty)."""
    path = Path(log_path)
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                pass
    return rows


def _integrate_power(rows: list[dict[str, object]]) -> tuple[float, float]:
    """Trapezoidal integration of gpu_power_w → (run_hours, energy_kwh)."""
    times: list[float] = []
    powers: list[float] = []
    for row in rows:
        ts = row.get("ts")
        pw = row.get("gpu_power_w")
        if isinstance(ts, int | float) and isinstance(pw, int | float):
            times.append(float(ts))
            powers.append(float(pw))
    if len(times) < 2:
        return 0.0, 0.0
    run_h = (times[-1] - times[0]) / 3600.0
    energy_wh = 0.0
    for i in range(1, len(times)):
        dt_h = (times[i] - times[i - 1]) / 3600.0
        avg_w = (powers[i] + powers[i - 1]) / 2.0
        energy_wh += avg_w * dt_h
    return run_h, energy_wh / 1000.0  # Wh → kWh


def summarise(
    log_path: str,
    rate_usd_per_hr: float,
    intensity_kg_per_kwh: float = _DEFAULT_INTENSITY_KG_KWH,
) -> dict[str, float]:
    """Reduce telemetry to run_hours/energy_kwh/cost_usd/co2_kg for Table II."""
    rows = _read_log(log_path)
    run_h, energy_kwh = _integrate_power(rows)
    return {
        "run_hours": round(run_h, 4),
        "energy_kwh": round(energy_kwh, 6),
        "cost_usd": round(run_h * rate_usd_per_hr, 4),
        "co2_kg": round(energy_kwh * intensity_kg_per_kwh, 6),
    }
