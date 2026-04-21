"""Cost and energy accounting from telemetry JSONL logs.

Pure functions — no side effects, no environment variable magic.
Energy is integrated from ``gpu_power_w`` samples using the trapezoidal rule.
CO₂ uses a configurable carbon-intensity factor (default 0.4 kg CO₂ / kWh,
the global average from Strubell et al. 2019~\\cite{strubell2019energy}).

Public API (2-in / 1-out per function):
  summarise(log_path, rate_usd_per_hr) -> dict[str, float]
"""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_INTENSITY_KG_KWH: float = 0.4  # Strubell et al. 2019 world average


def _read_log(log_path: str) -> list[dict[str, object]]:
    """Parse a JSONL telemetry file into a list of row dicts.

    Args:
        log_path: Path to the JSONL file written by core.telemetry.
        _sentinel: Unused placeholder to keep a 2-in shape at call sites.

    Returns:
        List of parsed JSON objects (empty list if file is absent or empty).
    """
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
    """Trapezoidal integration of gpu_power_w samples.

    Args:
        rows: Telemetry row dicts with 'ts' (Unix epoch) and 'gpu_power_w'.
        _sentinel: Unused; satisfies the 2-arg shape for internal helpers.

    Returns:
        ``(run_hours, energy_kwh)`` — (0, 0) when fewer than 2 GPU samples.
    """
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
    """Compute cost, energy and CO₂ from a telemetry JSONL log.

    Args:
        log_path: Path to the JSONL telemetry file.
        rate_usd_per_hr: GPU instance cost in USD / hour.

    Returns:
        Dict with keys ``run_hours``, ``energy_kwh``, ``cost_usd``,
        ``co2_kg``.
    """
    rows = _read_log(log_path)
    run_h, energy_kwh = _integrate_power(rows)
    return {
        "run_hours": round(run_h, 4),
        "energy_kwh": round(energy_kwh, 6),
        "cost_usd": round(run_h * rate_usd_per_hr, 4),
        "co2_kg": round(energy_kwh * intensity_kg_per_kwh, 6),
    }
