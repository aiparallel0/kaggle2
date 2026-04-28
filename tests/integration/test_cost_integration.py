"""test_cost_integration.py — trapezoidal energy on a 3-point log."""
from __future__ import annotations

import json
from pathlib import Path


def _write_log(path: Path, rows: list[dict[str, float]]) -> str:
    out = str(path / "telem.jsonl")
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return out


def test_trapezoidal_energy(tmp_path: Path) -> None:
    # t=0: 200 W, t=3600: 400 W, t=7200: 300 W
    # Trapezoid(0→3600): (200+400)/2 * 1 h = 300 Wh
    # Trapezoid(3600→7200): (400+300)/2 * 1 h = 350 Wh
    # Total energy = 650 Wh = 0.650 kWh
    rows = [
        {"ts": 0.0, "gpu_power_w": 200.0},
        {"ts": 3600.0, "gpu_power_w": 400.0},
        {"ts": 7200.0, "gpu_power_w": 300.0},
    ]
    log_path = _write_log(tmp_path, rows)
    from core.cost import summarise

    result = summarise(log_path, rate_usd_per_hr=1.0)
    assert abs(result["energy_kwh"] - 0.65) < 1e-6, result
    assert abs(result["run_hours"] - 2.0) < 1e-6, result
    assert abs(result["cost_usd"] - 2.0) < 1e-6, result
    # CO2 = 0.65 kWh * 0.4 kg/kWh = 0.26 kg
    assert abs(result["co2_kg"] - 0.26) < 1e-6, result


def test_missing_log(tmp_path: Path) -> None:
    from core.cost import summarise

    result = summarise(str(tmp_path / "nonexistent.jsonl"), rate_usd_per_hr=2.0)
    assert result["energy_kwh"] == 0.0
    assert result["cost_usd"] == 0.0


def test_fixture_log() -> None:
    """Smoke-test the committed fixture has enough rows to integrate."""
    import os
    fixture = os.path.join(
        os.path.dirname(__file__), "..", "fixtures", "telemetry_donut.jsonl"
    )
    from core.cost import summarise

    result = summarise(fixture, rate_usd_per_hr=0.5)
    # Fixture spans 10 s — energy should be non-zero
    assert result["energy_kwh"] > 0.0
    assert result["run_hours"] > 0.0
