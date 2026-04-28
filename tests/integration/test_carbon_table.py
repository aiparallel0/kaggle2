"""PR-D — Carbon-emissions table contract."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import write_min_config


def _write_env(out_dir: Path, tdp_w: float, wall_s: float) -> None:
    (out_dir / "env").mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu_tdp_w": tdp_w,
        "wallclock_seconds": wall_s,
    }
    (out_dir / "env" / "env_snapshot.json").write_text(json.dumps(payload))


def test_carbon_default_grid_factor(tmp_path: Path) -> None:
    from core.config import load_config
    from report.combine_ext import merge_carbon

    cfg = load_config(str(write_min_config(tmp_path)))
    _write_env(tmp_path, tdp_w=300.0, wall_s=3600.0)  # 1 hour at 300 W
    metrics: dict[str, object] = {}
    merge_carbon(cfg, metrics)
    # 300 W * 1 h = 0.3 kWh; 0.3 kWh * 0.475 = 0.1425 kg CO2e
    assert "carbon_kgco2e" in metrics
    assert "carbon_kwh" in metrics
    assert abs(float(metrics["carbon_kwh"]) - 0.3) < 1e-3  # type: ignore[arg-type]
    assert abs(float(metrics["carbon_kgco2e"]) - 0.1425) < 1e-3  # type: ignore[arg-type]


def test_carbon_extra_override(tmp_path: Path) -> None:
    from core.config import load_config
    from report.combine_ext import merge_carbon

    cfg = load_config(str(write_min_config(
        tmp_path, grid_factor=0.100,
    )))
    _write_env(tmp_path, tdp_w=300.0, wall_s=3600.0)
    metrics: dict[str, object] = {}
    merge_carbon(cfg, metrics)
    assert abs(float(metrics["carbon_grid_factor"]) - 0.100) < 1e-9  # type: ignore[arg-type]


def test_carbon_missing_env_silent(tmp_path: Path) -> None:
    from core.config import load_config
    from report.combine_ext import merge_carbon

    cfg = load_config(str(write_min_config(tmp_path)))
    metrics: dict[str, object] = {}
    merge_carbon(cfg, metrics)
    # No snapshot → no keys emitted.
    assert "carbon_kgco2e" not in metrics
