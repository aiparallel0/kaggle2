"""test_telemetry_nogpu.py — sampler emits a no-gpu line when nvidia-smi absent."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch


def test_start_stop_no_gpu(tmp_path: Path) -> None:
    out = str(tmp_path / "telem.jsonl")
    with patch("shutil.which", return_value=None):
        from core.telemetry import start_sampler, stop_sampler
        thread, event = start_sampler(out, interval_s=0.1)
        time.sleep(0.3)
        stop_sampler(thread, event)
    lines = Path(out).read_text().splitlines()
    assert lines, "Expected at least one JSONL line"
    row = json.loads(lines[0])
    assert row.get("note") == "no-gpu", f"Expected no-gpu note, got {row}"


def test_stop_returns_path(tmp_path: Path) -> None:
    out = str(tmp_path / "telem2.jsonl")
    with patch("shutil.which", return_value=None):
        from core.telemetry import start_sampler, stop_sampler
        thread, event = start_sampler(out, interval_s=0.1)
        returned = stop_sampler(thread, event)
    assert returned == out
