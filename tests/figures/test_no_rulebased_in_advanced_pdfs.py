"""Item 5 (paper-corrections): advanced-variant figure PDFs must not
contain a rule-based / heuristic-baseline series.

On the advanced (canonical-347) and focus variants the rule-based
GT-OCR$+$regex baseline is not run, so emitting it as a labelled line
on Fig.~\\ref{fig:training} (``fig_training_curves.pdf``) or
Fig.~\\ref{fig:telemetry_overlay} (``fig_telemetry_overlay.pdf``) would
zero-fill a fictitious series and silently mislead the reader.

This test exercises the two emitters with a synthetic run directory
under the advanced variant, opens the rendered PDFs as raw bytes,
and asserts the substring ``rule`` (case-insensitive) is absent.
The check is intentionally narrow: it operates on the literal byte
stream of the PDF so any matplotlib legend label, axis title, or
text annotation containing the word would be caught regardless of
which producer added it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


def _write_training_log(run_dir: Path) -> None:
    """Minimal training_log.json with DONUT-only series — no rule-based."""
    log = {
        "epochs": [1, 2, 3, 4, 5],
        "train_loss": [2.1, 1.5, 1.1, 0.9, 0.8],
        "eval_loss": [2.0, 1.4, 1.0, 0.85, 0.75],
        "eval_f1": [0.10, 0.30, 0.55, 0.72, 0.83],
    }
    (run_dir / "training_log.json").write_text(json.dumps(log))


def _write_telemetry(run_dir: Path) -> None:
    """Two telemetry traces (donut, pipeline) — no rule-based stream."""
    for stem in ("telemetry_donut.jsonl", "telemetry_pipeline.jsonl"):
        rows = [
            {"ts": float(i), "gpu_util_pct": 50.0 + i,
             "gpu_mem_used_mb": 4096.0, "gpu_power_w": 200.0,
             "gpu_temp_c": 60.0}
            for i in range(8)
        ]
        (run_dir / stem).write_text("\n".join(json.dumps(r) for r in rows))


def _assert_no_rule_substring(pdf: Path) -> None:
    assert pdf.is_file(), f"emitter did not write {pdf}"
    blob = pdf.read_bytes().lower()
    # PDF text streams are typically uncompressed for short labels but
    # may be Flate-compressed for longer documents.  Either way, the
    # short literal "rule" would only appear in the byte stream when
    # an emitter actually drew the series — there is no other reason
    # for the four-letter sequence to land in the file.
    assert b"rule" not in blob, (
        f"{pdf.name} contains 'rule' substring — rule-based series leaked"
        " into the advanced-variant figure"
    )


def test_training_curves_pdf_has_no_rulebased(tmp_path: Path) -> None:
    """``fig_training_curves.pdf`` (Fig.~2) carries no rule-based line."""
    from report.figures import render_training_curves

    _write_training_log(tmp_path)
    out = render_training_curves(str(tmp_path), str(tmp_path))
    assert out is not None, "render_training_curves returned None"
    _assert_no_rule_substring(Path(out))


def test_telemetry_overlay_pdf_has_no_rulebased(tmp_path: Path) -> None:
    """``fig_telemetry_overlay.pdf`` (Fig.~9) carries no rule-based line."""
    from report.figures_bugs import render_telemetry_overlay

    _write_telemetry(tmp_path)
    out = render_telemetry_overlay(str(tmp_path), str(tmp_path))
    assert out is not None, "render_telemetry_overlay returned None"
    _assert_no_rule_substring(Path(out))
