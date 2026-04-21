"""test_inject_new_keys.py — new \\VAR{} keys resolve or fall back to ---."""
from __future__ import annotations

import json
import os
from pathlib import Path

_NEW_KEYS = [
    "donut_f1_std",
    "pipeline_f1_std",
    "donut_f1_ci_lo",
    "donut_f1_ci_hi",
    "pipeline_f1_ci_lo",
    "pipeline_f1_ci_hi",
    "mcnemar_p",
    "donut_params_m",
    "pipeline_params_m",
    "assigner_params_k",
    "donut_peak_vram_gb",
    "pipeline_peak_vram_gb",
    "donut_train_minutes",
    "pipeline_train_minutes",
    "donut_samples_per_sec",
    "inference_latency_p50_ms",
    "inference_latency_p95_ms",
    "inference_latency_p99_ms",
    "donut_cost_usd",
    "pipeline_cost_usd",
    "donut_energy_kwh",
    "pipeline_energy_kwh",
    "donut_co2_kg",
    "pipeline_co2_kg",
    "gpu_model",
    "cuda_version",
    "vastai_host_id",
    "assigner_delta",
    # Paper-sprinkle keys: assigner telemetry, differential LR,
    # pipeline diagnostics, KD hooks. Surfaced from
    # assigner_metrics.json / pipeline_metrics.json / config.json.
    "assigner_best_epoch",
    "assigner_stopped_at",
    "assigner_best_val_loss",
    "lr_encoder",
    "lr_decoder",
    "empty_detection_fraction",
    "per_receipt_error_fraction",
    "parity_ok",
    "kd_attn_weight",
    "kd_logits_weight",
    "epochs_assigner",
]


def _build_template(keys: list[str]) -> str:
    return "\n".join(f"\\VAR{{{k}}}" for k in keys)


def test_all_new_keys_resolve_with_metrics() -> None:
    """Keys present in the metrics dict are substituted correctly."""
    from report.inject import inject_results

    metrics: dict[str, object] = {k: 0.5 for k in _NEW_KEYS}
    metrics["gpu_model"] = "RTX 4090"
    metrics["cuda_version"] = "12.4"
    metrics["vastai_host_id"] = "host-123"
    template = _build_template(_NEW_KEYS)
    result = inject_results(template, metrics)
    assert "\\VAR{" not in result, "Unreplaced \\VAR{} tokens remain"
    assert "---" not in result, "No key should fall back to --- when metrics are complete"


def test_absent_keys_fallback_to_dashes() -> None:
    """Keys absent from the metrics dict fall back to '---', not a LaTeX error."""
    from report.inject import inject_results

    template = _build_template(_NEW_KEYS)
    result = inject_results(template, {})
    assert "\\VAR{" not in result, "Unreplaced \\VAR{} tokens remain"
    assert result.count("---") == len(_NEW_KEYS), (
        f"Each absent key should become '---'; got {result.count('---')} dashes"
    )


def test_inject_compiles_with_fixture_metrics(tmp_path: Path) -> None:
    """End-to-end: template.tex inlines sections and injects fixture metrics."""
    import re
    fixture_metrics: dict[str, object] = {
        "donut_f1": 0.8500,
        "pipeline_f1": 0.8400,
        "assigner_delta": 0.0300,
        "seeds_used": [13, 17, 42],
        "donut_f1_mean": 0.850,
        "donut_f1_std": 0.005,
        "pipeline_f1_mean": 0.840,
        "pipeline_f1_std": 0.007,
        "epochs_donut": 15,
        "epochs_trocr": 10,
        "epochs_yolo": 30,
        "batch_size": 8,
        "lr": 1e-4,
        "precision": "bf16",
        "label_smoothing": 0.1,
        "yolo_img_size": 512,
        "img_w": 1280,
        "img_h": 960,
        "artifact_mode": "full",
    }
    metrics_path = tmp_path / "combined_metrics.json"
    metrics_path.write_text(json.dumps(fixture_metrics))

    template_path = Path(os.path.dirname(__file__)).parent / "report" / "template.tex"
    from report.inject import expand_inputs, inject_results

    template = template_path.read_text()
    expanded = expand_inputs(template, template_path.parent)
    filled = inject_results(expanded, fixture_metrics)

    # The inject backstop replaces \VAR{non-empty-key} patterns with ---.
    # \VAR{} (empty) in LaTeX comments is legitimate documentation and is
    # intentionally preserved; the regex [^}]+ requires >= 1 chars.
    remaining = re.findall(r"\\VAR\{[^}]+\}", filled)
    assert not remaining, f"Unreplaced \\VAR{{}} tokens remain: {remaining}"
    assert "\\documentclass" in filled
