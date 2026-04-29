"""Tests for the ``render_*_table`` LaTeX tabular emitters."""
from __future__ import annotations

from report.inject_tables import (
    inject_tables,
    render_env_table,
    render_extended_table,
    render_f1_table,
    render_latency_table,
    render_training_table,
)


def test_f1_table_produces_tabular() -> None:
    out = render_f1_table({"donut_f1": 0.78, "pipeline_f1": 0.80})
    assert out.startswith("\\begin{tabular}")
    assert out.rstrip().endswith("\\end{tabular}")
    # Every field is mentioned.
    for fld in ("company", "date", "address", "total", "macro"):
        assert fld in out


def test_f1_table_missing_value_renders_missing_cells() -> None:
    out = render_f1_table({})
    # v4 contract: every absent value renders as a typed marker, not ---.
    # 5 rows × 2 always-required columns (DONUT, Pipeline) = 10 hard
    # \MissingCell{} blockers.  The Rule-based column is intentionally
    # optional in canonical-SROIE mode (rulebased_* on the missing-OK
    # allow-list, see report/missing.py) and renders \textit{n/a} when
    # absent — counted separately so the assertion remains precise.
    assert out.count("\\MissingCell{") >= 10
    assert out.count("\\textit{n/a}") >= 5
    assert "---" not in out


def test_f1_table_resolves_populated_values() -> None:
    metrics = {f"donut_f1_{f}": 0.9 for f in ("company", "date", "address", "total")}
    metrics["donut_f1"] = 0.85
    out = render_f1_table(metrics)
    assert "90.0\\%" in out
    assert "85.0\\%" in out


def test_extended_table_has_rows_for_both_systems() -> None:
    out = render_extended_table({})
    # 2 systems × 4 fields = 8 data rows, plus one header row → 9 "\\".
    assert out.count("\\\\") == 9


def test_latency_table_columns() -> None:
    out = render_latency_table({
        "donut_latency_mean": 142.3, "donut_latency_p95": 200.0,
        "donut_throughput_batch1": 6.5, "donut_usd_per_img": 0.0012,
    })
    assert "142\\,ms" in out
    assert "200\\,ms" in out
    assert "\\$0.0012" in out


def test_env_table_structure() -> None:
    out = render_env_table({
        "git_sha": "abc1234", "seed": 42, "gpu_model": "RTX 4090",
    })
    assert "\\texttt{abc1234}" in out
    assert "42" in out
    # Missing keys render as \MissingCell{} markers (v4 contract);
    # allow-listed (n/a) keys render as \textit{n/a}.  Either way no
    # silent em-dashes.
    assert "---" not in out
    assert "\\MissingCell{" in out or "\\textit{n/a}" in out


def test_env_table_reads_prefixed_keys() -> None:
    """v4 fix: ``merge_env`` writes ``env_*`` / ``host_*`` prefixed keys
    (not bare ones); the env table must consume both naming conventions
    so Table~XIV no longer renders all em-dashes despite the producer
    running.
    """
    out = render_env_table({
        "env_git_sha": "deadbee", "host_gpu_model": "RTX A6000",
        "host_torch_version": "2.4.0", "host_cuda_version": "12.4",
    })
    assert "\\texttt{deadbee}" in out
    assert "RTX A6000" in out
    assert "2.4.0" in out
    assert "12.4" in out


def test_training_table_full_values() -> None:
    out = render_training_table({
        "donut_epochs": 30, "donut_best_epoch": 18,
        "donut_wall_clock_s": 3600.0, "donut_peak_vram_gb": 11.2,
        "donut_cost_usd": 0.42, "donut_energy_kwh": 0.18,
    })
    assert "30" in out
    assert "18" in out
    assert "11.2\\,GiB" in out
    assert "\\$0.42" in out
    # Item 58 (paper-corrections v13): Table X energy column now reads
    # from ``{system}_energy_kwh`` (the key actually emitted by the
    # cost producer; ``_energy_wh`` was unset, so the cell rendered
    # ``\textit{n/a}`` despite the value being known).  ``sig4`` keeps
    # 4 significant figures of 0.18 → ``0.1800``.
    assert "0.18" in out


def test_inject_tables_returns_all_five_blocks() -> None:
    blocks = inject_tables({})
    # Seven emitters now (Audit A2/B2): the original five tabular blocks
    # plus ``table_competitors`` (non-empty only under canonical_347)
    # plus ``table_bug_ablation`` (data-driven from
    # ``results/bug_timeline.json``); with empty metrics the
    # competitors block is ``""`` and excluded from the tabular-shape
    # check.
    assert set(blocks) == {
        "table_headline_f1", "table_extended",
        "table_latency", "table_env", "table_training",
        "table_competitors", "table_bug_ablation",
    }
    for key, block in blocks.items():
        if key == "table_competitors":
            assert block == "", "competitors block must be empty without canonical_347"
            continue
        assert block.startswith("\\begin{tabular}")
        assert block.rstrip().endswith("\\end{tabular}")


def test_inject_tables_competitors_emits_under_canonical() -> None:
    blocks = inject_tables({
        "test_set_kind": "canonical_347",
        "donut_f1": 0.83, "pipeline_f1": 0.79,
    })
    comp = blocks["table_competitors"]
    assert comp.startswith("\\begin{tabular}")
    assert "\\textbf{this work}" in comp
    assert "0.830" in comp and "0.790" in comp
    assert "\\cite{kim2022donut}" in comp
