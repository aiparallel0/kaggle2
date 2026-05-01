# Tracking matrix — what gets measured and where it lives

This document is the authoritative "no placeholders" reference.  Every
`\VAR{}` key in `report/template.tex` and `report/sections/*.tex`
traces back to either:

* **a producer** — a function that, during a successful `make all`,
  writes a real measurement to a specific file under `runs/<run_id>/`;
  or
* **a scoped-out key** — a placeholder that is intentionally not wired
  in this PR, with the reason documented below.

After a successful `make all`, `runs/<run_id>/metrics/unresolved_vars.json`
lists exactly which keys did not resolve.  The design goal is
`{"unresolved": [], "count": 0}` after a full, successful run with all
producers available.

## Producer → file matrix

| producer (function) | sidecar(s) it writes | keys exposed to `\VAR{}` |
|---|---|---|
| `report.combine.build_combined` | `metrics/combined_metrics.json` | `donut_f1`, `pipeline_f1`, `gtocr_rulebased_f1`, `donut_f1_<field>`, `rulebased_f1_<field>`, `epochs_*`, `batch_size`, `lr`, `precision`, `label_smoothing`, `warmup_steps`, `yolo_img_size`, `img_w`, `img_h`, `artifact_mode`, `lr_encoder`, `lr_decoder`, `kd_*_weight` |
| `report.combine.merge_cost_json` | reads `cost_donut.json`, `cost_pipeline.json` | `donut_cost_usd`, `pipeline_cost_usd`, `donut_energy_kwh`, `pipeline_energy_kwh`, `donut_co2_kg`, `pipeline_co2_kg`, `donut_train_minutes`, `pipeline_train_minutes` |
| `report.combine.merge_assigner_metrics` | reads `assigner_metrics.json` | `assigner_params_k`, `assigner_best_epoch`, `assigner_stopped_at` |
| `report.combine.merge_pipeline_diagnostics` | reads `pipeline_metrics.json`, `pipeline_meta.json` | `donut_params_m`, `pipeline_params_m`, `donut_peak_vram_gb`, `pipeline_peak_vram_gb`, `empty_detection_fraction`, `per_receipt_error_fraction`, `parity_ok` |
| `stages.eval.stage_eval` | `combined_metrics.json` (McNemar block) | `mcnemar_p`, `seeds_used`, `pipeline_bootstrap_ci_lo`, `pipeline_bootstrap_ci_hi`, `pipeline_em_bootstrap_ci_lo`, `pipeline_em_bootstrap_ci_hi`, `delta_f1_ci_lo`, `delta_f1_ci_hi`, `delta_em_ci_lo`, `delta_em_ci_hi` |
| `stages.eval_producers.emit_all` | `predictions/*.jsonl`, `metrics/extended_metrics.json` | `donut_precision_<field>`, `donut_recall_<field>`, `donut_em_<field>`, `donut_em_<field>_ci_lo`, `donut_em_<field>_ci_hi`, same for `pipeline_*` |
| `core.env_snapshot.write_env_snapshot` | `env/hostinfo.json`, `env/git_sha.txt`, … | `git_sha`, `config_sha256`, `torch_version`, `cuda_version`, `gpu_model`, `driver_version`, `seed`, `run_id` |
| `report.inject_tables.inject_tables` | — (in-memory dict) | `table_headline_f1`, `table_extended`, `table_latency`, `table_env`, `table_training` |
| `core.manifest.write_manifest` | `MANIFEST.json` | (not a `\VAR{}` producer — the file itself) |
| `data.sroie_canonical.ensure_canonical_test_set` | `<data_dir>/test/{img,entities}/` (×347 each) | (data producer, not a `\VAR{}` key — but powers `test_set_kind=canonical_347` and the `\VAR{donut_f1}` / `\VAR{pipeline_f1}` measured under the canonical Task-3 split when `canonical_sroie_enabled=true`) |
| `stages.eval.stage_eval` (canonical-active branch) | `combined_metrics.json` | `test_set_kind`, `test_set_size` |
| `data.zone_prior_fit.fit_zone_prior` + `score_train_acc` | `results/zone_prior.json`, `metrics/zone_prior_diag.json` | `zone_prior_train_acc` |
| `core.error_metrics.count_zone_violations` (aggregated in `stages.eval_producers.emit_all`) | `metrics/extended_metrics.json` | `zone_violation_count_company`, `zone_violation_count_total` |
| `models.eval_pipeline.eval_pipeline` (zone-gated branch) | `metrics/extended_metrics.json` | `pipeline_f1_total_zone_gated` |
| `report.wrapper_delta.merge_wrapper_delta` | reads `metrics/wrapper_delta_metrics.json`, `metrics/ablation_focus_sigma.json`, `metrics/error_decomposition.json`, `metrics/faithfulness_metrics.json`, `metrics/calibration_metrics.json`, `metrics/latency_metrics.json` | wrapper-Δ matrix: `donut_total_bare`, `donut_total_focus_sigma`, `donut_total_focus_sigma_delta_ci` (and `..._cord`); `layoutlmv3_total_bare`, `layoutlmv3_total_focus_sigma`, `layoutlmv3_total_focus_sigma_delta_ci` (and `..._cord`); `pipeline_f1_total_baseline`, `pipeline_f1_total_focus_sigma`, `pipeline_f1_total_focus_sigma_delta_ci`, `pipeline_f1_total_focus_sigma_ci_half`; `cord_macro_focus_sigma`, `cord_layoutlmv3_delta`, `pipeline_total_bare_cord`, `pipeline_focus_sigma_delta_ci_cord`. Headline (post-Σ): `pipeline_f1_focus_sigma`, `pipeline_ned_focus_sigma`, `pipeline_em_focus_sigma`. LayoutLMv3 cells: `layoutlmv3_f1`, `layoutlmv3_ned`, `layoutlmv3_em`, `layoutlmv3_f1_<field>`. Ablation: `ablation_row{1..6}_total_f1`, `ablation_row{2..6}_delta_ci`, `ablation_row3_delta_point`. Error decomposition: `err_class_<class>_n`, `err_class_<class>_pct`, `err_class_<class>_recovered` for `class ∈ {tax, ocr1, other, ocr2, lead, ocr3, zero, cash, negative, residue}`. Faithfulness: `faithfulness_deletion_auc`, `faithfulness_insertion_auc`, `faithfulness_random_baseline`. Calibration: `calibration_{ece,mce,brier}_{pre,post}`. Latency: `lat_{donut,pipeline,verifier}_{p50,p95,p99,mean}`. Provenance: `cord_test_n`, `layoutlmv3_epochs`, `pipeline_total_failures`. |

## Scoped-out keys (intentionally unwired in this PR)

| key | why | path to "real value" |
|---|---|---|
| `yolo_map50`, `yolo_map50_95`, `yolo_ap_<class>` | SROIE has no field-level gold bounding boxes — the dataset is KIE, not detection | would require a second dataset with box annotations (e.g. CORD) |
| `trocr_cer`, `trocr_wer` | current pipeline does not thread per-crop (gold_text, pred_text) pairs through `eval_pipeline.py` | would require instrumenting `_detect_and_read` to return intermediate OCR outputs alongside final fields |
| `lat_donut_p50`, `lat_donut_p95`, `lat_donut_p99`, same for `pipeline` | per-inference timing requires a dedicated timing-burst stage (cold-start, batch-1, batch-8) separate from the F1-critical eval loop | add `stages.latency` + `--stage latency` flag |
| per-epoch `curves/*.csv` (loss, LR, grad-norm, GPU util) | requires threading the `core.tracking.Tracker` into `models/donut_train.py`, `models/yolo_train.py`, `models/trocr_train.py` | mechanical instrumentation pass in a follow-up PR |
| `gtocr_rulebased_*`, `rulebased_*`, `oracle_patch_*` (canonical run only) | the SROIE Task-3 archive ships KIE entities only — no GT box / OCR streams — so the GT-OCR-rulebased baseline and oracle-patch diagnostic cannot be measured against the canonical 347-image test set | switch to `--paper-variant baseline` (500/63/63 internal split) where SROIE training-set GT boxes are available; both runs ship side-by-side via the bifurcated `template_baseline.tex` / `template_focus.tex` |
| `donut_total_focus_sigma`, `layoutlmv3_total_focus_sigma`, `pipeline_f1_total_focus_sigma`, `cord_macro_focus_sigma`, all `*_delta_ci` and `*_cord` variants (NeurIPS template only) | requires the dedicated wrapper-Δ run that evaluates each upstream head (DONUT / LayoutLMv3 / FOCUS-T) in two configurations (Bare / +Σ) on each test set (SROIE Task-3 / CORD-v2). The producer (`report.wrapper_delta.merge_wrapper_delta`) is wired and consumes `metrics/wrapper_delta_metrics.json`; the run that emits this sidecar has not yet been executed on a GPU box. | execute `bash scripts/run_focus_ablation.sh` plus `bash scripts/run_5seed_sweep.sh` with `layoutlmv3_enabled=true` and `cord_eval_enabled=true`; the eval producer in `stages.eval` writes the sidecar at run end. |
| `ablation_row{1..6}_total_f1`, `ablation_row{2..6}_delta_ci`, `ablation_row3_delta_point` | requires the per-component ablation grid (`scripts/run_focus_ablation.sh`) which produces six `runs/<id>/` directories whose aggregator emits `metrics/ablation_focus_sigma.json` consumed by `merge_wrapper_delta`. | run the ablation script on a GPU box; `aggregate_seeds.py` writes the sidecar. |
| `err_class_*_n`, `err_class_*_pct`, `err_class_*_recovered` | requires a per-receipt error classifier that consumes `predictions/per_field_errors.jsonl` and emits `metrics/error_decomposition.json` with the 10 classes enumerated in `RESEARCH_DECONSTRUCTION.md` §4. | classifier exists in `core/error_metrics.py`; the dedicated emitter for the per-class counts is a one-screen wrapper that aggregates the existing per-field errors. |
| `faithfulness_deletion_auc`, `faithfulness_insertion_auc`, `faithfulness_random_baseline` | producer `models.attention_faithfulness` exists (273 LoC) but has not been wired into `stages.eval` to emit `metrics/faithfulness_metrics.json` on the headline run. | enable `faithfulness_enabled=true` in `configs/default.json` and re-run `--stage eval`. |
| `calibration_{ece,mce,brier}_{pre,post}` | calibration producers (ECE / MCE / Brier) live in `core/extra_stats.py` but are not yet wired to emit `metrics/calibration_metrics.json` for the post-Σ confidence variant. | mechanical instrumentation pass: capture per-receipt witness count `W(ŷ)` in `models/total_arithmetic.py`, emit alongside per-receipt confidence in `predictions/pipeline_preds.jsonl`. |
| `lat_{donut,pipeline,verifier}_{p50,p95,p99,mean}` | requires a dedicated `--stage latency` step (cold-start, batch-1, batch-8 timing burst) separate from the F1-critical eval loop. | add `stages/latency.py` + CLI flag; producer `core.latency_metrics` is already shipped. |

Scoped-out keys appear in `metrics/unresolved_vars.json` after a
successful run.  The corresponding LaTeX figures (e.g.
`fig_yolo_pr.pdf`) are wrapped in `\iffigurefile{}` so they silently
skip when their PDFs are not emitted — no empty floats in the paper.

## Figure → producer matrix

Every Section-C figure (see `report/figures_section_c.py`) early-returns
when its input data is not present.  The matrix below documents the
input dependency:

| figure | input file(s) | status after `make all` |
|---|---|---|
| `fig_curves.pdf` | `curves/donut_loss.csv` etc. | scoped-out (requires Tracker in train loops) |
| `fig_f1_grouped.pdf` | `metrics/extended_metrics.json` | **real** (producer wired) |
| `fig_calibration.pdf` | `predictions/donut_preds.jsonl` + confidences | partial (preds wired; confidences require logprob hook) |
| `fig_latency.pdf` | `metrics/latency_*.json` | scoped-out |
| `fig_cost.pdf` | `cost_donut.json`, `cost_pipeline.json` | **real** |
| `fig_errors.pdf` | `predictions/per_field_errors.jsonl` | **real** |
| `fig_yolo_pr.pdf` | `metrics/yolo_metrics.json` | scoped-out |
| `fig_trocr.pdf` | `metrics/trocr_metrics.json` | scoped-out |
| `fig_assigner.pdf` | `metrics/assigner_diagnostics.json` | partial (top-k wired; entropy requires attention-hook) |
| `fig_gpu_series.pdf` | `curves/gpu_*.csv` | scoped-out |
| `fig_samples.pdf` | `predictions/donut_preds.jsonl` + image paths | **real** |

## Schema versioning

Every JSON schema in `core/schemas.py` declares a `schema_version`
integer so future breaking changes to the producer layout can be
detected and migrated safely.  The current versions are in
`core.schemas.SCHEMA_VERSIONS`.

## Audit recipe

```bash
# After make all finishes:
cat runs/<run_id>/metrics/unresolved_vars.json
# {"unresolved": [<list of scoped-out keys>], "count": N}

# Cross-check against this document: every key in the list MUST appear
# in the "scoped-out keys" table above.  Any unexpected key is a bug.

# The producer → file matrix above is the contract for real values.
```
