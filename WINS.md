# Decision-gate verdict — PR-B (multi-seed canonical SROIE)

This file records the per-PR decision-gate verdict so reviewers can
see whether the next tier in the
[`assigner-system PR plan`](README.md#prs) was reached without
landing additional code.

## PR-A — Repair pass (T-0)

| Criterion | Status |
|---|---|
| `make check` clean | ✅ |
| `make paper` clean (`grep -nE '\b(1157\|161\|192\|128\|384\|d=N\|L=N\|one[- ]third\|33%\|half)\b' report/sections/`) | ✅ — 0 hits |
| `combined_metrics.json` populates `assigner_d_model`, `assigner_params_m`, `param_ratio_phrase`, `delta_f1_pvalue_mcnemar_exact` | ✅ — see `report/combine_ext.py::merge_assigner_arch` |
| Reference checkpoint `runs/20260427T071206Z-fd9d7b0/` loads bit-exact | ⏳ — verify in CI on first checkpoint upload |
| No "GAT" or "one-third" in compiled PDF | ✅ |

## PR-B — Multi-seed canonical SROIE (T-1)

| Criterion | Status |
|---|---|
| `configs/canonical_5seed.json` committed | ✅ |
| `report/combine_seeds.py::aggregate_seeds` emits `assigner_n_seeds`, `assigner_f1_mean`, `assigner_f1_std`, `delta_f1_mean`, `delta_f1_ci_lo/hi`, `delta_f1_pvalue_mcnemar_exact` | ✅ |
| `inject_tables.py` renders `mean ± stdev` when `n_trials > 1` | ⏳ — wired through existing `n_trials` branch |
| `tests/test_seed_aggregate.py` green | ✅ |

**Decision-gate verdict (placeholder, post-run):**
- `delta_f1_ci_lo` ≥ 0 → **STRONG PASS** — T-1 reached without PR-C.
- `delta_f1_ci_lo` ≥ −0.005 AND `delta_f1_mean` ≥ −0.005 → **PASS**.
- otherwise → **FAIL** — proceed to PR-C.

The actual numbers from the canonical 5-seed run land here on first
GPU build; until then the verdict is `pending`.

## PR-C — F1-pushing strategies (T-1 → T-2 gating)

| Criterion | Status |
|---|---|
| S0 address-assembly scorer (`models/pipeline_consensus_score.py::_score_address_assembly`) | ✅ — scaffold |
| S1 anchor-then-extend address head (`models/attention_model.py`, gated by `address_anchor_extend`) | ✅ — gated, default off |
| S2 priors_v3 distractor disambiguation (`models/attention_priors.py::N_TEXT_PRIORS_V3`) | ✅ |
| S3 concat fusion (`fusion="concat"`) | ✅ — gated |
| Bug-timeline 14–17 toggles in `bug_flags` | ✅ |

## PR-D — Cross-dataset (T-2)

| Criterion | Status |
|---|---|
| `data/cord.py::load_cord` | ✅ |
| `models/layoutlmv3_eval.py::eval_layoutlmv3` | ✅ — gated stub |
| `models/llm_eval.py::eval_llm_zeroshot` | ✅ — gated, content-hash cache |
| `report/figures_carbon.py::emit_carbon_figure` | ✅ |
| `report/sections/section_taxonomy.tex` | ✅ |

## PR-E — Pareto sweep (T-3)

| Criterion | Status |
|---|---|
| `PRE_REGISTRATION.md` committed | ✅ |
| `stages/sweep.py::run_sweep` | ✅ |
| `report/figures_pareto_full.py::emit_pareto_full_figure` | ✅ |
| `data/synthetic_receipts.py::generate_synthetic_receipts` | ✅ |
| `configs/sweep/{tiny,small,base,large}.json` | ✅ |

The sweep cells (24 assigner runs + reference rows) have not yet run;
pre-registration is locked at PR-E open.
