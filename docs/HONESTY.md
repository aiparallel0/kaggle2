# Research-Integrity Audit & Open Gaps

This file is the **honest accounting** of what FOCUS-Σ does and does
not demonstrate.  It is consulted before every paper-build so the
abstract / discussion / limitations track the underlying reality.

> **2026-05 update — NeurIPS reframing.**  The project's central
> contribution has been reformulated from "comparative DONUT vs
> Pipeline study on SROIE" to "FOCUS-$\Sigma$: Subset-Sum Arithmetic
> Witnesses for Verified Information Extraction".  See
> `docs/RESEARCH_DECONSTRUCTION.md` for the cold-reading that
> motivated the reframing and `report/template_neurips.tex` for the
> NeurIPS paper structure.  This file remains the source of truth
> for what is and is not measurable on the current artefacts.
>
> **Section coherence (post-2026-05).**  The NeurIPS template now
> uses dedicated `report/sections/experiments_neurips.tex` and
> `report/sections/results_neurips.tex` files that match the
> wrapper-$\Delta$ × ablation × cross-dataset narrative promised by
> `intro_neurips.tex` (the IEEE/ICDAR template continues to use the
> original `experiments.tex` / `results.tex`).  The wrapper-$\Delta$
> producer (`report/wrapper_delta.py`) consumes the
> `metrics/wrapper_delta_metrics.json` /
> `metrics/ablation_focus_sigma.json` /
> `metrics/error_decomposition.json` /
> `metrics/faithfulness_metrics.json` /
> `metrics/calibration_metrics.json` /
> `metrics/latency_metrics.json` sidecars and forwards their flat
> scalars to `\VAR{}` keys.  When a sidecar is absent the producer
> deliberately leaves its keys unresolved — no synthetic numbers are
> fabricated — and the corresponding cells appear in
> `metrics/unresolved_vars.json` for the audit gate.  See
> `docs/TRACKING.md` for the per-key path-to-real-value table.

## 1. What FOCUS-Σ does demonstrate (single-seed, single-dataset, headline)

| Claim | Evidence | Caveat |
| --- | --- | --- |
| Pipeline reaches 0.858 macro F1 on canonical SROIE Task-3 (n=347) | `runs/20260430T125211Z-f598952/combined_metrics.json` headline `pipeline_f1` = 0.858 | seed 42 only; n=5 sweep configured (`configs/canonical_5seed.json`, `scripts/run_5seed_sweep.sh`) but not yet executed |
| Pipeline beats DONUT macro F1 (0.858 vs 0.827) | same artefact, `donut_f1`=0.827, `pipeline_f1`=0.858, `f1_gap`=−0.031 | seed 42; paired-bootstrap CI on Δ = ±0.022 — sign is resolved at this n |
| FOCUS uses ~⅓ of `donut-base` parameters | 65.77 M (TrOCR-small + YOLOv8n + 1.16 M assigner) vs ≈200 M | full inference footprint |
| Learned assigner beats rule-based by 0.110 F1 | `assigner_f1 - rulebased_f1` ≈ 0.110 on canonical 347 split | net delta of all FOCUS components — see §2.5 below |
| Per-field attention maps are interpretable | `runs/<id>/figures/fig_attention_heatmap.pdf` | qualitative; no deletion / insertion faithfulness measurement (see §2.7) |
| FOCUS-Σ Identity 3 is sound and complete under standard receipt construction | `report/sections/focus_sigma_theory.tex` Prop. 1 + Prop. 2 | proofs assume the itemisation hypothesis (eq. 2); §6 catalogues counter-examples |

## 2. What FOCUS-Σ does NOT yet demonstrate

The gaps below are the difference between this single-seed,
single-dataset measurement and a NeurIPS-tier contribution.  The
shipped harness is parametric in every dimension below — adding each
result is a configuration edit, not a code change — but the runs
themselves have not been completed.

### 2.1 Multi-seed variance

- Headline numbers are seed=42 only.
- The 347-image test split yields a paired-bootstrap 95% CI half-width
  of approximately ±0.022 absolute F1.
- `configs/canonical_5seed.json` configures `seeds=[42,1,2,3,5]` with
  `n_trials=5`.  **Run `bash scripts/run_5seed_sweep.sh` on a vast.ai
  instance to execute and aggregate** (≈ 6.5–8 hours wall-clock on a
  single RTX 4090; ≈ 90 min if parallelised across 5 instances via
  `scripts/vastai_swarm.sh`).
- The aggregator script is `scripts/aggregate_seeds.py`; it produces
  `aggregate.csv` and `aggregate.json` with mean / std / 95% CI per
  metric key.

### 2.2 Cross-dataset generalisation (CORD-v2)

- All current results are SROIE only.
- **`data/cord.py` is now a real implementation** (was a stub before
  the 2026-05 NeurIPS reframing): it loads `naver-clova-ix/cord-v2`
  via `datasets.load_dataset`, materialises images to disk, and
  exposes `CordReceipt.item_prices()` for FOCUS-Σ subset-sum
  verification.
- The pipeline / DONUT / LayoutLMv3 evaluation arms have not yet been
  run on CORD; the loader is gated behind `cord_eval_enabled` (to be
  added to `configs/default.json` before the first cross-dataset run).
- Running CORD requires only adding `--cross-dataset cord` to the
  eval CLI; the eval producer is dataset-agnostic by construction.

### 2.3 LayoutLMv3 head-to-head

- **`models/layoutlmv3_eval.py` is now a real implementation** (was a
  70-line stub returning zeroed Metrics before the 2026-05 reframing).
  The new file (332 LoC) implements:
  - Token-classification fine-tune (`train_layoutlmv3`) following the
    public LayoutLMv3 SROIE recipe (5e-5 LR, 8 epochs, batch 8,
    BIO-tagged token labels over the 4-field schema).
  - Per-receipt forward pass (`LayoutLMv3Predictor.predict_one`) that
    consumes the same YOLO+TrOCR word-level input as the FOCUS arm
    (head-to-head fairness).
  - Symmetric normalisation through `models.normalize_bundle`.
- The fine-tune itself has not yet been run on a GPU box.  Setting
  `layoutlmv3_enabled=true` in `configs/default.json` plus running
  `make all` triggers it on the next vast.ai run.

### 2.4 DONUT re-evaluation

- The 20260430T125211Z run reports DONUT F1 = 0.827 (success).
- The earlier-version-of-this-file claim that "DONUT F1 = 0.0000"
  applied to a previous run; closed by the 2026-05 trace pass.

### 2.5 Per-component ablation

- The 0.110 F1 "learned vs rule-based" delta is the **net effect** of:
  trained encoder + zone prior + regex router + arithmetic consensus +
  OCR-drift correction + confidence-gated delegation + span heads +
  registration-suffix stripping + FOCUS-Σ Identity 3 + bare-TAX
  demoter.
- **`scripts/run_focus_ablation.sh` is now a real runner.**  Six rows:
  baseline → +FOCUS-T → +FOCUS-Σ I₃ → +OCR-drift 1-edit → +OCR-drift
  2-edit → +retrain knobs.  Each row produces a full `runs/<id>/`
  directory; the aggregator emits `ablation.csv` summarising the
  marginal delta of each component.
- Not yet executed.

### 2.6 Inference latency / calibration

- Both flagged as out-of-scope in the paper's limitations section.
- Latency producers (`core/latency_metrics.py`) exist; they were not
  enabled on the headline run.
- Calibration figures are stubbed (ECE/MCE/Brier producers wired,
  reliability diagrams not rendered).

### 2.7 Faithfulness of the attention-map interpretability claim

- The paper describes the attention map as "directly interpretable"
  but does not quantify this.
- Standard interpretability measures (deletion AUC, insertion AUC,
  pointing game accuracy) are not yet implemented.
- Adding `models/attention_faithfulness.py` is a 100-line job; it
  is the cheapest quantitative complement to the qualitative
  attention-heatmap figure (Fig. 4).

### 2.8 FOCUS-Σ theoretical soundness counter-examples

- Section §6 of `report/sections/focus_sigma_theory.tex` catalogues
  three failure modes: bundling discounts, rounding adjustments,
  service charges.
- Empirical measurement of how often each fires on SROIE / CORD is
  not yet in the run artefacts; it requires a per-receipt I₃ witness
  count that the eval pipeline does not currently log (one extra
  field in `runs/<id>/predictions/per_field_errors.jsonl` would
  capture it).

## 3. Eval-symmetry guarantees (now enforced)

The pre-audit code routed pipeline preds/GT through the FOCUS
punctuation+casefold normaliser while DONUT preds/GT went through
the strict-only legacy normaliser.  That asymmetry could give the
pipeline a free 0.01–0.03 F1 head-start on receipts where SROIE GT
differed from OCR by trailing punctuation only.

`models/normalize_bundle.py` now defines `FIELD_NORMALISERS_DONUT`
as **identical** to `FIELD_NORMALISERS_PIPELINE`, so both arms see
the same canonical form for company / address / date / total.
Symmetric application to pred AND gold inside `normalize_bundle`
ensures this is canonicalisation, not leakage.

The same map is now applied to the LayoutLMv3 arm (in
`models/layoutlmv3_eval.eval_layoutlmv3` after the per-receipt
forward pass), closing the symmetric-normalisation guarantee
across all three arms.

## 4. Hardcoded thresholds vs test-set leakage

All confidence floors and zone gates in `configs/default.json`
(`total_confidence_threshold=0.35`, `focus_company_confidence_floor=0.20`,
`zone_totals_floor=0.5`, etc.) are **hardcoded defaults**.  They were
chosen during development on the val split.  They are not tuned
per-receipt at eval time and are not refit on the test set.

The zone prior parameters in `data/zone_prior_fit.py` are fit on the
**train split only** (500 receipts of the 626-receipt training pool).
The fit output lives in `results/zone_prior.json`.

## 5. Promotion criteria

A FOCUS-Σ result graduates from "single-seed point estimate" to a
publishable claim when **all** of the following hold:

1. `n_trials ≥ 3` with mean / std / paired-bootstrap CI reported
   (run `scripts/run_5seed_sweep.sh`).
2. DONUT, FOCUS-Σ, and LayoutLMv3 all evaluated on the same canonical
   SROIE 347-image test split with the same eval harness
   (the 2026-05 LayoutLMv3 implementation makes this a single config
   flip).
3. At least one second dataset evaluated (CORD-v2; the 2026-05
   `data/cord.py` makes this a single CLI flag).
4. Per-component ablation table populated
   (run `scripts/run_focus_ablation.sh`).
5. Latency + calibration measurements alongside F1.
6. FOCUS-Σ counter-example incidence measured per dataset (§2.8).

## 6. Reproducibility checklist

NeurIPS-2024 reproducibility checklist filled in
`report/sections/repro_checklist.tex`.  Every claim there points to
either a configuration file, a script, or an artefact in
`runs/<run_id>/`.

Until all six promotion criteria hold this file is the visible
reminder that the contribution is **promising-and-reproducible**, not
**proven-at-publication-scale**.
