# Research-Integrity Audit & Open Gaps

This file is the **honest accounting** of what FOCUS does and does
not demonstrate. It is consulted before every paper-build so the
abstract / discussion / limitations track the underlying reality.

## 1. What FOCUS does demonstrate (single-seed, single-dataset)

| Claim | Evidence | Caveat |
| --- | --- | --- |
| FOCUS pipeline reaches ≥ 0.85 F1 on SROIE Task-3 (347 test) | `runs/<id>/metrics/pipeline_metrics.json` headline `assigner_f1` | seed 42 only |
| FOCUS uses ~1/3 the parameter count of `donut-base` | 65.77 M vs 200 M (TrOCR-small + YOLOv8n + 1.16 M assigner) | full inference footprint, *not* trainable-only |
| FOCUS uses ~1/2 the parameter count of LayoutLMv3-base | 65.77 M vs 133 M | published LayoutLMv3 number, **not** re-implemented |
| Learned assigner beats rule-based by ≥ 0.10 F1 | `pipeline_metrics.json` `assigner_f1 - rulebased_f1` | same upstream features (YOLO + TrOCR) so the delta is the assigner contribution |
| Per-field attention maps are interpretable | `runs/<id>/figures/fig_attention_heatmap.pdf` | qualitative; no deletion / faithfulness number |

## 2. What FOCUS does NOT yet demonstrate

The gaps below are the difference between this single-seed,
single-dataset measurement and a venue-grade contribution. The
shipped harness is parametric in every dimension below — adding
each result is a configuration edit, not a code change — but the
runs themselves have not been completed.

### 2.1 Multi-seed variance

- **Headline numbers are seed=42 only.**
- The 347-image test split yields a paired-bootstrap 95% CI half-width
  of approximately ±0.02 absolute F1; per-seed variance from the
  full train-eval pipeline (DONUT + YOLO + TrOCR + assigner) has
  not been measured.
- `configs/canonical_5seed.json` configures `seeds=[42,1,2,3,5]` and
  `n_trials=5`; running it produces five independent end-to-end
  evaluations with mean / std / 95% CI fields populated automatically
  in `combined_metrics.json`. **It has not been executed.**
- Until it is, every "matches DONUT" / "beats LayoutLMv3" claim must
  be qualified as a single-seed point estimate, not a population
  statistic.

### 2.2 Cross-dataset generalisation

- All results are SROIE only.
- CORD (Korean restaurant receipts, 30-field ontology) and
  WildReceipt (in-the-wild conditions) are mentioned in the paper's
  related-work section and flagged as future work.
- The pipeline's modularity argument — swap YOLOv8 → DocTR, swap
  TrOCR → Tesseract — has not been tested on a second dataset.
- Until cross-dataset numbers exist the "modular pipeline" framing
  is an architectural statement, not an empirical one.

### 2.3 LayoutLMv3 head-to-head

- Comparators table cites the **published** LayoutLMv3 number on
  Task-3 (0.857 F1, 133 M params).
- We have **not** retrained or re-evaluated LayoutLMv3 on our exact
  hardware / split / preprocessing. `models/layoutlmv3_eval.py`
  exists but is opt-in (`layoutlmv3_enabled=false` in default
  config) and has no shipped checkpoint.
- Until a side-by-side run on the same canonical 347-image test set
  is completed, the FOCUS-vs-LayoutLMv3 comparison is "our 65.77 M
  model on the canonical test vs LayoutLMv3's published 133 M
  number on the canonical test" — same test set, but different
  preprocessing / training environment.

### 2.4 DONUT re-implementation gap

- The shipped run reports DONUT F1 = 0.0000 because DONUT was not
  successfully evaluated on this particular run (the placeholder
  appears throughout the advanced-template paper as
  `0.0000` / `0.0%`).
- A successful DONUT eval on the same canonical 347-image split
  is required before any "matches DONUT" claim can be made
  honestly. The discussion section frames the contribution as
  "matching our re-implemented DONUT" — that re-implementation
  must produce a non-zero F1 first.

### 2.5 Per-component ablation

- The 0.110 F1 "learned vs rule-based" delta is the **net effect**
  of: trained encoder, zone prior, regex router, arithmetic
  consensus, OCR-drift correction, confidence-gated delegation,
  span heads, and registration-suffix stripping.
- We have not isolated the contribution of any single component.
- A `stages.ablate_bugs` runner exists; a "components ablation"
  variant would re-run the eval with each subsystem disabled in
  turn. Not yet implemented.

### 2.6 Inference latency / calibration

- Both are flagged as out-of-scope in the paper's limitations.
- Latency producers (`core/latency_metrics.py`) exist; they were
  not enabled on the headline run.
- Calibration figures are stubbed (ECE/MCE/Brier producers wired,
  reliability diagrams not rendered).

## 3. Eval-symmetry guarantees (now enforced)

The pre-audit code routed pipeline preds/GT through the FOCUS
punctuation+casefold normaliser while DONUT preds/GT went through
the strict-only legacy normaliser. That asymmetry could give
the pipeline a free 0.01–0.03 F1 head-start on receipts where
SROIE GT differed from OCR by trailing punctuation only.

`models/normalize_bundle.py` now defines `FIELD_NORMALISERS_DONUT`
as **identical** to `FIELD_NORMALISERS_PIPELINE`, so both arms see
the same canonical form for company / address / date / total.
Symmetric application to pred AND gold inside `normalize_bundle`
ensures this is canonicalisation, not leakage.

## 4. Hardcoded thresholds vs test-set leakage

All confidence floors and zone gates in `configs/default.json`
(`total_confidence_threshold=0.35`, `focus_company_confidence_floor=0.20`,
`zone_totals_floor=0.5`, etc.) are **hardcoded defaults**. They
were chosen during development on the val split. They are not
tuned per-receipt at eval time and are not refit on the test set.

The zone prior parameters in `data/zone_prior_fit.py` are fit on
the **train split only** (500 receipts of the 626-receipt training
pool). The fit output lives in `results/zone_prior.json`.

## 5. Promotion criteria

A FOCUS result graduates from "single-seed point estimate" to a
publishable claim when **all** of the following hold:

1. `n_trials ≥ 3` with mean / std / paired-bootstrap CI reported.
2. DONUT, FOCUS, and at least one of {LayoutLMv3, BROS, TILT}
   evaluated on the same test split with the same eval harness.
3. At least one second dataset (CORD or WildReceipt).
4. Per-component ablation table populated.
5. Latency + calibration measurements alongside F1.

Until then this file is the visible reminder that the contribution
is **promising-and-reproducible**, not **proven-at-publication-scale**.
