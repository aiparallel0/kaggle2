# Experiments E1-E10: necessary-experiment list, triage, and real results

This file is the honest experiment ledger for the synthesis paper. E1-E4 were
RUN as analysis-only on data already on disk (no model inference, no network,
stdlib Python only) and carry their real numbers, exact n/scope, and caveat.
E5-E10 are BLOCKED, each with the concrete one-line reason; they remain
`\pending{...}` verbatim in `main.tex`.

Reproduce: `python3 experiments/run_analysis.py` (deterministic, seed 12345,
verified bit-identical on re-run). Real outputs: `experiments/results_E1_E4.json`.

GLOBAL SCOPE WARNING. E1-E3 are a **CORD-only proxy, n=100, a single
beam-margin-paper pipeline** (`arith-gating/predictions`). This is NOT the
integrated multi-corpus four-way benchmark. E4 only re-confirms PRIOR-WORK
aggregate numbers from `triology/runs`; it claims nothing new. NO per-receipt
join between the arith-gating predictions and the triology runs was performed
anywhere (different pipelines / id spaces); doing so is forbidden and was not
done.

---

## RUN

### E1 - Non-redundancy / error-decorrelation  [RUN]
- Method: inner-join `cord_arith.jsonl` x `cord_signals_receipt.jsonl` on
  `receipt_id`. Axis-A error = NOT `arith_pass` on applicable receipts
  (`n_applicable>0`); `n_applicable==0` receipts have no arithmetic identity,
  so the symbolic axis abstains and they are EXCLUDED from the Axis-A error
  set (not silently scored). Confidence-error proxy = below-median split on
  `c_seq` (and, separately, `softmax_confidence`). phi/Matthews correlation,
  2x2 counts, two-sided permutation p (20000 shuffles, seed 12345).
- Matched join n: **100 / 100** (full match).
- Applicable n = **83**; excluded `n_applicable==0` = **17**.
  Axis-A error rate on applicable = **0.349** (29/83).
- Result (`low_c_seq` proxy; `softmax` proxy identical here):
  **phi/MCC = +0.287**, 2x2 (n11,n10,n01,n00) = (20, 9, 21, 33),
  **permutation p = 0.011** (two-sided).
- Honest interpretation: a WEAK but statistically significant POSITIVE
  association between Axis-A error and low-confidence events on this single
  pipeline (phi^2 ~= 8% shared variance; most error mass is non-overlapping).
  The two error types are NOT fully decorrelated here. This does NOT establish
  or displace the cited prior-work orthogonality claim, which stands on its
  own multi-corpus evidence.
- Caveat: CORD-only proxy, n=100 (83 applicable), single beam-margin-paper
  pipeline; confidence-error is a median-split proxy, not a calibrated error.

### E2 - Composition vs alone  [RUN]
- Method: on the joined CORD set (n=100), three accept policies:
  confidence-alone (`c_seq >= median`), Axis-A-alone (`arith_pass`),
  composed (`arith_pass AND c_seq >= median`). Ground truth: receipt correct
  iff stored predicted `fields == ground_truth` (exact match on already
  normalized strings in `cord_arith`).
- Results (n=100; c_seq median threshold 0.99322):
  - confidence-alone: precision **0.84**, coverage **0.50** (42/50).
  - Axis-A-alone: precision **0.70**, coverage **0.54** (38/54).
  - composed: precision **0.83**, coverage **0.36** (30/36).
  - Blind-spots caught by composition: **10** receipts Axis-A-alone wrongly
    accepts, **2** receipts confidence-alone wrongly accepts.
- Caveat: ground truth is exact-string match on stored normalized fields,
  NOT the paper's official KIE scorer; CORD-only proxy, n=100. Composition
  trades coverage for blind-spot removal; it does not beat confidence-alone
  precision on this single pipeline.

### E3 - Precision-coverage frontier  [RUN]
- Method: sweep `c_seq` threshold over [0,1] step 0.05; report
  (coverage, precision) for confidence-alone vs composed. Full table in
  `experiments/results_E1_E4.json` (`E3.frontier`).
- Endpoints (CORD-only proxy, n=100): at thr 0.0 confidence-alone
  cov 1.00 / prec 0.59 vs composed cov 0.54 / prec 0.70; at thr 0.95
  confidence-alone cov 0.93 / prec 0.63 vs composed cov 0.52 / prec 0.73;
  at thr 1.0 both empty.
- Caveat: same CORD-only proxy scope and ground-truth limitation as E2.

### E4 - Verify-only consolidation of PRIOR-WORK numbers  [RUN, prior work]
- Method: read `triology/runs` AGGREGATE JSON only (PAPER_TABLE.json,
  time_budget_cpu.json). No per-receipt join. Re-confirm the cited numbers.
- Confirmed:
  - Pooled composed precision = **182/184 = 0.989**; recomputed Wilson
    lower bound = **0.961** (matches the stored CI table [0.961, 0.997]).
    main.tex states 0.989 / Wilson LB 0.961 -> MATCH.
  - WildReceipt intersection = **113/114** -> MATCH.
  - CPU latency median = **4.07 us** (standalone DP, CPU-only, as disclosed
    in the prior work's honesty note; CORD excluded there, not fabricated).
- These stay `\priorfact` (cited prior work), NOT new contributions. E4 only
  confirms main.tex did not misquote them.

---

## BLOCKED (remain \pending in main.tex)

### E5 - Cross-pipeline / integrated multi-corpus four-way benchmark  [BLOCKED]
Reason: arith-gating prediction ids and triology run ids are different
pipelines / id spaces; no provably valid per-receipt join exists, so the
integrated cross-corpus decorrelation and four-way head-to-head cannot be
computed honestly.

### E6 - Multiple natural shift pairs (Axis B battery)  [BLOCKED]
Reason: needs new corpora + model inference to produce additional
natural-shift beam-margin distributions; no network, no GPU, not on disk.

### E7 - Controlled mechanism / synthetic-shift study  [BLOCKED]
Reason: requires generating synthetic shifts and re-running model inference
with independently dialed difficulty vs distribution distance; needs new
data + inference, unavailable here.

### E8 - Real end-to-end deployment latency  [BLOCKED]
Reason: no GPU; prior work measured CPU-only standalone DP latency (4.07 us
median, explicitly disclosed). True end-to-end pipeline latency cannot be
measured in this CPU-only environment without fabrication.

### E9 - Alternative-verifier bake-off  [BLOCKED]
Reason: line-item / subtotal+tax / rounding verifiers are not implemented and
the stored fields lack the structured per-line-item data needed to run them
honestly on this data.

### E10 - Power-resolving per-corpus replication / broadened corpora-languages-backbones  [BLOCKED]
Reason: needs new data collection and model inference across additional
corpora, languages, and backbones; no network, no GPU, not on disk.
