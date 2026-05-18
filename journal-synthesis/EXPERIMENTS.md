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
PREPARED FOR VAST.AI: `vastai/e5_integrated_benchmark.py` (runs both axes
through ONE pipeline on a shared receipt set so receipt_ids align by
construction, fixing the join blocker; H1 four-way head-to-head + H2
decorrelation). See `vastai/README_RUNBOOK.md`. STATUS HERE: still
BLOCKED (no GPU/models/network); not run; no results committed.

### E6 - Multiple natural shift pairs (Axis B battery)  [BLOCKED]
Reason: needs new corpora + model inference to produce additional
natural-shift beam-margin distributions; no network, no GPU, not on disk.
PREPARED FOR VAST.AI: `vastai/e6_multi_shift_pairs.py` (parameterised
pair list; reuses arith-gating beam-margin extraction; log2 variance
ratio + KS + permutation p per pair). See `vastai/README_RUNBOOK.md`.
STATUS HERE: still BLOCKED; not run; no results committed.

### E7 - Controlled mechanism / synthetic-shift study  [BLOCKED]
Reason: requires generating synthetic shifts and re-running model inference
with independently dialed difficulty vs distribution distance; needs new
data + inference, unavailable here.
PREPARED FOR VAST.AI: `vastai/e7_mechanism_synthetic_shift.py`
(independent difficulty vs distribution-distance knobs; margin-variance
vs location signals; beam-width/length-norm/architecture ablation
hooks; H3). See `vastai/README_RUNBOOK.md`. STATUS HERE: still BLOCKED;
not run; no results committed.

### E8 - Real end-to-end deployment latency  [BLOCKED]
Reason: no GPU; prior work measured CPU-only standalone DP latency (4.07 us
median, explicitly disclosed). True end-to-end pipeline latency cannot be
measured in this CPU-only environment without fabrication.
PREPARED FOR VAST.AI: `vastai/e8_end_to_end_latency.py` (real GPU
end-to-end decode+gate latency, triology timing methodology, warmup
discarded; carries the 4.07 us CPU prior figure forward verbatim as
contrast, never overwrites it). See `vastai/README_RUNBOOK.md`. STATUS
HERE: still BLOCKED; not run; no results committed.

### E9 - Alternative-verifier bake-off  [BLOCKED]
Reason: line-item / subtotal+tax / rounding verifiers are not implemented and
the stored fields lack the structured per-line-item data needed to run them
honestly on this data.
PREPARED FOR VAST.AI: `vastai/e9_alt_verifier_bakeoff.py` (implements
line-item x qty, subtotal+tax, rounding verifiers, faithful to triology
identities; precision/coverage/orthogonality vs subset-sum on freshly
decoded receipts that DO carry per-line-item structure; H4). See
`vastai/README_RUNBOOK.md`. STATUS HERE: still BLOCKED; not run; no
results committed.

### E10 - Power-resolving per-corpus replication / broadened corpora-languages-backbones  [BLOCKED]
Reason: needs new data collection and model inference across additional
corpora, languages, and backbones; no network, no GPU, not on disk.
PREPARED FOR VAST.AI: `vastai/e10_power_and_breadth.py` (per-corpus /
per-backbone composed precision + Wilson CI half-width vs a
pre-registered target + additional-n-needed solve; broadened
corpora/backbone via list args). See `vastai/README_RUNBOOK.md`.
STATUS HERE: still BLOCKED; not run; no results committed.

### Severe tests (S1-S4) - robustness-check the NEGATIVE; NOT thesis-rescue  [PREPARED, cache-only]
PURPOSE (stated honestly): the clean decode-once run produced a NEGATIVE
for the journal thesis (H1 composed WORSE than every single signal at
matched cost: -0.25 / -0.41 / -0.34; H3 margin-variance tracks DIFFICULTY
Spearman -0.68 not SHIFT +0.05; H2 error-association weak and
SIGN-UNSTABLE across runs: +0.29 -> +0.21 -> -0.11). S1-S4 exist ONLY to
check whether that negative is ROBUST or a small-n / OCR-cord_dev /
difficulty-confound artifact. They are explicitly designed to let the
negative STAND if it is real. NO tuning, threshold-picking, or
stratification choice here can manufacture a positive; every decision
rule is fixed before the data is read and the interpretation rule is
written into `results/SEVERE.json` next to each number.

- S1 DIFFICULTY-CONTROLLED DECORRELATION: bin receipts by difficulty
  proxy (1 - c_seq) into quartiles; within each stratum compute phi/MCC
  between the Axis-A error event and the confidence-error event with
  permutation p and a bootstrap CI on phi. Verdict rule: the H2
  association was the difficulty confound (robust-negative) if |phi|->~0
  and p is not significant within every computable stratum; it only
  "survives" if phi stays non-trivial with a STABLE SIGN across strata
  AND corpora. Sign-instability = artifact, negative stands.
- S2 PLACEBO-AXIS NEGATIVE CONTROL: replace real Axis-A with a random
  gate matched to Axis-A's empirical accept rate (seeded, >=200 placebo
  draws); compare composed(real A, B) vs composed(placebo, B) on
  precision-at-matched-coverage with a bootstrap CI on the difference.
  Verdict rule: real Axis-A composition only "wins" if the bootstrap CI
  lower bound on (real - placebo) precision is > 0; otherwise the
  two-axis story is illusory and the H1 negative stands.
- S3 POWER / MINIMUM DETECTABLE EFFECT: by simulation, the smallest true
  H1 advantage (composed minus best single) detectable at this n with
  80% power, plus the achieved post-hoc CI width. Verdict rule:
  CONCLUSIVE iff the 80%-power MDE <= a pre-stated 0.05 abs-precision
  care-about gap AND the post-hoc CI excludes a +0.05 advantage; else
  UNDERPOWERED (cannot distinguish "no effect" from "no power" - this
  does NOT rescue the thesis, it only forbids calling the negative
  conclusive).
- S4 SPLIT-STABILITY: recompute H1 matched-cost deltas and
  Spearman(margin-variance, difficulty) / (margin-variance, c_seq) for
  cord_dev-only, wildreceipt-only AND pooled, each with bootstrap CIs.
  Verdict rule: the negative is STABLE only if the sign of every H1
  delta and the H3 difficulty-Spearman is the same across both corpora
  and pooled; a corpus-specific sign flip means the pooled negative is
  driven by one corpus / the OCR set and is reported as corpus-specific,
  not a thesis-level refutation.

COST/SCOPE: zero GPU, zero decode. `vastai/severe_tests.py` reads ONLY
the existing Stage-A decode-once cache via the same cache-load path the
CPU experiments use; it REFUSES to run if a corpus cache is missing
(never invents records) and writes `results/SEVERE.json` only with a
real `computed_on`. Wired as run_parallel.sh "STAGE C" after Stage B
(resumable via the same computed_on skip).

HARD LIMITATION the severe tests CANNOT remove: the on-box CORD is the
OCR-derived n=100 `cord_dev` validation split, not a non-OCR gold CORD
*test* set. The live HF mirror (`naver-clova-ix/cord-v2`, Donut-style:
`ground_truth`+`image`, no words/bboxes) makes `fetch_data.py` REQUIRE
`--ocr` for the token arrays, and the gold CORD *test* split was not
obtainable in the prep environment (HF mirror fetch failed). The gold
totals themselves come from `_decode_donut_gt(ground_truth)` (NOT OCR),
so S1-S4 still operate on faithful gold totals; but a TRUE non-OCR gold
CORD *test* set is NOT feasible with the available fetchers here. S1-S4
robustness-check WITHIN the cord_dev + wildreceipt scope; they cannot
and do not claim to lift the OCR-cord_dev scope itself.

### Full-scale E1-E3 (lift the n=100 CORD proxy caveat)
PREPARED FOR VAST.AI: `vastai/e1e3_fullscale.py` re-runs E1/E2/E3 with
the SAME metric math as `experiments/run_analysis.py` on the full
fetched corpus once models/data are available, removing the GLOBAL
SCOPE WARNING (CORD-only, n=100, single pipeline). STATUS HERE: the
n=100 proxy E1-E3 RUN numbers above are unchanged and remain the only
recorded values; the full-scale re-run is not run and commits no
results.

---

## CI/CD automation: `vastai/cicd/`

`vastai/cicd/` automates the **infrastructure** around these
experiments (provision a GPU -> bootstrap -> `run_parallel.sh` Stages
A-C -> fetch raw results -> generate an audit checklist -> open a
**human-gated** pull request -> destroy the GPU), plus a local-GPU
variant and a braked self-repeating loop. See `vastai/cicd/README.md`.

**HUMAN GATE (reiterated).** This automation is infrastructure only. It
NEVER edits `main.tex`, never writes a paper claim or number, never
decides a scientific outcome, and never auto-merges. Every generated PR
is labelled `needs-human-audit` with a "DO NOT MERGE INTO PAPER WITHOUT
HUMAN SIGN-OFF vs PREREGISTRATION.md" banner. No number reaches the
paper until a human has audited it against `PREREGISTRATION.md` and
signed off. The scientific decision stays human.
