# Shared substrate plan (NOT implemented)

This is the structure/plan for the shared benchmark + pipeline that
Section 6 (integrated system) depends on. Implementation is the excluded
"run experiments" step (item 3): NOT done here.

## Why it must be shared
Both axes must run on the SAME receipts so the four-way head-to-head
(composed vs. A vs. B vs. confidence) is apples-to-apples and so the
error-decorrelation measurement is valid.

## Planned layout
```
benchmark/
  data/        # receipt sets per corpus + the natural shift pairs (not included)
  axis_a/      # subset-sum gate runner (component; from prior work)
  axis_b/      # beam-margin variance runner (component; from prior work)
  compose/     # the accept/abstain/flag-shift policy + operating-point sweep
  eval/        # matched-false-alarm / matched-cost protocol, CIs, pre-reg checks
  release/     # frozen artifact for the released benchmark
```

## Interfaces (to be specified before implementation)
- Common per-receipt record: id, corpus, backbone, gold total, predicted
  total, softmax confidence, subset-sum verdict, beam-margin variance.
- Composition consumes that record only; no axis re-implements extraction.
- Eval consumes composition output; all metrics/CIs are the pre-registered
  ones (see ../PREREGISTRATION.md).

## Honesty note
No synthetic or placeholder data is to be committed as if real. The
`data/` and `release/` contents stay empty until the real experiments run.
