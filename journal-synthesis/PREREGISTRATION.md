# Pre-registration record

Fixed BEFORE any new experiment is run. No new number enters main.tex until
the experiment named here has been executed and its result recorded against
the hypothesis below.

## Primary hypothesis (H1, integrated system)
The composed two-axis policy (Axis A AND Axis B) achieves a strictly better
operating point than (a) Axis A alone, (b) Axis B alone, and (c) confidence
alone, evaluated at MATCHED false-alarm rate AND MATCHED human-review cost on
the shared benchmark.

- Primary endpoint: precision at matched coverage/cost (exact metric fixed
  here before running).
- Decision rule: H1 supported only if the composed policy dominates all three
  baselines on the primary endpoint with pre-specified CIs not overlapping.
- Negative result is publishable and will be reported as such.

## Secondary hypotheses
- H2 (non-redundancy): the two axes' error events are uncorrelated beyond a
  pre-specified threshold (error-decorrelation measurement).
- H3 (Axis B mechanism): margin-variance compression tracks distribution
  distance, not difficulty, under controlled synthetic shifts.
- H4 (Axis A robustness): subset-sum is not dominated by an alternative
  structural verifier on precision at matched coverage.

## Carried-forward null expectations (must NOT be "fixed" by experiments)
- Axis B remains below confidence baselines on raw per-receipt AUROC. This is
  expected, reported, and NOT treated as a failure of the experiment.
- The framework remains an applied formalisation; no methodological-novelty
  claim is registered or pursued.

## Analysis plan
- All CIs and tests specified per endpoint before data collection.
- Multiple-pair / multiple-corpus results reported individually, not pooled
  away.
- Pre-registration vs. final results consistency check is part of the
  excluded internal-QA step (item 5), to be done before submission.

## Frozen experiment list (the excluded "run" step, item 3)
Axis A: bake-off; precision-coverage-cost frontier; power replication;
end-to-end latency.
Axis B: multiple natural pairs; mechanism study; ablations + negative
controls; operational monitor.
Integrated: four-way head-to-head; blind-spot coverage; error-decorrelation;
adversarial/stress probe.
Boundary: re-confirm below-baseline AUROC on the new data.
