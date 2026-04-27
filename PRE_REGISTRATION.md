# Pre-registered hypotheses — kaggle2 PR-E Pareto sweep

This document is the **pre-registered** scaling claim for the 24-cell
Pareto sweep specified in PR-E.  It must be committed *before* the
sweep runs so the published results cannot be retroactively reframed.

## Sweep matrix

| size  | hidden | layers | heads | params  |
|-------|--------|--------|-------|---------|
| tiny  | 96     | 2      | 4     | ≈0.30M  |
| small | 128    | 3      | 4     | ≈0.60M  |
| base  | 192    | 3      | 8     | ≈1.16M  |
| large | 256    | 4      | 8     | ≈2.00M  |

× datasets `{SROIE-347, CORD-100}` × seeds `[42, 1, 2]` = **24 assigner runs**.

Plus reference: DONUT-base, LayoutLMv3-{base, large} on the same
datasets/seeds.  FUNSD is dropped because of schema mismatch.

## Hypotheses

**H1 — Saturation.** The macro-F1 of the YOLO+TrOCR+Attention
pipeline saturates at ≈base size on both SROIE-347 and CORD-100.

> *Operationalisation:* `H1_PASS` ⇔ `f1(base) - f1(small) >= 0.02`
> AND `f1(large) - f1(base) <= 0.01` on both datasets.

**H2 — Param efficiency.** The pipeline matches LayoutLMv3-base
within `ε = 0.05` macro-F1 at `≤25%` of LayoutLMv3-base parameters
on both datasets.

> *Operationalisation:* `H2_PASS` ⇔ `f1(pipeline_base) >=
> f1(layoutlmv3_base) - 0.05` AND `params(pipeline_base) <= 0.25 *
> params(layoutlmv3_base)` on both datasets.

**H3 — Curve shape invariance.** The F1-vs-log(params) curve shape
is the same on SROIE and CORD up to a scale factor.

> *Operationalisation:* `H3_PASS` ⇔ Pearson correlation between
> per-size delta-F1 vectors `(SROIE: f1_size - f1_tiny)` and
> `(CORD: f1_size - f1_tiny)` is `>= 0.85`.

## Reporting protocol

The paper must report `PASS / FAIL / PARTIAL` per `H1 / H2 / H3`
honestly.  No retroactive hypothesis reframing.  The title and
abstract reflect the *actual* frontier discovered by the sweep,
not the original "Beating DONUT" framing.

## Provenance

This file is committed at the start of PR-E and must not be
modified after the first sweep run finishes.  Any subsequent edit
is a violation of pre-registration discipline.

Locked: PR-E open.
