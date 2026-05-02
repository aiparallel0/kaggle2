# Legacy runs — pre-bifurcation snapshots

This directory holds run artefacts produced **before** the repository was
bifurcated into the parallel `paper2/` and `paper3/` trees.  Every run
under this folder was emitted by the unified codebase whose entry point
was the top-level `models/`, `stages/`, and `report/` directories — none
of which exist after the bifurcation.

| Run ID | Origin | Notes |
|---|---|---|
| `20260430T125211Z-f598952` | unified codebase, single seed (42) | Headline `pipeline_f1`=0.858 cited in early drafts |
| `20260501T181315Z-16f953f` | wave-3 NeurIPS reframe scaffold | wrapper-Δ producer wired but no real measurements |
| `20260502T110548Z-ef8b741` | DONUT paper-recipe scaffold | recipe locked to Kim et al. 2022 specs |

These artefacts are kept for **provenance only**.  They cannot be
reproduced by either of the two new paper trees because their producing
code has been moved to per-paper paths and renamed.

New runs land under `paper2/runs/<run_id>/` (Paper 2 system) or
`paper3/runs/<run_id>/` (Paper 3 system); both trees emit their own
self-contained MANIFEST and metrics sidecars.

Do not delete this directory — it is the audit trail for the
bifurcation cutover.
