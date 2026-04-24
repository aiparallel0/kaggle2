# `results/` — fixtures-only, fully git-tracked

This directory contains **shipped reviewer fixtures** only.  It is
fully tracked in git and is never written to at runtime.  Every
run-generated artefact (metrics JSON, training curves, predictions,
attention samples, figures, compiled paper, environment snapshots)
goes under `runs/<run_id>/` — see `core/runlayout.py` for the full
layout contract and `runs/README.md` for the per-run sub-directory
breakdown.

## Contents

| File               | Purpose                                                   |
|--------------------|-----------------------------------------------------------|
| `bug_timeline.json`| Reviewer-facing F1 before/after timeline for the 13 silent F1-destroying bugs.  Consumed by `report.figures_bugs.render_bug_timeline`; seeded into `runs/<run_id>/` at paper-stage start by `stages/paper.py::_seed_bug_timeline_fixture`. |

## Rules

1. Nothing under `results/` is written to by `stages/train.py`,
   `stages/eval.py`, `stages/paper.py`, or any `models/*_train.py`.
   Writers must route through `core/runlayout.py`.
2. New fixtures are added only when a figure or table emitter needs a
   reviewer-auditable baseline that cannot be regenerated from the
   run's own metrics JSON.
3. The `.gitignore` at the repo root ignores `runs/` wholesale but
   leaves `results/` tracked.  Do not re-introduce `results/*` +
   `!results/bug_timeline.json` exceptions.

## Why this matters for vast.ai → Copilot round-trips

At the end of a training run the operator needs exactly one folder
worth downloading: `runs/<run_id>/`.  That folder is self-contained
(no references back into the repo tree) and carries a `MANIFEST.json`
enumerating every file with `sha256` / `size_bytes` / `mtime_utc` /
`producer_stage`.  `scripts/pack_run.sh` tars it into a single
`<run_id>.tar.zst` + `.sha256` sidecar that uploads cleanly to a
Copilot PR review thread.  Keeping `results/` strictly fixture-only
is what makes that single-archive round-trip work.
