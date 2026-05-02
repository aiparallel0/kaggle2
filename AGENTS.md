# AGENTS.md — kaggle2 two-tree repository

The repository hosts **two independent papers** (`paper2/` and
`paper3/`) plus shared infrastructure (`core/`, `data/`, top-level
`models/`).  Coding agents working in this repository should pick the
correct tree for the work they are doing and never cross paper
boundaries unintentionally.

## TL;DR for any agent opening this repo

1. **Decide which paper your task belongs to.**  If your task is
   about regex assignment, zone-prior HMM, the 14-bug catalogue, or
   the IEEE Access replication study, you are in **paper2/**.  If
   your task is about FOCUS-T cross-attention, FOCUS-Σ verification,
   the GAT/CNN ensemble, the SVKIE multi-prior framework, or the
   ICDAR-main submission, you are in **paper3/**.  If your task is
   about DONUT/YOLO/TrOCR base trainers, dataset loaders, or RNG
   seeding, you are in shared substrate (`core/`, `data/`, top-level
   `models/`).

2. **Never let one paper import from the other.**  `paper2/*` may not
   `import paper3.*` and vice versa.  This is the bifurcation
   contract; violating it makes the two papers indefensible at
   submission time.

3. **Never reintroduce shared LaTeX sections.**  The two papers have
   disjoint section files (`paper2/report/sections/intro.tex`,
   `paper3/report/sections/intro.tex`, etc.).  If a shared paragraph
   needs to exist, it lives in two paper-specific copies, written
   from scratch in each.

4. **Each paper has its own config.**  `paper2/configs/default.json`
   has all FOCUS-T / FOCUS-Σ / GAT flags set to False;
   `paper3/configs/default.json` has them all set to True.  Never
   mix.  Use the per-tree config loaders
   (`paper2/config_paper2.py::load_paper2_config` and
   `paper3/config_paper3.py::load_paper3_config`) which enforce the
   correct flag invariants in code.

5. **Each paper writes to its own runs directory.**  Paper 2 writes
   to `paper2/runs/<id>/`; Paper 3 writes to `paper3/runs/<id>/`.
   The `runs_legacy/` directory contains pre-bifurcation artefacts
   and is read-only.

## Repo map

```
core/         shared dataclasses (Receipt, Metrics, ExpConfig),
              RNG seeding, manifest writer, base metrics, base config
data/         shared SROIE downloader + dataset loaders + crops cache
models/       shared base trainers: DONUT, YOLO, TrOCR + normalisers
              + detect/oracle/miss-tracker (called by both papers)
stages/       shared stage orchestrators (called by both papers'
              main entry points; dispatch by config flags)
paper2/       Paper 2 — IEEE Access replication study
              ├── config_paper2.py    Paper 2 config loader
              ├── main_paper2.py      Paper 2 entry point
              ├── configs/            Paper 2 config presets
              ├── models/             Paper 2-specific code
              │     (rule_*_paper2, zone_prior_paper2,
              │      postprocess_*_paper2, total_post_paper2,
              │      date_post_paper2)
              ├── stages/             Paper 2-specific orchestrators (deferred)
              ├── report/
              │   ├── template_paper2.tex
              │   ├── references.bib
              │   └── sections/       13 disjoint LaTeX sections
              ├── tests/              Paper 2 tests
              ├── results/            Paper 2 fixtures
              ├── docs/               Paper 2 docs
              └── runs/               Paper 2 runs
paper3/       Paper 3 — ICDAR main: SVKIE framework
              ├── config_paper3.py    Paper 3 config loader
              ├── main_paper3.py      Paper 3 entry point
              ├── configs/            Paper 3 config presets
              ├── models/             Paper 3-specific code
              │     (focus_*_paper3, total_arithmetic_paper3,
              │      consensus_paper3, corrections_paper3,
              │      attention_faithfulness_paper3,
              │      layoutlmv3_eval_paper3, etc.)
              ├── stages/             Paper 3-specific orchestrators (deferred)
              ├── report/
              │   ├── template_paper3.tex
              │   ├── references.bib
              │   ├── wrapper_delta_paper3.py
              │   ├── paper_f1_gap_paper3.py
              │   └── sections/       14 disjoint LaTeX sections
              │                       (incl. svkie_theory.tex,
              │                        figure_architecture.tex)
              ├── tests/              Paper 3 tests
              ├── results/            Paper 3 fixtures
              ├── docs/               Paper 3 docs
              └── runs/               Paper 3 runs
runs_legacy/  pre-bifurcation runs, read-only
app/          FastAPI demo (paper-neutral, deployed at image-to-text.fit)
deploy/       nginx + systemd config (paper-neutral)
```

## Hard invariants

1. **Disjoint imports.**  No `paper2.*` import inside `paper3/*`
   files; no `paper3.*` import inside `paper2/*` files.
2. **Disjoint LaTeX.**  No `\input{paper3/...}` inside
   `paper2/report/template_paper2.tex` and vice versa.
3. **Disjoint runs.**  Each paper writes only under its own
   `runs/<id>/` subtree.
4. **`runs_legacy/` is read-only.**  Never write new artefacts there.
5. **Shared substrate stays paper-neutral.**  `core/`, `data/`, and
   the top-level `models/donut_*`, `models/yolo_*`, `models/trocr_*`,
   `models/normalize*`, `models/gen_config`, `models/miss_tracker`,
   `models/oracle`, `models/detect` files do not reference Paper 2 or
   Paper 3 by name in their docstrings or comments.

## What is currently deferred

Per the post-bifurcation status in `README.md`:

- Top-level `models/eval_pipeline.py` and `stages/{train,eval,paper}.py`
  still dispatch by config flags (FOCUS on/off) rather than living in
  per-paper trees.  They should be split into
  `paper2/models/eval_pipeline_paper2.py` /
  `paper3/models/eval_pipeline_paper3.py` and similarly for stages.
- Tests are not yet split into `paper2/tests/` and `paper3/tests/`.
- `make check` does not currently pass post-bifurcation because the
  moved paper-specific files retain imports from their pre-move paths.

These are mechanical fixes scheduled for the next refactor session.

## Verifying a change

```bash
# Paper 2 paper compile (will work once stages are split):
python paper2/main_paper2.py --stage paper

# Paper 3 paper compile (will work once stages are split):
python paper3/main_paper3.py --stage paper

# Shared substrate sanity (current top-level make check; broken
# post-bifurcation, fix queued):
make check
```

## What agents should NOT do

- Add new files at the top level outside `core/`, `data/`, top-level
  `models/`, `stages/`, `app/`, `deploy/`, `docs/`, `scripts/`, `tests/`.
- Cross-import between `paper2/` and `paper3/`.
- Reintroduce shared LaTeX sections under `paper2/report/sections/` or
  `paper3/report/sections/`.
- Modify files under `runs_legacy/`.
- Reintroduce the deleted top-level `report/` directory or
  `paper_fixed.tex`.
