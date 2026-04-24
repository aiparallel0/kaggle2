# AGENTS.md — Agent Guide for kaggle2

This file is the single source of truth for coding agents (Copilot coding agent,
Codex, Cursor, Aider, Claude Code, etc.) working in this repository.
Read it in full before making any change.

---

## TL;DR for agents

- **≤166 LOC per file** in the core pipeline dirs (`core/`, `data/`, `models/`,
  `report/`, `stages/`, `main.py`). Count logical lines; blank lines and
  docstrings still count.
- **≤18 files total** across those same dirs. Prefer editing an existing file
  over adding a new one.
- **Every public function is 2-in/1-out typed**: at most two arguments (a typed
  input + a typed config/context), exactly one typed return value. Use the
  types in `core/types.py` (`Receipt`, `Metrics`, etc.) — never ad-hoc dicts.
- **`make check` is the test suite** (`mypy --strict` over the pipeline dirs +
  `ruff` + import smoke). A mypy error is a build failure. Run it before
  opening a PR.
- **Never write to `results/` at runtime** — that directory is fixtures-only
  and git-tracked. All run artefacts go under `runs/<run_id>/` via
  `core/runlayout.py::resolve_layout`. Do not hard-code paths under `runs/`.
- **Prefer small, surgical diffs.** The repo's value proposition is that the
  whole pipeline fits in a context window.

---

## Repo map

```
core/         config schema + validation, typed dataclasses (Receipt, Metrics,
              ExpConfig), errors, shared metrics, seed_everything, runlayout,
              manifest, env_snapshot, schemas
data/         SROIE download, train/val/test split (persisted to
              results/split.json), crop extraction helpers
models/       donut_train, donut_eval, yolo_train, trocr_train,
              assigner_train, attention_assign, pipeline_eval,
              pipeline_consensus, _gen_config
report/       LaTeX template injection (inject_vars, inject_tables),
              figure emitters, combine (aggregates all metrics sidecars),
              references
scripts/      vastai_bootstrap.sh — one-shot install + make check
stages/       Orchestrator targets: train.py, eval.py, paper.py.
              Invoked by main.py via --stage train | eval | paper | all.
              Also: eval_gtocr_rulebased (rule-based baseline, CPU-only)
app/          FastAPI demo server (drag-and-drop receipt inference UI).
              Served by `make serve` → http://localhost:8000
deploy/       kaggle2.service (systemd) + nginx-teb2.conf — production
              deployment at https://portearchive.com/teb2/
tests/        pytest suite — split persistence, metrics, configs, no GPU
main.py       CLI orchestrator: --stage train | eval | paper | all
              (also eval_gtocr_rulebased / eval_rulebased_gold alias)
```

`stages/` is the boundary between user-facing CLI flags and model code.
`app/` and `deploy/` are excluded from the 18-file cap.

---

## Hard invariants (do not break)

1. **File-count cap.** The core pipeline (`core/` + `data/` + `models/` +
   `report/` + `stages/` + `main.py`) must contain **≤ 18 files**. `tests/`,
   `scripts/`, `deploy/`, and `app/` are excluded. Before adding a new file,
   check whether the logic can be absorbed into an existing module.

2. **Per-file LOC cap.** Every file in those same directories must be
   **≤ 166 logical lines**. When a file approaches the cap, refactor in place
   or move helpers into an existing file — do not spawn a new module casually.

3. **2-in/1-out contracts.** Every public function takes at most two arguments
   (a typed input and a typed config/context) and returns exactly one typed
   value. Use `core/types.py` types (`Receipt`, `Metrics`, `ExpConfig`, etc.)
   rather than untyped `dict`. Keyword-only helper arguments are permitted for
   pure-utility private functions (`_foo`).

4. **mypy-as-test.** `make check` runs `mypy --strict` over `core/`, `data/`,
   `models/`, `report/`, `stages/`, and `main.py`. A type error is a build
   failure. Do not add `# type: ignore` without a one-line justification
   comment on the same line.

5. **`results/` is fixtures-only and git-tracked.** `runs/<run_id>/` is
   git-ignored and is the **only** place stages may write artefacts. All
   writers must call `core/runlayout.py::resolve_layout` to obtain concrete
   paths. Never hard-code a path string under `runs/`. See `results/README.md`
   and `runs/README.md` for the full contract.

6. **The 13 F1-destroying bugs are guarded.** The bugs enumerated in
   `README.md` (lm_head dedup, decoder_start_token_id, token2json list merge,
   fp16 overflow, YOLO imgsz mismatch, TrOCR under-training, val==test
   leakage, YOLO path resolution, stale generation_config, tie_word_embeddings,
   num_items_in_batch kwargs, outer-tag flattening, warmup precedence) each
   have a guard assertion or code path in the repo. **Do not remove or relax
   those guards.** If you touch `models/_gen_config.py`,
   `models/donut_train.py`, `models/donut_eval.py`, or the YOLO/TrOCR
   trainers, re-read the bug list in `README.md` first.

7. **Reproducibility.** `core.seed.seed_everything(config.seed)` must be
   called at startup of any new entrypoint. Any new `DataLoader` must use the
   seeded `worker_init_fn` + `torch.Generator` pattern already present in the
   existing trainers.

8. **Split persistence.** The 500/63/63 SROIE split is persisted to
   `results/split.json` on the first train run. Never regenerate the split
   during `eval` — load it from `results/split.json`.

9. **No silent placeholders.** Every `\VAR{}` key in the LaTeX paper must be
   resolved by a real producer listed in `docs/TRACKING.md`. If you add a new
   metric to the paper, add its producer and list it in `docs/TRACKING.md`.
   Unresolved keys land in `runs/<run_id>/metrics/unresolved_vars.json` and
   are a review blocker.

---

## How to verify a change

Run these commands in order; each command proves something distinct:

```bash
make check    # ruff + mypy --strict + import smoke (≤60 s, no GPU)
make test     # pytest: split persistence, metrics, configs (no GPU)
python main.py --stage eval_gtocr_rulebased   # CPU-only F1 smoke, no HF Hub
make all      # full train + eval + paper; GPU required (≈75 min on RTX 4090)
```

- **`make check` and `make test` are required before opening any PR.**
- **`make all` is required only for changes that could affect F1** (trainer
  code, config defaults, eval logic, assigner, pipeline).

---

## Editing guidance for agents

- **Small, surgical diffs.** The whole pipeline fits in a context window; keep
  it that way. Large refactors that don't affect behaviour are not welcome.

- **Trainer ↔ eval parity.** When modifying a trainer
  (`models/*_train.py`), update the matching eval (`models/*_eval.py` or
  `stages/eval.py`) in the same PR.

- **Config changes need three touch-points.** When adding a config key: add it
  to `config.json`, validate it in `core/config.py`, and document it in the
  `README.md` "Configuration" table.

- **New metrics need four touch-points.** Wire it through:
  1. Producer in `models/` or `stages/eval.py`
  2. Aggregator in `stages/paper.py` / `report/combine.py`
  3. `\VAR{}` reference in the `report/` template
  4. Row in `docs/TRACKING.md`

- **Never write to `./results/` or `./report/` at runtime.** Always route
  through `core/runlayout.resolve_layout`.

- **New dependencies need three touch-points.** Add to `requirements.txt` AND
  `pyproject.toml`, then re-run `make check` to confirm no import errors.

- **Do not add `# type: ignore`** without a one-line justification on the same
  line, e.g. `# type: ignore[attr-defined]  # ultralytics stub gap`.

---

## Round-trip contract (vast.ai ↔ Copilot)

After a successful `make all`, every artefact lives under one self-contained
folder:

```
runs/<run_id>/
├── metrics/          combined_metrics.json, extended_metrics.json, …
├── curves/           per-epoch CSV time series
├── predictions/      donut_preds.jsonl, pipeline_preds.jsonl, …
├── attention/        attention_samples.npz + meta JSON
├── figures/          every PDF/PNG cited in the paper
├── paper/            paper_filled.tex + paper_filled.pdf
├── env/              git_sha.txt, pip_freeze.txt, config_snapshot.json, …
└── MANIFEST.json     relpath + sha256 + size_bytes + producer_stage
```

Pack / transfer / verify:

```bash
# On vast.ai:
make all                               # produces runs/<run_id>/
make pack                              # → <run_id>.tar.zst + .sha256

# Restore + verify on the receiving side:
bash scripts/unpack_run.sh <archive>.tar.zst
```

When reasoning about a received archive, **trust `MANIFEST.json` as the index
of truth** and verify sha256 before drawing conclusions about metrics. Every
file listed in `MANIFEST.json` has a `producer_stage` field (train / eval /
paper / bootstrap) so stale vs. fresh artefacts are distinguishable.

`run_id` format: `<UTC>-<git-short-sha>`, e.g. `20260424T103055Z-a1b2c3d`,
generated by `core.runlayout.make_run_id`.

---

## What agents should NOT do

- **Do not add new top-level directories.**
- **Do not split a module "for readability"** if the original is still under
  the 166 LOC cap. Splitting is only justified when the file would otherwise
  exceed the cap.
- **Do not add ML framework abstractions** (trainer wrappers, config
  frameworks, plugin systems, experiment-tracking integrations). The repo's
  thesis is that plain PyTorch + HF Transformers + a strict layout is enough.
- **Do not commit anything under `runs/`**, `data/sroie_cache/`, or
  fine-tuned weights.
- **Do not modify `results/bug_timeline.json`** — it is a reviewer fixture
  seeded into each run archive by `stages/paper.py`.
- **Do not rename `--stage eval_rulebased_gold`** — it is a backward-compatible
  alias for `--stage eval_gtocr_rulebased` and must remain resolvable.
- **Do not introduce unresolved `\VAR{}` keys** in the LaTeX templates without
  adding a corresponding producer and updating `docs/TRACKING.md`.

---

## Where to look first

| File / path | Purpose |
|---|---|
| `main.py` | CLI orchestrator; add `--stage` flags here |
| `core/config.py` | Config schema, validation, all guard assertions |
| `core/runlayout.py` | Path contract; every writer must call `resolve_layout` |
| `core/types.py` | Typed dataclasses: `Receipt`, `Metrics`, `ExpConfig`, … |
| `stages/train.py` | Training stage orchestration |
| `stages/eval.py` | Evaluation stage, McNemar test, split loading |
| `stages/paper.py` | Paper assembly, `\VAR{}` injection, fixture seeding |
| `models/donut_train.py` | DONUT fine-tuning; see Bug 1/2/3/4/9/10/11/12/13 guards |
| `models/pipeline_consensus.py` | Inference-side strategies L + H (no retrain needed) |
| `docs/TRACKING.md` | Metric producer matrix — consult before adding a `\VAR{}` |
| `results/README.md` | Why `results/` is fixtures-only |
| `runs/README.md` | Per-run layout, pack/unpack round-trip |

---

*Last verified: 6d96f24*
