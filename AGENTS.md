# AGENTS.md — Agent Guide for kaggle2

Single source of truth for coding agents (Copilot coding agent, Codex, Cursor,
Aider, Claude Code, etc.).

---

## TL;DR

- **≤166 LOC per file** in `core/`, `data/`, `models/`, `report/`, `stages/`,
  `main.py`. Blank lines + docstrings count.
- **2-in/1-out typed contracts**: every public function takes at most two args
  (typed input + typed config), returns one typed value. Use `core/types.py`.
- **`make check` is the test suite** (`mypy --strict` + `ruff` + import smoke).
- **Never write to `results/`** (fixtures-only). All outputs go under
  `runs/<run_id>/` via `core/runlayout.resolve_layout`.
- **Small, surgical diffs.** The whole pipeline fits in a context window.

---

## Repo map

```
core/         config schema, types (Receipt, Metrics, ExpConfig), errors,
              shared metrics, seed, runlayout, manifest, env_snapshot
data/         SROIE download, split (persisted to results/split.json), crops
models/       donut_train, donut_eval, yolo_train, trocr_train, focus_train,
              focus_inference, eval_pipeline, consensus, oracle, gen_config
report/       LaTeX template injection, figure emitters, combine
stages/       Orchestrator targets: train.py, eval.py, paper.py
              Invoked via --stage train | eval | eval_rule_gtocr | paper | all
tests/        pytest suite organized in focus/, data/, paper/ subdirs
app/          FastAPI demo server (drag-and-drop UI)
deploy/       systemd + nginx config for production
main.py       CLI orchestrator
```

---

## Hard invariants

1. **Per-file LOC cap.** ≤166 lines in `core/`, `data/`, `models/`, `report/`,
   `stages/`, `main.py`.

2. **2-in/1-out contracts.** Every public function takes ≤2 args (typed input +
   typed config) and returns one typed value. Use `core/types.py` types.

3. **mypy-as-test.** `make check` runs `mypy --strict`. A type error is a build
   failure. Don't add `# type: ignore` without a one-line justification.

4. **`results/` is fixtures-only.** All outputs go to `runs/<run_id>/` via
   `core/runlayout.resolve_layout`. Never hard-code paths under `runs/`.

5. **Every Bug-N guard in `docs/bugs.md` must still fire.** Do not remove or
   relax guards; reroute when callers are renamed.

6. **Reproducibility.** Call `core.seed.seed_everything(config.seed)` at every
   entrypoint. DataLoaders must use seeded `worker_init_fn` + `torch.Generator`.

7. **Split persistence.** 500/63/63 split is persisted to `results/split.json`.
   Never regenerate during eval.

8. **No silent placeholders.** Every `\VAR{}` key must have a producer listed in
   `docs/TRACKING.md`.

---

## How to verify a change

```bash
make check    # ruff + mypy --strict + import smoke (≤60 s, no GPU)
make test     # pytest: split persistence, metrics, configs (no GPU)
python main.py --stage eval_rule_gtocr   # CPU-only F1 smoke, no HF Hub
make all      # full train + eval + paper; GPU required (≈75 min on RTX 4090)
```

- **`make check` and `make test` required before any PR.**
- **`make all` only for F1-affecting changes** (trainer, config, eval, pipeline).

---

## Editing guidance

- **Small, surgical diffs.** No large refactors that don't change behaviour.
- **Trainer ↔ eval parity.** Update matching eval when modifying a trainer.
- **Config changes need 3 touch-points:** `configs/default.json`, `core/config.py`,
  `README.md` Configuration table.
- **New metrics need 4 touch-points:** producer, aggregator, `\VAR{}` template,
  `docs/TRACKING.md`.
- **New dependencies:** add to `requirements.txt` AND `pyproject.toml`.

---

## What agents should NOT do

- Add new top-level directories.
- Split a module "for readability" if under 166 LOC.
- Add ML framework abstractions (trainer wrappers, config frameworks).
- Commit anything under `runs/`, `data/sroie_cache/`, or weights.
- Modify `results/bug_timeline.json`.
- Introduce unresolved `\VAR{}` keys.

---

## Where to look first

| Path | Purpose |
|---|---|
| `main.py` | CLI orchestrator |
| `core/config.py` | Config schema, validation, guards |
| `core/types.py` | Typed dataclasses |
| `stages/train.py` | Training orchestration |
| `stages/eval.py` | Evaluation, McNemar test |
| `models/donut_train.py` | DONUT fine-tuning; Bug guards |
| `models/consensus.py` | Inference-side strategies |
| `docs/TRACKING.md` | Metric producer matrix |
