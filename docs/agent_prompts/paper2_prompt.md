# Paper 2 — Replication study + bug catalogue (kaggle2 / paper2/ session)

Paste this prompt into a Claude Code session opened in the
**kaggle2 repository** with the `paper2/` tree as the working scope.

---

## Prompt

```
You are working in the kaggle2 repository, paper2/ tree.  The repository
hosts two parallel research papers (paper2/ and paper3/) plus shared
substrate (core/, data/, top-level models/).  Your scope is paper2/
ONLY: do not modify or import from paper3/.

GOAL — Produce Paper 2: a complete IEEE-Access-format research paper
on end-to-end vs modular receipt KIE, with a 14-bug catalogue as the
methodology contribution.  Execute the multi-seed evaluation, populate
the bug atlas with measured F1 deltas, and emit the handoff bundle for
the assembly agent.

THESIS OF PAPER 2
=================
"End-to-end DONUT (paper-faithful Kim et al. 2022 recipe) versus a
deliberately non-learned modular pipeline (YOLOv8 + TrOCR + regex
field assignment + 3-state header/items/totals zone-prior HMM +
per-field deterministic post-processing) on the canonical ICDAR-2019
SROIE Task-3 split (n=347).  Methodology contribution: 14-bug
catalogue with measured F1 deltas and source-level guards."

CURRENT STATE OF paper2/
========================
- paper2/configs/default.json — pinned to FOCUS-disabled, regex+
  zone-prior on, canonical SROIE 347.
- paper2/models/ — 9 paper-2-specific files: rule_*_paper2,
  zone_prior_paper2, postprocess_*_paper2, total_post_paper2,
  date_post_paper2.
- paper2/report/template_paper2.tex + 13 sections (intro, related,
  problem, method, experiments, results, discussion, limitations,
  broader_impact, conclusion, bugs, appendix, repro_checklist).
- paper2/main_paper2.py — entry point; routes to shared stage
  orchestrators which currently still dispatch by config flag.

KNOWN-DEFERRED STATE (per README.md)
====================================
- Top-level models/eval_pipeline.py and stages/{train,eval,paper}.py
  still dispatch by config flag.  paper2/main_paper2.py routes to
  them; this works for Paper 2 because the paper2 config has all
  FOCUS flags off, so the dispatch falls into the rule-based path.
- The moved paper-2-specific files retain their pre-move imports
  (e.g., `from models.rule_regex import ...`) which now point at
  non-existent paths.  Per-tree mypy is therefore broken.  You may
  need to fix these imports as part of Step 0.

EXECUTION PLAN
==============
Step 0 — Make paper2/main_paper2.py runnable end-to-end.
  - Walk paper2/models/*.py and update every import that references
    a moved file to use the new paper2.models.<name>_paper2 path.
  - Run `make check-paper2` (in Makefile, currently expected to fail)
    and iterate until it passes.

Step 1 — Execute training + evaluation on a vast.ai 4090 instance.
  - Provision an RTX 4090 with at least 24GB VRAM and 50GB disk.
  - From a clean clone:
        bash scripts/vastai_bootstrap.sh
        python paper2/main_paper2.py --stage all
  - This produces paper2/runs/<run_id>/ with:
        metrics/combined_metrics.json
        metrics/extended_metrics.json
        cost_*.json
        env/{git_sha.txt,config_snapshot.json,nvidia_smi.txt,...}
        MANIFEST.json
  - Total wall-clock: ~6-8 hours on a single 4090.  Budget: ~$3.

Step 2 — Multi-seed sweep.
  - Edit paper2/configs/default.json::seeds to [42, 1, 2, 3, 5] and
    n_trials to 5.  Re-run --stage eval for each seed (or use the
    existing scripts/run_5seed_sweep.sh adapted for paper2).
  - Aggregate via scripts/aggregate_seeds.py (adapted for paper2/runs/).
  - Produces paper2/runs/aggregate.csv + aggregate.json with per-key
    mean / std / paired-bootstrap 95% CI.

Step 3 — Bug-atlas measurements.
  - For each Bug-1..Bug-14 entry in docs/bugs.md, toggle the bug-flag
    OFF in the config and re-run --stage eval.  Record the F1 delta
    against the all-on headline.
  - Persist measurements to paper2/results/bug_atlas.json.

Step 4 — Build the handoff bundle.
  - paper2/handoff/results.json — flat dict feeding \VAR{} injection.
    Keys must include every \VAR{} referenced in
    paper2/report/sections/*.tex (run report/check_artefacts to
    enumerate).
  - paper2/handoff/figures/ — every PDF the paper cites.
  - paper2/handoff/sections_diff.md — any prose tweaks made.
  - paper2/handoff/manifest.json — sha256 of every bundle file.
  - paper2/handoff/HANDOFF_README.md — what the assembler should know,
    including any keys marked null and why.

CONSTRAINTS
===========
- No fabricated numbers.  Every \VAR{} key in handoff/results.json
  must trace to a real paper2/runs/<run_id>/ artefact.
- Do not modify paper3/ or report from paper3/runs/.
- Do not modify shared substrate (core/, data/, top-level models/)
  unless absolutely necessary; if you must, document the change in
  HANDOFF_README.md.
- Stay strictly within Paper 2's scope: NO learned cross-attention
  assignment, NO FOCUS-Σ verifier, NO ensemble heads, NO theory
  propositions.  Those belong to Paper 3.
- Multi-seed harness uses paired-bootstrap CIs, not arithmetic mean
  of per-seed CIs.

START WITH STEP 0 — fix the broken imports in paper2/models/*.py and
get make check-paper2 passing.  Report progress at the end of each
step.  Stop and ask only if a step fails in a way that materially
changes the bug catalogue (e.g., a Bug-N guard cannot be ablated
because the pre-fix path no longer exists).
```
