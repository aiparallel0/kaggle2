# Paper 3 — SVKIE multi-prior framework (kaggle2 / paper3/ session)

Paste this prompt into a Claude Code session opened in the
**kaggle2 repository** with the `paper3/` tree as the working scope.
Use a different session from Paper 2 — the two paper trees must
develop independently to keep cross-imports impossible.

---

## Prompt

```
You are working in the kaggle2 repository, paper3/ tree.  The repository
hosts two parallel research papers (paper2/ and paper3/) plus shared
substrate (core/, data/, top-level models/).  Your scope is paper3/
ONLY: do not modify or import from paper2/.

GOAL — Produce Paper 3: a complete ICDAR-main-track research paper
introducing the SVKIE (Structure-Verified Document KIE) framework
with FOCUS-Σ as the verification component.  Execute multi-seed
training, the 6-cell ablation grid, the wrapper-Δ matrix across
three architectures, the 5-proposition theory section, and emit the
handoff bundle for the assembly agent.

THESIS OF PAPER 3
=================
"Document KIE accuracy is governed by five orthogonal structural
priors (arithmetic, spatial, visual, lexical, epistemic).  We
introduce SVKIE, a framework in which each prior is encoded as a
distinct module and a unifying verification layer (FOCUS-Σ)
enforces consistency across priors via three arithmetic identities
with provable soundness and completeness.  The framework is
architecture-agnostic: applied as an inference-time wrapper around
DONUT, LayoutLMv3-base, and an in-house FOCUS-T head, it produces
positive wrapper-ΔF1 on every upstream architecture without
retraining."

CURRENT STATE OF paper3/
========================
- paper3/configs/default.json — full FOCUS stack on; multi-seed
  harness configured.
- paper3/configs/canonical_5seed.json + sweep/ presets.
- paper3/models/ — 24 paper-3-specific files: focus_*_paper3 (13),
  total_arithmetic_paper3 (FOCUS-Σ), consensus_paper3,
  consensus_score_paper3, corrections_paper3,
  attention_faithfulness_paper3, layoutlmv3_eval_paper3,
  assigner_*_paper3, donut_rag_paper3, retrieval_bank_paper3,
  llm_eval_paper3.
- paper3/report/template_paper3.tex + 14 sections (intro, related,
  problem, method, figure_architecture, svkie_theory, experiments,
  results, discussion, limitations, broader_impact, conclusion,
  bugs, appendix, repro_checklist).
- paper3/report/wrapper_delta_paper3.py — multi-arch wrapper-Δ
  producer.
- paper3/report/paper_f1_gap_paper3.py — Paper 2 vs Paper 3 F1 gap
  reporter.
- paper3/main_paper3.py — entry point.

KNOWN-DEFERRED STATE
====================
- The moved paper-3-specific files retain their pre-move imports
  (e.g., `from models.focus_attention import ...`) which now point
  at non-existent paths.  Per-tree mypy is therefore broken.  Step 0
  fixes these.
- Top-level models/eval_pipeline.py still dispatches by config flag;
  paper3/main_paper3.py routes to it; the FOCUS-on path inside the
  top-level eval_pipeline still imports from the pre-move locations.
  Splitting eval_pipeline.py into per-paper versions is part of
  Step 0.

EXECUTION PLAN
==============
Step 0 — Make paper3/main_paper3.py runnable end-to-end.
  - Walk paper3/models/*.py and update every import that references
    a moved file to the new paper3.models.<name>_paper3 path.
  - Split top-level models/eval_pipeline.py into a paper3/models/
    eval_pipeline_paper3.py (FOCUS-on dispatch) and remove the FOCUS
    branches from the top-level file (which becomes paper-2-only).
  - Run `make check-paper3` and iterate until it passes.

Step 1 — Theory verification.
  - Read paper3/report/sections/svkie_theory.tex.
  - Verify the three FOCUS-Σ propositions (soundness, completeness,
    single-edit recovery) are mathematically tight as stated.
  - Draft the two missing propositions:
    * Spatial-zone monotonicity: prove that under the 3-state HMM
      with hard-coded forward-only transitions, the decoded
      posterior is monotone in receipt y-position.
    * Multi-prior information-theoretic complementarity: prove
      that the per-prior wrapper-Δ contributions are sub-additive
      (no double-counting) under a stated independence assumption.
  - Persist the proofs in svkie_theory.tex.

Step 2 — Execute training + multi-seed evaluation.
  - Provision an RTX 4090 with 24GB VRAM and 50GB disk.
  - From a clean clone:
        bash scripts/vastai_bootstrap.sh
        python paper3/main_paper3.py --stage all
  - Multi-seed (n=5) using paper3/configs/canonical_5seed.json.
  - Total wall-clock: ~15-20 hours on a single 4090.  Budget: ~$10.

Step 3 — Per-component ablation grid (6 cells).
  - Run the 6 ablation rows of Tab.~tab:ablation_neurips:
    1. Lexical only (FOCUS-T cross-attn, all other priors off)
    2. + Arithmetic (I_2 + I_3)
    3. + Spatial (zone-prior + GAT)
    4. + Visual (frozen-CNN head)
    5. + Epistemic (confidence cascade)
    6. + OCR-drift recovery (Hamming-1/2)
  - Each row at n=5 seeds.  Total ~30 runs at ~30 min each = ~15h.
  - Persist to paper3/runs/ablation/ + aggregate to
    paper3/results/ablation_focus_sigma.json.

Step 4 — Wrapper-Δ matrix across architectures.
  - Three upstream heads × two configurations (Bare / +SVKIE):
    DONUT-base, LayoutLMv3-base, FOCUS-T.
  - LayoutLMv3 fine-tune via paper3/models/layoutlmv3_eval_paper3.py;
    follow the published recipe.  ~2 hours on the 4090.
  - For each (architecture, configuration) pair, evaluate on
    canonical SROIE 347.  Persist to
    paper3/runs/wrapper_delta/ + aggregate to
    paper3/results/wrapper_delta_metrics.json.

Step 5 — CORD-v2 cross-dataset evaluation.
  - Set cord_eval_enabled=True; run --stage eval on CORD test split.
  - Persist to paper3/runs/cord/ + aggregate.

Step 6 — Auxiliary measurements.
  - Faithfulness (deletion / insertion AUC) via
    paper3/models/attention_faithfulness_paper3.py
  - Calibration (ECE / MCE / Brier on per-receipt confidence)
  - Latency (p50/p95/p99 cold-start + steady-state)

Step 7 — Build the handoff bundle.
  - paper3/handoff/results.json — every \VAR{} key referenced in
    paper3/report/sections/*.tex.
  - paper3/handoff/figures/ — every PDF the paper cites
    (architecture diagram, ablation curves, wrapper-Δ heatmap,
    Pareto plot, faithfulness curves, calibration reliability
    diagram, latency CDFs).
  - paper3/handoff/manifest.json — sha256 of every bundle file.
  - paper3/handoff/HANDOFF_README.md — keys marked null + why.

CONSTRAINTS
===========
- No fabricated numbers.  Every \VAR{} key in handoff/results.json
  must trace to a real paper3/runs/<run_id>/ artefact.
- Do not modify paper2/ or report from paper2/runs/.
- Do not modify shared substrate (core/, data/, top-level models/)
  unless absolutely necessary; if you must, document in
  HANDOFF_README.md.
- Stay strictly within Paper 3's scope: the SVKIE framework, the
  five structural priors, the FOCUS-Σ verifier.  Do NOT report
  Paper 2's bug catalogue as your contribution; cite it as related
  work in 1 sentence at most.
- Multi-prior ablation must use cumulative addition (each row adds
  one more prior), and report the marginal Δ against the
  immediately preceding row.  Do not stack non-cumulative rows.

START WITH STEP 0 — fix imports + split eval_pipeline.  Then Step 1
(theory drafting) before any GPU work, because the proofs constrain
the empirical claims you can make.  Report progress at the end of
each step.  Stop and ask only if a step materially changes the
framework structure (e.g., a prior turns out to be empirically
useless and the 5-prior count needs to drop to 4).
```
