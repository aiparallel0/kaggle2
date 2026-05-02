# Paper 4 — Compressed SVKIE (new paper-4 repo session)

Paste this prompt into a Claude Code session opened in an **empty
working directory** that will become the paper-4 repository.

---

## Prompt

```
You are creating a new repository for Paper 4 — a compression study
on the SVKIE (Structure-Verified Document KIE) system from Paper 3
of the same research programme.

GOAL — Produce a complete, ready-to-train repository plus an
IEEE-format paper template framed as future work.  This repo is a
SKELETON: no empirical results yet.  Results will be filled in a
later GPU run.  Your output must be a working repo + a paper that
reads as complete future-work scoped, not as a template-with-blanks.

THESIS OF PAPER 4
=================
"Compression Crossover for Document KIE: applying parameter-efficient
techniques (int8 / int4 quantization, structured pruning, half-depth
distillation) to the SVKIE pipeline.  Pareto frontier of macro-F1
versus on-device memory footprint (MB).  Hypothesis: compression
can place a structurally-verified end-to-end system at the same
memory footprint as the modular pipeline baseline, closing the
deployment gap without abandoning structural verification."

CONSTRAINTS
===========
- This paper has NO empirical results in this PR.  All numerical
  claims in the manuscript must be marked clearly as "future work"
  or "expected results" with explicit \emph{(to be measured on
  next vast.ai run)} qualifiers.  Do NOT fabricate F1 cells.
- The paper must read as a complete, standalone contribution on
  the efficiency axis.  Use full research-paper language with
  future-tense empirical claims where appropriate.
- The repository must be ready to run: Pruna OSS integration
  scaffolded, quantization configs prepared, evaluation harness
  mirroring the SVKIE evaluation pipeline.
- Format: IEEE Access conference template (or NeurIPS workshop
  style if pivoting to ENLSP / ES-FoMo).

REPOSITORY STRUCTURE TO PRODUCE
==============================
paper4/
├── core/                     # symlinked or copied from kaggle2/core/
├── data/                     # symlinked or copied from kaggle2/data/
├── models/
│   ├── compression_paper4.py # Pruna integration entry points
│   ├── quantize_paper4.py    # int8 / int4 wrappers
│   ├── prune_paper4.py       # structured pruning wrappers
│   ├── distill_paper4.py     # half-depth distillation harness
│   └── eval_paper4.py        # SVKIE eval over compressed models
├── stages/
│   ├── compress_paper4.py    # the compression sweep orchestrator
│   ├── eval_paper4.py
│   └── paper_paper4.py
├── configs/
│   ├── default.json
│   └── sweep/
│       ├── int8.json
│       ├── int4.json
│       ├── prune25.json
│       ├── prune50.json
│       ├── distill_half.json
│       └── int8_prune25.json
├── report/
│   ├── template_paper4.tex
│   ├── references.bib
│   └── sections/             # 13--14 disjoint LaTeX sections
├── scripts/
│   ├── run_compression_sweep.sh
│   └── vastai_bootstrap.sh
├── tests/                    # Pruna config validation, no GPU needed
├── results/
├── runs/
├── main_paper4.py
├── Makefile
└── README.md

ARCHITECTURAL CONSIDERATIONS (read carefully)
============================================
1. The Pareto plot's x-axis MUST be ON-DEVICE MEMORY FOOTPRINT (MB),
   NOT parameter count.  Quantization changes bytes-per-parameter,
   not parameter count; using parameter count would be conceptually
   wrong and a reviewer would flag it immediately.  Add a secondary
   "params" column to the table for context.

2. The compression must preserve the FOCUS-Σ verifier's correctness.
   The verifier is a rule-based bounded-DP, not a neural network, so
   quantization does NOT affect it.  The compression target is the
   upstream cross-attention assigner + DONUT/LayoutLMv3 backbones.
   A dedicated "compression-aware verification" section must argue
   that the structural verifier acts as a regularizer against
   catastrophic compression failure: if the verifier rejects a
   candidate post-compression, fall back to the rule-based path.

3. The hypothesis in the abstract must be FALSIFIABLE.  Define
   crossover precisely upfront:
       "A compression cell crosses iff:
        (effective-memory ≤ pipeline-baseline-MB) AND
        (mean F1 across n=5 seeds ≥ pipeline-F1 − bootstrap-CI-half-width)."
   Whichever way the data falls when measured, the paper writes
   itself.

DELIVERABLES
============
1. Working paper4/ repository (as above) that compiles
   `python paper4/main_paper4.py --stage paper` end-to-end on a
   fresh clone WITHOUT any GPU run, producing a future-work-framed
   PDF.
2. handoff/ bundle:
     - results.json — flat dict; every empirical key is null with
       a note in HANDOFF_README.md::unmeasured.
     - figures/ — placeholder SVG/PDF figures for the Pareto plot,
       memory-footprint chart, etc., with captions but synthetic
       data marked clearly.
     - sections_diff.md
     - manifest.json
     - HANDOFF_README.md
3. README.md with the run-this-on-vastai instructions for when the
   compression sweep is executed:
       bash scripts/vastai_bootstrap.sh
       python paper4/main_paper4.py --stage all
4. A note in the paper's roadmap section about the data dependency
   on Paper 3's trained checkpoint (paper4 cannot execute its
   compression sweep until kaggle2/paper3/ has produced a trained
   SVKIE checkpoint).

WORKFLOW
========
Step 1 — Repository scaffolding.  Create the directory structure
above with empty/stub files.  Add Makefile, README, .gitignore.

Step 2 — Pruna OSS integration scaffold.  Write models/quantize_paper4.py,
models/prune_paper4.py, models/distill_paper4.py with the API
surface to call Pruna.  Verify with `pip install pruna` import smoke.

Step 3 — Eval harness.  Write models/eval_paper4.py that consumes a
compressed model + the SVKIE pipeline path from kaggle2/paper3/ and
emits the same metrics shape paper3 emits.

Step 4 — LaTeX template + sections.  Write each section (intro,
related, problem, method, experiments, results, discussion,
limitations, broader_impact, conclusion, appendix, repro_checklist).
Mark every empirical claim as future work; write the prose so the
paper is publishable on the day the GPU run completes.

Step 5 — Build the handoff bundle.  Mark every empirical \VAR{} key
null in results.json with a clear unmeasured-list in
HANDOFF_README.md.  The assembler will render null cells as
\MissingCell{key}.

CONSTRAINTS
===========
- No fabricated numbers.  Future-work claims are explicit.
- The repo must NOT depend on kaggle2 source code at runtime; it
  may COPY relevant pieces (core types, eval harness) at scaffold
  time, but should be self-contained for a future GPU run.
- The Pareto x-axis is memory footprint, not parameter count.

START WITH STEP 1 — repository scaffolding.  Show the directory
structure first; wait for confirmation before populating Pruna
integration.  This is a scaffolding job, not a measurement job.
```
