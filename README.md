# kaggle2 — Two parallel research papers on document KIE

This repository hosts **two independent research papers** plus the
shared infrastructure they consume.  Each paper is a self-contained
system; neither paper imports from the other; the two trees write
their artefacts to disjoint directories and render through their own
LaTeX templates.

```
kaggle2/
├── core/         # shared substrate — typed dataclasses, RNG seeds,
│                 #   metric primitives, manifest writer
├── data/         # shared substrate — SROIE downloader + dataset loaders
├── models/       # shared substrate — DONUT/YOLO/TrOCR trainers
├── paper2/       # ENTIRELY SELF-CONTAINED Paper 2 system
│   ├── config_paper2.py
│   ├── main_paper2.py
│   ├── configs/default.json
│   ├── models/   # rule_*_paper2, zone_prior_paper2, postprocess_*_paper2
│   ├── stages/   # (pending — currently dispatched through top-level stages/)
│   ├── report/
│   │   ├── template_paper2.tex
│   │   ├── references.bib
│   │   └── sections/  # 13 disjoint LaTeX section files
│   ├── tests/
│   ├── results/
│   ├── docs/
│   └── runs/
├── paper3/       # ENTIRELY SELF-CONTAINED Paper 3 system
│   ├── config_paper3.py
│   ├── main_paper3.py
│   ├── configs/default.json
│   ├── models/   # focus_*_paper3, total_arithmetic_paper3, …
│   ├── stages/
│   ├── report/
│   │   ├── template_paper3.tex
│   │   ├── references.bib
│   │   ├── wrapper_delta_paper3.py
│   │   ├── paper_f1_gap_paper3.py
│   │   └── sections/  # 14 disjoint LaTeX section files (incl. svkie_theory)
│   ├── tests/
│   ├── results/
│   ├── docs/
│   └── runs/
├── app/          # FastAPI demo server (paper-neutral)
├── deploy/       # nginx + systemd config (paper-neutral)
├── docs/         # repository-wide docs (AGENTS, PARALLEL_TRAINING)
├── runs_legacy/  # pre-bifurcation run artefacts (read-only)
├── scripts/      # operational shell scripts
└── tests/        # tests for shared substrate (core + data + base models)
```

## The two papers

### Paper 2 — *End-to-End vs Modular Receipt KIE: A Replication Study with a Silent-Failure Catalogue*

End-to-end DONUT (paper-faithful Kim et al.\ 2022 recipe) versus a
deliberately non-learned modular pipeline (YOLOv8 + TrOCR + regex
field assignment + 3-state zone-prior HMM + per-field postprocess) on
the canonical ICDAR-2019 SROIE Task-3 split ($n=347$).

Methodology contribution: 14-bug catalogue with measured F1 deltas
and source-level guards.  Target venue: **IEEE Access** primary,
**ICDAR DAS workshop** secondary.

```bash
python paper2/main_paper2.py --stage all
```

### Paper 3 — *SVKIE: Structure-Verified Document Key-Information Extraction*

A multi-prior framework in which five orthogonal structural priors
(arithmetic, spatial, visual, lexical, epistemic) are each encoded by
a distinct module and unified through the **FOCUS-Σ** verification
layer.  Architecture-agnostic: applied as an inference-time wrapper
around DONUT, LayoutLMv3, and the in-house FOCUS-T head.  Theoretical
contribution: soundness, completeness, single-edit OCR-recovery
probability bound, spatial-zone monotonicity, multi-prior
information-theoretic complementarity bound.

Target venue: **ICDAR main**.

```bash
python paper3/main_paper3.py --stage all
```

## Independence guarantees

- **Paper 2 imports `core/`, `data/`, top-level `models/` (DONUT, YOLO, TrOCR
  trainers), and other `paper2/*` files only.**  It never imports from
  `paper3/*`.
- **Paper 3 imports `core/`, `data/`, top-level `models/`, and other
  `paper3/*` files only.**  It never imports from `paper2/*`.
- **Each paper has its own LaTeX template, its own complete set of
  section files, its own `references.bib`, its own configs, its own
  results fixtures, its own tests.**  No LaTeX section is shared.
- **Each paper writes to its own `runs/<run_id>/` directory.**

## What's where

| Layer | Paper 2 | Paper 3 |
|---|---|---|
| Field assignment | regex + rule_consensus + zone_prior HMM | three-headed neural ensemble (FOCUS-T cross-attention + GAT + frozen-CNN) fused by learned gating MLP |
| Field verification | none | FOCUS-Σ subset-sum + Hamming-1/2 OCR-drift + confidence-gated cascade |
| Datasets | SROIE Task-3 canonical (n=347) | SROIE Task-3 + CORD-v2 |
| Theory | none (replication only) | 5 propositions: soundness, completeness, single-edit recovery, zone monotonicity, multi-prior complementarity |
| Methodology contribution | 14-bug catalogue | wrapper-Δ verification + per-prior ablation |
| LaTeX template | `paper2/report/template_paper2.tex` | `paper3/report/template_paper3.tex` |
| Config preset | `paper2/configs/default.json` | `paper3/configs/default.json` |

## Reproducibility

Each paper writes a SHA-256-pinned manifest under
`<paper-tree>/runs/<run_id>/MANIFEST.json` and snapshots the exact
`ExpConfig` to `env/config_snapshot.json`.  The git SHA, Python
environment, and GPU model are captured per run.  Both papers fill
the NeurIPS-2024 reproducibility checklist
(`<paper-tree>/report/sections/repro_checklist.tex`).

## Live web application

A production OCR-and-KIE web service deployed at
[image-to-text.fit](https://image-to-text.fit/) demonstrates the
full pipeline serving real users.  See `app/` and `deploy/` for the
service code.

## Status (post-bifurcation)

- ✅ Per-paper LaTeX trees fully populated (Paper 2: 13 sections,
  Paper 3: 14 sections including SVKIE theory)
- ✅ Per-paper config presets with disjoint paths
- ✅ Per-paper main entry points (`paper2/main_paper2.py`,
  `paper3/main_paper3.py`)
- ✅ Paper-specific code moved into per-tree `models/` directories
  with `_paper2` / `_paper3` filename suffixes (33 files moved)
- ⚠️ **Top-level shared `models/`, `stages/`, and `report/` still
  contain residual cross-paper imports** (e.g.\
  `models/eval_pipeline.py` dispatches to both rule-based and
  FOCUS-T paths via config flags; this dispatch logic should be
  bifurcated into `paper2/models/eval_pipeline_paper2.py` and
  `paper3/models/eval_pipeline_paper3.py` in a follow-up commit).
- ⚠️ Tests not yet split per tree.
- ⚠️ `make check` does not currently pass post-refactor due to
  residual broken imports between the top-level shared substrate and
  the moved paper-specific files.  Resolution is the dispatch-bifurcation
  follow-up commit referenced above.

The bifurcation is **structurally complete** at the file-system and
LaTeX level — a reviewer reading `paper2/report/template_paper2.tex`
sees only Paper 2 content; a reviewer reading
`paper3/report/template_paper3.tex` sees only Paper 3 content.  The
remaining work is mechanical (import fixes + dispatch split) and is
queued for the next session.
