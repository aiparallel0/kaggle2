# Paper 2 vs Paper 3 — bifurcation contract

This file is the **single source of truth** for what belongs to Paper 2
vs Paper 3.  Every architectural component, every config knob, every
section file, and every test artefact is assigned to exactly one of
the two papers (or to the shared substrate, which both reference but
neither claims as a contribution).

> **Why this file exists.**  Paper 2 and Paper 3 share a codebase, a
> dataset, and a DONUT baseline.  Without an explicit bifurcation
> contract, prose drift, copied tables, or accidental claims would
> erode the disjointness that prevents self-plagiarism at submission
> time.  This file is consulted before every paper build; the audit
> script `scripts/audit_paper_overlap.sh` enforces it mechanically.

## TL;DR

| Layer                  | Paper 2 (replication study) | Paper 3 (architectural extensions) |
|------------------------|-----------------------------|------------------------------------|
| **Field assignment**   | Regex + rule_consensus + zone_prior HMM | **Three-headed neural ensemble:** FOCUS-T cross-attention + GAT over line-graph + frozen-CNN visual head, fused by a learned gating MLP |
| **Field verification** | None | FOCUS-Σ subset-sum + Hamming-1/2 OCR-drift recovery + confidence-gated cascade |
| **Datasets**           | SROIE Task-3 canonical 347 only | SROIE Task-3 + CORD-v2 |
| **Theory**             | None | Soundness, completeness, single-edit recovery proofs |
| **Headline metric**    | F1 / NED / EM | F1 + ΔF1 + paired-bootstrap CI + McNemar p + ECE/MCE/Brier |
| **Methodology contribution** | 14-bug catalogue (Bugs 1–14) | wrapper-Δ verification + ensemble decomposition |
| **Target venue**       | IEEE Access / IJDAR / ICDAR DAS | ICDAR main / IJDAR / NeurIPS workshop |
| **LaTeX template**     | `report/template_paper2.tex` | `report/template_focus.tex` or `report/template_neurips.tex` |
| **Config preset**      | `configs/paper2.json` | `configs/default.json` (advanced focus) or `configs/canonical_5seed.json` |

## Per-component assignment

| Module / file | Paper 2 | Paper 3 | Shared substrate |
|---|:---:|:---:|:---:|
| `models/donut_train.py` |  |  | ✓ |
| `models/donut_eval.py`  |  |  | ✓ |
| `models/yolo_train.py`  |  |  | ✓ |
| `models/trocr_train.py` |  |  | ✓ |
| `models/rule_regex.py`     | ✓ |  |   |
| `models/rule_consensus.py` | ✓ |  |   |
| `models/rule_eval.py`      | ✓ |  |   |
| `models/rule_fields.py`    | ✓ |  |   |
| `models/zone_prior.py`     | ✓ | ✓ |   |
| `data/zone_prior_fit.py`   | ✓ | ✓ |   |
| `models/total_post.py`     |   | ✓ |   |
| `models/postprocess_address.py` | ✓ | ✓ |   |
| `models/postprocess_company.py` | ✓ | ✓ |   |
| `models/normalize_bundle.py`    |   |   | ✓ |
| `models/focus_attention.py`     |   | ✓ |   |
| `models/focus_pipeline.py`      |   | ✓ |   |
| `models/focus_train.py`         |   | ✓ |   |
| `models/focus_inference.py`     |   | ✓ |   |
| `models/focus_gat.py`           |   | ✓ |   |
| `models/focus_cnn.py`           |   | ✓ |   |
| `models/focus_ensemble.py`      |   | ✓ |   |
| `models/focus_addr_penalty.py`  |   | ✓ |   |
| `models/focus_priors.py`        |   | ✓ |   |
| `models/total_arithmetic.py` (FOCUS-Σ) |   | ✓ |   |
| `models/consensus.py` (FOCUS-Σ) |   | ✓ |   |
| `models/corrections.py` (Hamming OCR-drift) |   | ✓ |   |
| `models/attention_faithfulness.py` |   | ✓ |   |
| `models/layoutlmv3_eval.py`     |   | ✓ |   |
| `data/cord.py`                  |   | ✓ |   |
| `core/error_metrics.py`         |   |   | ✓ |
| `core/extra_stats.py` (calibration) |   | ✓ |   |
| `core/latency_metrics.py`       |   | ✓ |   |
| `report/template_paper2.tex`    | ✓ |   |   |
| `report/sections/intro_paper2.tex`   | ✓ |   |   |
| `report/sections/results_paper2.tex` | ✓ |   |   |
| `report/template_neurips.tex`   |   | ✓ |   |
| `report/sections/experiments_neurips.tex` |   | ✓ |   |
| `report/sections/results_neurips.tex`     |   | ✓ |   |
| `report/sections/focus_sigma_theory.tex`  |   | ✓ |   |
| `report/sections/intro_neurips.tex`       |   | ✓ |   |
| `report/wrapper_delta.py`       |   | ✓ |   |
| `report/paper_f1_gap.py`        |   | ✓ |   |
| `docs/bugs.md` (Bugs 1–14)      | ✓ |   |   |
| `docs/bugs.md` (Bugs 18, 19)    |   | ✓ |   |
| `docs/RESEARCH_DECONSTRUCTION.md` |   | ✓ |   |
| `docs/HONESTY.md`               |   |   | ✓ |
| `docs/TRACKING.md`              |   |   | ✓ |

## Architectural distance — the principal contribution boundary

Paper 2's pipeline reads OCR text into **rule-based** field assignment:
regex matching, rule consensus across regex variants, a 3-state HMM
zone prior over header / items / totals.  No neural component
participates in the field-assignment decision.

Paper 3's pipeline replaces rule-based assignment with a
**three-headed neural ensemble**:

| Head | File | Modality | Trainable params | Pre-trained weights |
|---|---|---|---:|---|
| H1 — cross-attention | `models/focus_attention.py` | Text + spatial priors | ~400K | none (trained on SROIE train fold) |
| H2 — graph-attention (GAT) | `models/focus_gat.py` | Text embeddings + kNN graph over bbox centres | ~120K | none |
| H3 — visual (CNN) | `models/focus_cnn.py` | Frozen ResNet-18 over line bbox crops | ~50K (projection only) | ImageNet (frozen) |
| Gate | `models/focus_ensemble.py` | 2-layer MLP fusing H1+H2+H3 outputs + zone-prior summary | ~3K | none |

Heads H2 and H3 are the architectural differentiators introduced
specifically for Paper 3.  H1 is the existing FOCUS-T head; H2 is an
existing GAT module that has been off-by-default and is now turned on
as Paper 3's second head; H3 is new (`models/focus_cnn.py`).  The
gating MLP is also new (`models/focus_ensemble.py`).

After ensemble assignment, the FOCUS-Σ subset-sum verifier
(`models/total_arithmetic.py`, `models/consensus.py`) checks the
candidate value for the `total` field against three arithmetic
identities ($\mathrm{I}_1, \mathrm{I}_2, \mathrm{I}_3$) and applies
Hamming-1 / Hamming-2 OCR-drift recovery
(`models/corrections.py`).  None of these are present in Paper 2.

## Anti-overlap policy

The two papers will share their description of the **shared
substrate** (DONUT recipe, YOLO+TrOCR detect-and-read upstream,
symmetric normaliser, manifest pipeline, reproducibility infrastructure).
The shared substrate is described in **at most one paragraph per
paper**, with both paragraphs differing in word choice and ordering.
The single allowed verbatim element is the BibTeX entry for the
shared dataset / public model checkpoints.

| Surface | Allowed shared content | Forbidden shared content |
|---|---|---|
| Abstract | None — written from scratch per paper | Any sentence verbatim |
| Introduction | None — written from scratch per paper | Contributions list elements |
| Related Work | Up to 3 shared paragraphs of citation prose | More than 3 sentences verbatim |
| Method | The shared substrate paragraph (1× per paper) | Method-specific contribution prose |
| Experiments | Test split / hardware sentences | Experimental design specifics |
| Results | None — different headline tables, different figures | Any cell value |
| Discussion / Limitations / Conclusion | None — written from scratch per paper | Reframings of the same sentence |
| Bibliography | All entries cited in either paper | (no policy — bib entries are tools, not prose) |

The audit script `scripts/audit_paper_overlap.sh` computes a rolling
5-word shingle hash over the prose of both papers' compiled `.tex`
sources and reports any shingle that appears in both papers more than
once.  CI fails when shared-shingle ratio exceeds 5 % of the smaller
paper's word count.

## Headline F1 gap contract

The F1 gap between Paper 2's pipeline arm and Paper 3's pipeline arm
is the **single most important number** for both papers — Paper 2
cites it as the engineering ceiling it does not aim to cross; Paper 3
cites it as the empirical contribution of its architectural extensions.
The gap is computed by `report/paper_f1_gap.py`, which reads the two
runs' `combined_metrics.json::pipeline_f1` and writes the comparison
to `runs/<paper3_run_id>/metrics/paper_f1_gap.json`.

Both papers' results sections cite the gap via the shared `\VAR{}`
keys `paper2_pipeline_f1`, `paper3_pipeline_f1`, and
`paper2_paper3_f1_gap`.  Citing the gap from a single source guarantees
the two papers cannot disagree on the contribution magnitude.

## Promotion criteria

A paper graduates from "draft" to "submission-ready" when:

**Paper 2:**
1. `make check` passes with `configs/paper2.json` as the live config.
2. `pipeline_f1` ≥ 0.78 on the canonical 347 test split (engineering ceiling for a non-learned pipeline).
3. DONUT `f1` ≥ 0.83 under the paper-faithful recipe.
4. Bug catalogue Bugs 1–14 have measured F1 deltas in the live run, not fixture-only.
5. Anti-overlap audit score ≤ 5 %.

**Paper 3:**
1. `make check` passes with the advanced focus config + `focus_ensemble_enabled=true`.
2. `pipeline_f1` ≥ `paper2_pipeline_f1 + 0.05` (substantial gap, not noise).
3. Multi-seed (n=5) paired-bootstrap CI on $\Delta$F1 strictly positive at 95 %.
4. Wrapper-$\Delta$ matrix populated for all three architectures (DONUT, LayoutLMv3, FOCUS-T).
5. Anti-overlap audit score ≤ 5 %.
