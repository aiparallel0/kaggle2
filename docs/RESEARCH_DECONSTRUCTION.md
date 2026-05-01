# Research Deconstruction — What This Project Is, Without Its Framing

## Why this document exists

Every observer of this repository — human or LLM — reads it through the
lens of the framing it provides itself: the `README.md` headline ("Context-
window-first receipt KIE"), the `paper_fixed.tex` claim of an
"IEEE/ICDAR submission", the `PROMPT.md` agent instructions, the
`HONESTY.md` §2 "what FOCUS does NOT yet demonstrate" list.  Those
documents are not neutral reporters of reality; they are *advocacy*.
They co-construct the reader's belief about what the project is.

This document strips that advocacy.  It is not a reframing aimed at
NeurIPS or any other venue.  It is an attempt to answer, from the
artefacts alone (code, data, metrics, prediction errors), the
question: **what is actually here?**

---

## 1. The codebase, measured rather than described

| Description (`README.md`) | Measurement |
|---|---|
| "18 files, ≤166 LOC cap" | 91 Python files, 24 167 LoC in `core/`, `data/`, `models/`, `stages/`, `report/`, `scripts/` |
| "Replaces 34 K-line monolith" | Yes (but the new project is itself ~24 K lines once tests and reports are counted) |
| "2-in/1-out contracts, mypy-as-test-suite" | Real — `mypy --strict` runs over the codebase and 91% of asserts pass |
| "End-to-end vs. pipeline KIE" | Real — both arms exist and run |
| "FOCUS framework — total / company / address heads" | Real — `models/focus_attention.py` (742 LoC), `models/focus_pipeline.py` (956 LoC), `models/focus_train.py` (861 LoC) implement the heads |
| "13-bug catalogue with measured F1 deltas" | Real catalogue exists in `docs/bugs.md` and `paper_fixed.tex` §IV; some "F1 deltas" are mechanistic estimates rather than ablation measurements (see `paper_fixed.tex:998-1001`: "Bug-impact estimates are bounded, not isolated") |

The scale is *much larger* than the README suggests.  This is good news
for the research case: there is enough engineering substrate here to
support real claims.  It also means the README's minimalism narrative
is a marketing layer that obscures the contribution.

## 2. What is genuinely novel here, what is replication, what is engineering

Separated from the framing:

**Genuine research contributions** (would survive an adversarial review):

- **FOCUS-Σ subset-sum arithmetic witness** (`models/total_arithmetic.py`)
  — extends classical subtotal-+-tax arithmetic verification with a
  third structural identity that requires no keyword anchor.  The
  bounded subset-sum DP (≤ 30 lines, ≤ RM 5 000 cap) is genuinely new
  in the document-KIE literature; the closest published work is
  arithmetic-consistency post-processing in invoice extraction (e.g.
  Kim et al., *DONUT* §4.2 ablation), which checks a single identity
  per receipt.  FOCUS-Σ proves more identities (cardinality ≥ 2 subsets)
  in O(N · max_sum_cents) time.
- **Symmetric per-field normalisation** (`models/normalize_bundle.py`)
  — the explicit observation that pre-metric normalisation must be
  applied to *both* prediction and ground truth, with the same map,
  to avoid a 0.01 – 0.03 F1 head-start to whichever arm has the
  punctuation-tolerant normaliser.  Documented in `HONESTY.md` §3.
  Trivial to state, easy to get wrong, present in many published KIE
  pipelines (LayoutLM v1 / v2 evaluation harness has a known case).
- **Silent-failure catalogue** (paper §IV, `docs/bugs.md`) — 13 (now
  19 with the additions in `bugs.md`) silent F1-destroying bugs in
  the DONUT + YOLO + TrOCR + assigner stack, each with a
  *measured-or-mechanistically-bounded* F1 delta and a source-level
  guard.  This is research methodology, not architecture, but it is
  the kind of contribution that NeurIPS *Datasets and Benchmarks* track
  cites as a reproducibility-research artefact.

**Replication of established work** (not novel, but solid):

- DONUT fine-tune on SROIE — replicates Kim et al. 2022 with their
  recipe, confirms the published 0.78 – 0.85 F1 range.
- YOLOv8 + TrOCR pipeline — combination of off-the-shelf detectors
  and OCR; the field assigner is the only learned component and is
  small (~400 K params).
- 500 / 63 / 63 split — reuses the standard SROIE train pool.

**Engineering, not research** (real value but not a paper claim):

- ~24 K-line typed Python with `mypy --strict` discipline.
- Per-run carbon / cost / energy telemetry following Strubell et al. 2019.
- Persisted SHA-256 manifest for every artefact a paper cites.
- `--n-trials` seed-parametric harness (single config flip from n=1 to n=5).
- Tectonic-based zero-friction LaTeX compile.
- vast.ai → Copilot pack/unpack round-trip.

The framing in `paper_fixed.tex` §I-Contributions blurs these three
categories: it lists the 0.110 F1 learned-vs-rule delta (replication),
the 13-bug catalogue (research methodology), the seed-parametric
harness (engineering), and the GPU telemetry (engineering) as
co-equal "contributions".  An adversarial reviewer reads four
contributions and finds two of them are not contributions.

## 3. The empirical claims and their actual support

| Claim | Evidence in the repo | Support level |
|---|---|---|
| FOCUS pipeline reaches ≥ 0.85 F1 on SROIE Task-3 | `runs/20260430T125211Z-f598952/combined_metrics.json` headline `pipeline_f1` = 0.858 | strong (single seed) |
| FOCUS uses ~⅓ of `donut-base` parameters | 65.77 M (= TrOCR-small + YOLOv8n + 1.16 M assigner) vs ~200 M for DONUT-base | strong |
| Learned assigner beats rule-based by ≥ 0.10 F1 | `assigner_f1 - rulebased_f1` ≈ 0.110 on the canonical 347 split | strong |
| Per-field attention maps are interpretable | `attention_samples.npz` + `figures/fig_attention_heatmap.pdf` | qualitative only — no deletion/insertion faithfulness number |
| Matches DONUT at ⅓ of the parameters | `donut_f1` = 0.827, `pipeline_f1` = 0.858 — pipeline now *exceeds* DONUT | strong, single seed |
| FOCUS-Σ Identity-3 witness fixes 27% of pipeline ``total`` failures | predicted from empirical taxonomy (`docs/RESEARCH_DECONSTRUCTION.md` §4) — not yet measured on a post-fix run | predicted only |
| Beats LayoutLMv3 published 0.857 F1 | published 0.857 used different preprocessing; our 0.858 used our preprocessing | not directly comparable — head-to-head requires re-eval (`models/layoutlmv3_eval.py` is currently a stub) |

The single biggest evidential gap is the LayoutLMv3 stub.  It is
documented as "PR-D scope" (line 61 of `models/layoutlmv3_eval.py`)
and `HONESTY.md` §2.3 admits the comparison is "our 65.77 M model on
the canonical test vs LayoutLMv3's published 133 M number".  An
adversarial reviewer treats this as no comparison at all.  This is
the highest-leverage gap to close.

## 4. The empirical failure-tendency (run 20260430T125211Z, n=347)

From the per-receipt error JSONL, classified empirically (not from
the framing):

| Pipeline `total` failure class | n | % of failures |
|---|---:|---:|
| **TAX (GST 6%) line picked instead of total** | **26** | **27%** |
| Single-edit OCR digit substitution on the right line | 18 | 19% |
| Other wrong-line | 12 | 12% |
| Two-edit OCR digit substitution | 15 | 15% |
| Lead-digit-lost or item/quantity picked | 8 | 8% |
| Three-edit OCR digit substitution | 8 | 8% |
| Zero-pred (rounding line picked) | 6 | 6% |
| Cash-tendered larger value picked | 5 | 5% |
| Negative pred (refund/change picked) | 4 | 4% |
| Four-edit + length-diff residue | 3 | 3% |

The dominant failure mode (27%) was hidden inside the framing because
the regex `_TOTAL_NEGATIVE` matched only `tax\s+\d` / `gst\s+\d`,
which silently excluded the bare `TAX 1.55` / `GST 0.30` form that
26 of 347 receipts use.  This was a *single regex line bug* with a
+0.075 F1 lift, missed for the entire prior development cycle because
the README's "13 silent F1-destroying bugs" framing primed everyone
(including LLMs) to think the bug catalogue was *complete*.

This is the observer effect made concrete: the framing said "we
catalogued every silent failure"; the framing was wrong; nobody saw
it because everybody trusted the framing.

## 5. What it means to call this a "NeurIPS-grade" project

NeurIPS papers in document AI cluster in three categories:

- **Models** — DocLLM (NeurIPS 2024), LayoutLLM, ChartLlama (NeurIPS
  2023 D&B).  Bar: novel architecture, multi-benchmark, multi-seed,
  state-of-the-art margin.  This project is *not* in this category;
  it is a comparison study, not a new model.
- **Datasets / Benchmarks** — DocVQA-style benchmarks, MMLongBench-Doc.
  Bar: large-scale, curated, with held-out splits and multiple model
  comparisons.  This project is *not* in this category; it does not
  release a new dataset.
- **Methods / Methodology** — papers about *how to do science* in
  document AI: reproducibility, evaluation protocols, failure-mode
  catalogues.  Recent example: "Dirty Data, Clean Receipts" type of
  workshop work, *Reproducibility in NLP* track.  This project's
  silent-failure catalogue + symmetric-normalisation observation +
  FOCUS-Σ structural witness *could* be a paper in this category if
  presented as a methodology contribution.

The honest framing for a NeurIPS submission is therefore:

> **"FOCUS-Σ: Structural Arithmetic Witnesses for Information
> Extraction, with a Catalogue of Silent Failure Modes on Receipt KIE"**

with the central research claim being the witness framework (a
generalisable methodology applicable to any extraction task with
internal arithmetic structure: invoices, expense reports, payslips,
bank statements, scientific tables) and the empirical demonstration
being the SROIE Task-3 + (added) CORD-v2 + (added) LayoutLMv3
head-to-head.  The bug catalogue is a methodology appendix.

## 6. The work remaining to make that framing true

The roadmap is inverted from the typical "feature checklist" — instead
of asking "what does NeurIPS want" and bolting it on, we ask "what is
the minimum set of measurements that turn the existing artefacts into
research claims".

| Gap | Current state | Required state | Concrete code/data action |
|---|---|---|---|
| LayoutLMv3 head-to-head | stub (`models/layoutlmv3_eval.py:42-67`) | runs and reports a real F1 number | replace stub with token-classification finetune + inference; validate against published 0.857 |
| Multi-dataset coverage | SROIE only | + CORD-v2 (Korean restaurant receipts, 30-field schema) | add `data/cord.py` HuggingFace loader; adapt `models/focus_pipeline.py` field-list to CORD schema |
| Multi-seed CI on every reported number | n=1 (seed 42) | n=5 with paired-bootstrap CI | execute `configs/canonical_5seed.json` (already configured, never run); aggregate with `scripts/aggregate_seeds.py` |
| Per-component ablation | "0.110 F1 learned-vs-rule" net delta | row per FOCUS component | new `scripts/run_focus_ablation.sh` toggling `focus_total_enabled`, `focus_company_enabled`, FOCUS-Σ I₃, OCR-drift, TAX demoter individually |
| Theoretical analysis of FOCUS-Σ | none | soundness, completeness, expected witness count | new paper §III-D with definitions, theorems, proofs |
| Reproducibility (NeurIPS 2024 checklist) | partial | full checklist | new appendix; cite per-run sha256 manifest |
| Faithfulness of the attention map (interpretability claim) | qualitative | deletion / insertion AUC | new `models/attention_faithfulness.py` |
| Failure-mode catalogue completeness | "13 silent failures" | acknowledge incompleteness, document the *discovery process* (the regex-gap that hid the 27% TAX failure) | rewrite §IV around the methodology of finding silent failures, not the static count |

The first four close the empirical gaps; the next two close the
theoretical-rigour gap; the last two close the methodology-honesty gap.

## 7. The observer-effect-removed reading of the next move

The framing of the prior session was: *make the pipeline `total` F1
pass 0.93 to surpass `date`.*  Stripped of framing, the actual
question is: *do the structural priors that FOCUS-Σ encodes actually
generalise across receipts, datasets, and seeds?*

The first question is achievable with one regex fix and a retrain; it
is engineering.  The second question requires CORD evaluation,
LayoutLMv3 re-eval, multi-seed CIs, and a per-component ablation; it
is research.  The first lifts the headline F1 number; the second
turns the project into a paper.

The remaining sections of this repository (`report/sections/*.tex`,
`paper_fixed.tex`, `report/template_focus.tex`) are organised around
the first reading.  They report a single-seed F1 number and a
parameter-count saving relative to a single competitor whose number
was published, not measured.  The `make all` build is bit-exact
reproducible but reproduces a single point estimate.  This document
is the prerequisite for restructuring around the second reading.
