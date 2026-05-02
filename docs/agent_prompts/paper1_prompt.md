# Paper 1 — Multi-dataset DONUT training (Donut-web-app session)

Paste this prompt into a Claude Code session opened in the
**Donut-web-app repository** (the production OCR-and-KIE web service
at `image-to-text.fit` plus the legacy DONUT training artefacts).

---

## Prompt

```
You are working in the Donut-web-app repository.  This repo holds the
production OCR-and-KIE web service deployed at https://image-to-text.fit/
plus the legacy research artefacts (run logs, training PDFs, evaluation
outputs) from a multi-dataset DONUT training programme conducted earlier
in the project lifetime.

GOAL — Produce Paper 1: a complete, IEEE-Access-format research paper
on data-axis improvements to DONUT for receipt KIE, by re-rendering
the existing legacy artefacts into a modern LaTeX manuscript.

THESIS OF PAPER 1
=================
"Multi-dataset training of DONUT for receipt key-information
extraction: demonstrating that a curated mix of CORD-v2 +
WildReceipt + SROIE training data lifts SROIE Task-3 macro-F1
above 0.88 without any architectural modification to the DONUT
backbone."

CONTEXT FROM THE SIBLING kaggle2 REPOSITORY
==========================================
Paper 1 sits in a four-paper research programme (data axis = Paper 1;
architecture axis = kaggle2/paper2/; structural verification = kaggle2/
paper3/; efficiency = future paper-4 repo).  Each paper is a complete
standalone contribution; in particular, Paper 1's text MUST NOT
underplay itself as "just the data axis" or hint that it is one of
a series.  Use full research-paper language and present the
contribution as complete.

The paper's LaTeX structure follows the kaggle2 paper 2/3 convention:
- 13--14 disjoint section files under report/sections/
- A `\VAR{}` injection mechanism in report/inject.py
- A SHA-256-pinned manifest per run under runs/<run_id>/MANIFEST.json
- A NeurIPS-2024-style reproducibility checklist appendix

CONSTRAINTS
===========
- Use ONLY real numbers from the legacy PDF artefacts and run logs
  already in this repository.  DO NOT fabricate any F1, NED, EM, or
  ablation cell.  When a number is genuinely missing, mark it null
  in handoff/results.json and list it in handoff/HANDOFF_README.md
  under `unmeasured`.
- The paper must read as a complete, standalone contribution on the
  data-curation axis.  Use full research-paper language.
- Format: IEEE Access conference template.  Single column.
- Bibliography: include all foundational citations (DONUT, SROIE,
  CORD-v2, WildReceipt) plus relevant data-centric AI references.
- Include a "Reproducibility" appendix listing the run-IDs in this
  repository whose artefacts back every cited number.

DELIVERABLES
============
1. paper1/ directory with:
     - template_paper1.tex (IEEE Access format)
     - sections/ (intro, related, problem, method, experiments,
                  results, discussion, limitations, broader_impact,
                  conclusion, bugs, appendix, repro_checklist)
     - references.bib
2. handoff/ bundle ready for the assembly agent:
     - results.json — flat {key: value} dict for \VAR{} injection
     - figures/    — every PDF the paper cites
     - sections_diff.md — any prose tweaks
     - manifest.json — sha256 of every file
     - HANDOFF_README.md — what the assembler should know
3. A make-target (`make paper1-pdf`) that compiles the paper end-to-end
   from the legacy artefacts via the inject pipeline.
4. A README.md describing how to reproduce the paper from the legacy
   run-IDs.

WORKFLOW
========
Step 1 — Inventory.  Walk the repository, list every legacy PDF +
run-log + result-JSON.  For each artefact, note: which paper it was
intended for, which numbers it cites, what date it was produced.
Output the inventory as INVENTORY.md before writing any LaTeX.

Step 2 — Number-provenance check.  Cross-check 5--10 of the cited
numbers against the run-log JSON to confirm they are reproducible
from the on-disk artefacts.  If any number cannot be sourced, list
it under unmeasured rather than carrying it forward.

Step 3 — Section drafts.  Write each LaTeX section.  Mine the legacy
PDFs for figure captions and table structures; rewrite the prose
to be standalone and to use full research-paper language (no "this
is the data-axis paper of a programme").

Step 4 — Handoff bundle.  Materialise paper1/handoff/ with
results.json + figures/ + manifest.json + HANDOFF_README.md.
The assembler will inject results.json into the LaTeX and compile
to PDF.

START WITH STEP 1 — produce INVENTORY.md.  Do not write any LaTeX
prose until you have audited the legacy artefacts and confirmed
which numbers are real.  Wait for explicit confirmation between
steps if a number you expected to find is missing.

The web service code at /app/ is operational; do NOT modify it.
Only the paper artefacts are in scope for this session.
```
