# Per-paper agent prompts

Four self-contained prompts, one per paper.  Each prompt is designed
to be pasted into a **separate Claude Code session** with the right
repository checked out; each session produces a structured handoff
bundle that is then uploaded to a downstream **assembly agent**
(Claude.ai web) which performs the final LaTeX `\VAR{}` injection
and PDF compile.

## Why this split

- **Paper 1** lives in the **Donut-web-app repo** (not kaggle2).
  Its agent re-renders legacy training results into the modern
  multi-dataset DONUT paper.  The kaggle2 repo serves as the
  reference for paper structure / reproducibility conventions only.
- **Paper 2** lives in this repo's `paper2/` tree.  Its agent
  trains DONUT (paper-faithful recipe) + the rules+zone-prior
  pipeline, runs the multi-seed sweep, populates the bug atlas, and
  emits the headline F1 table.
- **Paper 3** lives in this repo's `paper3/` tree.  Its agent
  trains the SVKIE multi-prior framework, runs the 6-cell ablation
  grid and the wrapper-Δ matrix across DONUT / LayoutLMv3 / FOCUS-T,
  drafts the five theorem proofs (Sec.~\ref{sec:svkie_theory}), and
  emits the cross-architecture results bundle.
- **Paper 4** lives in a separate (future) repo.  Its agent
  scaffolds the compression study with Pruna OSS, writes the
  future-work-framed paper template, and prepares the repo for
  later GPU runs — no measured results in this PR.

## Handoff contract — what each agent uploads back

Each agent produces a single bundle directory at
`<paper-tree-root>/handoff/` containing:

```
handoff/
├── results.json        # flat {key: value} dict — feeds \VAR{} injection
├── figures/            # PDF/PNG figures referenced by the paper
├── sections_diff.md    # any prose tweaks the agent made to its sections
├── manifest.json       # SHA-256 of every file in the bundle
└── HANDOFF_README.md   # what the assembler should know
```

The **assembly agent** (Claude.ai web session) then:
1. Reads `handoff/results.json` and feeds it to `report.inject`.
2. Drops `handoff/figures/*` into the paper's `report/figures/` dir.
3. Applies any `sections_diff.md` patches.
4. Compiles the LaTeX to PDF and verifies `unresolved_vars.json` is
   empty.
5. Returns the compiled PDF + a one-page summary of what changed.

## How to use these prompts

1. Open four separate Claude Code sessions, one per repo:
   - Paper 1 → Donut-web-app repo
   - Paper 2 → kaggle2 repo
   - Paper 3 → kaggle2 repo (different session from Paper 2 to keep
     contexts disjoint and prevent accidental cross-imports)
   - Paper 4 → paper-4 repo (creates the repo as part of its work)
2. Paste the corresponding prompt as the first message.
3. Let the agent run end-to-end; it will produce the handoff bundle.
4. Upload the handoff bundle to a Claude.ai assembly session along
   with the paper's LaTeX template (already in this repo for Papers 2
   and 3) and ask it to compile the PDF.

## Honesty constraints (apply to all four prompts)

- Every numeric claim in the handoff bundle must trace to a real
  artefact (a `runs/<id>/metrics/*.json` file or an equivalent on-disk
  source).  Fabricated numbers are an immediate disqualification.
- When a number is genuinely not yet measured (Paper 4's compression
  cells, or Paper 3's wrapper-Δ on a not-yet-trained LayoutLMv3),
  the bundle must mark it `null` and add an entry to
  `handoff/HANDOFF_README.md::unmeasured` listing what would unblock
  it.  The assembly agent will render `null` cells as
  `\MissingCell{key}` rather than as fake numbers.
- Source-level citations (paths inside the repo) are mandatory for
  every methodology claim.

## Files in this directory

| File | Purpose |
|---|---|
| `README.md` | this file — workflow overview |
| `paper1_prompt.md` | for the Donut-web-app session |
| `paper2_prompt.md` | for the kaggle2 / `paper2/` session |
| `paper3_prompt.md` | for the kaggle2 / `paper3/` session |
| `paper4_prompt.md` | for the paper-4 (new repo) session |
