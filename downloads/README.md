# Download bundles for cloud-agent upload

Four ZIP bundles, one per paper, sized for upload to a cloud-agent
chat session (Claude.ai, ChatGPT, Cursor, Aider, …).  Each bundle
is self-contained: the agent receives the full paper tree (LaTeX
template + sections + references + configs + per-paper Python code
where applicable) without needing to clone the parent repository.

## Bundles

| File | Size | Contents | Cloud agent's job |
|---|---:|---|---|
| `paper2_bundle.zip` | 59 KB | Complete `paper2/` tree from the kaggle2 repo: 9 paper-2-specific Python modules, IEEE Access LaTeX template, 13 disjoint section files, references.bib, config presets, results fixtures, tests, docs.  Excludes `paper2/runs/` (empty in this snapshot). | Compile the paper PDF from existing artefacts; or fill new `\VAR{}` keys after a vast.ai run. |
| `paper3_bundle.zip` | 171 KB | Complete `paper3/` tree from the kaggle2 repo: 24 paper-3-specific Python modules (FOCUS-T, GAT, CNN, ensemble, FOCUS-Σ verifier, Hamming-drift recovery, attention faithfulness, LayoutLMv3 head), ICDAR-main LaTeX template, 14 disjoint section files (incl. SVKIE 5-proposition theory section), wrapper-Δ + paper-F1-gap producers, references.bib, config presets (default + canonical_5seed + 4 sweep presets), results fixtures, tests, docs. | Compile the paper PDF; expand the SVKIE theory; populate `\VAR{}` keys after the wrapper-Δ + ablation runs. |
| `paper1_template_bundle.zip` | 29 KB | **Template** for Paper 1 (data-axis multi-dataset DONUT).  Mirrors `paper2/`'s structure; LaTeX sections derived from `paper2/` but rewritten for the data-curation thesis (intro, results, conclusion); `\VAR{}` keys parameterised for `donut_f1`, `donut_f1_single`, `donut_f1_cord`, `donut_f1_total_*`, training-fold provenance. | Cloud agent fills numbers from the Donut-web-app repo's legacy artefacts (per `docs/agent_prompts/paper1_prompt.md`). |
| `paper4_template_bundle.zip` | 46 KB | **Template** for Paper 4 (efficiency-axis Pruna compression of SVKIE).  Mirrors `paper3/`'s structure; LaTeX sections derived from `paper3/` but rewritten for the compression-crossover thesis; every empirical claim explicitly framed as future work; Pareto plot table parameterised for memory-footprint x-axis (NOT parameter count). | Cloud agent scaffolds the Paper 4 repo (per `docs/agent_prompts/paper4_prompt.md`); fills numbers after the GPU compression sweep. |

## How the bundles are organised

Each bundle expands to a self-contained tree:

```
paper2/                              (or paper3/, paper1_template/, paper4_template/)
├── (per-paper Python code)          # paper2/, paper3/ only — code modules
├── configs/
│   └── default.json                 # paper-specific config preset
├── report/
│   ├── template_paperN.tex          # IEEE Access (paper 1, 2, 4) or ICDAR-main (paper 3) LaTeX template
│   ├── references.bib               # complete bibliography
│   └── sections/
│       ├── intro.tex                # paper-specific framing
│       ├── related.tex
│       ├── problem.tex
│       ├── method.tex
│       ├── experiments.tex
│       ├── results.tex              # all numbers as \VAR{} placeholders
│       ├── discussion.tex
│       ├── limitations.tex
│       ├── broader_impact.tex
│       ├── conclusion.tex
│       ├── bugs.tex                 # bug catalogue (paper 2: 14; paper 3: 14+2)
│       ├── appendix.tex
│       └── repro_checklist.tex
├── results/                         # bundled fixtures
└── tests/                           # paper-specific tests
```

## Compile

Each bundle compiles to PDF with:

```bash
unzip <bundle>.zip
cd <paper-tree>/report
pdflatex template_paperN.tex
bibtex template_paperN
pdflatex template_paperN.tex
pdflatex template_paperN.tex
```

Or with tectonic for zero-friction:

```bash
unzip <bundle>.zip
cd <paper-tree>/report
tectonic template_paperN.tex
```

The compiled PDF will render `\VAR{}` keys as `\MissingCell{key}`
markers wherever the cloud agent has not yet filled in real numbers.
This is the intended behaviour — `\MissingCell` is the audit gate
that ensures fabricated numbers cannot silently appear in the PDF.

## Companion files

- `docs/agent_prompts/README.md` — explains the per-paper agent
  workflow and the handoff-bundle contract.
- `docs/agent_prompts/paper{1,2,3,4}_prompt.md` — the four prompts
  to paste into separate Claude Code sessions when the cloud agent
  is the upstream code-runner rather than the LaTeX compiler.
- `docs/presentation/grand_narrative.tex` — 22-slide Beamer deck
  introducing the four-paper programme + live web app for a
  professor-meeting context.

## Rebuilding the bundles

After modifying the source trees, rebuild with:

```bash
zip -qr downloads/paper2_bundle.zip          paper2/          -x "paper2/runs/*" "*/__pycache__/*" "*.pyc"
zip -qr downloads/paper3_bundle.zip          paper3/          -x "paper3/runs/*" "*/__pycache__/*" "*.pyc"
zip -qr downloads/paper1_template_bundle.zip paper1_template/ -x "*/__pycache__/*" "*.pyc"
zip -qr downloads/paper4_template_bundle.zip paper4_template/ -x "*/__pycache__/*" "*.pyc"
```
