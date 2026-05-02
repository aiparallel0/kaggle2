# Presentation — grand narrative for the four-paper research programme

Beamer slide deck for the meeting that introduces the four-paper
research programme on document key-information extraction (KIE).

## What's here

- `grand_narrative.tex` — 22-slide Beamer deck covering all four
  papers (data axis, architecture axis, structural verification,
  efficiency) plus the live web application.

## Compile

```bash
cd docs/presentation
pdflatex grand_narrative.tex      # standard LaTeX
# or
tectonic grand_narrative.tex      # zero-friction
```

Produces `grand_narrative.pdf` (~22 pages, slide-aspect-ratio).

## Structure

| Slide | Content |
|---:|---|
| 1 | Title |
| 2 | Why document KIE matters |
| 3 | Three orthogonal axes that govern F1 |
| 4 | The four papers + live application (overview table) |
| 5--6 | Paper 1 — Data axis (multi-dataset DONUT training) |
| 7--8 | Paper 2 — Architecture axis (replication + bug catalogue) |
| 9--13 | Paper 3 — SVKIE framework (5 priors, FOCUS-Σ, ablation, wrapper-Δ) |
| 14--15 | Paper 4 — Efficiency axis (Pruna compression crossover) |
| 16 | Live web application (image-to-text.fit) |
| 17 | Reproducibility and open science |
| 18 | Four-axis programme as a unified diagram |
| 19 | Roadmap and timeline |
| 20 | Why the programme is more than four papers |
| 21 | Open questions and future directions |
| 22 | Thank you / Q&A |

## Important framing notes for the meeting

1. **Slide 12 numbers are TARGETS, not measured.**  Frame in the
   meeting as *"expected from the next vast.ai run, scheduled this
   week."*  Single-seed baseline confirms 0.86; the multi-seed
   ablation grid is queued.

2. **Slide 16 web app demo.**  Pre-load
   [image-to-text.fit](https://image-to-text.fit/) before the meeting;
   have a screenshot ready as fallback if the laptop can't reach the
   internet from the room.

3. **Dual-framing rule.**  The presentation makes the four-axis arc
   explicit (Paper 3 integrates priors from Papers 1 + 2; Paper 4
   compresses Paper 3).  The papers themselves do NOT reference each
   other as parents/children — each paper is written as a complete
   standalone contribution.  This is enforced by the per-tree LaTeX
   structure of the kaggle2 repository (`paper2/`, `paper3/`).

4. **Repo provenance for each paper:**
   - Paper 1: Donut-web-app repository (separate session)
   - Paper 2: kaggle2 repository, `paper2/` tree
   - Paper 3: kaggle2 repository, `paper3/` tree
   - Paper 4: future paper-4 repository (separate session)
   - Web app: Donut-web-app repository, deployed via systemd + nginx

## Updating the slides

Numbers come straight from the section text — no `\VAR{}` injection
mechanism here (this isn't a paper build).  Edit the `.tex` file
directly when actual measured numbers come in from the next vast.ai
runs.
