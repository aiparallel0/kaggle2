# Overleaf import bundle — kaggle2 paper

This directory holds everything you need to take a finished kaggle2 run
and recompile its IEEE paper inside [Overleaf](https://www.overleaf.com/)
with **zero local LaTeX setup**.  Two paper variants ship side-by-side;
pick the one that matches your run.

---

## Variants

| Variant     | Template file                  | Test set            | Arms compared                                                                          | Comparable to public leaderboard? |
|-------------|--------------------------------|---------------------|----------------------------------------------------------------------------------------|------------------------------------|
| `focus`     | `report/template_focus.tex`    | 347 canonical Task-3 (auto-downloaded, sha256-pinned) | DONUT vs YOLO+TrOCR+**Attention assigner**                                              | **Yes**                            |
| `baseline`  | `report/template_baseline.tex` | 63-image internal (from 500/63/63 split of 626 train) | DONUT vs YOLO+TrOCR+**regex** vs **GT-OCR+regex** baseline                              | No (internal split)                |

`focus` is the default.  Run with the baseline variant via:

```bash
python main.py --paper-variant baseline --stage all
```

`config.json` mirrors the choice via the `paper_variant` key, and
`main.py` automatically flips `canonical_sroie_enabled` to match
(`focus`→`true`, `baseline`→`false`).

---

## What to upload to Overleaf

For either variant, upload the following from a completed run
(`runs/<run_id>/`) into a fresh Overleaf project:

1. **Paper source** — `runs/<run_id>/paper/paper_filled.tex`
   (already has every `\VAR{}` resolved by `stage_paper`).
2. **Bibliography** — copy `report/references.bib` next to the `.tex`.
3. **Figures** — every PDF under `runs/<run_id>/figures/` plus any
   flat `runs/<run_id>/fig_*.pdf`.  Both directories should land
   under the Overleaf project root so `\graphicspath` resolves them.
4. **Section sources** — the entire `report/sections/` directory.
   `paper_filled.tex` already has `\input{sections/...}` lines
   inlined by `stage_paper.expand_inputs`, so this step is only
   required if you want to edit individual sections inside Overleaf.

---

## One-shot pack helper

The included script bundles everything above into a single zip ready to
drag into the Overleaf "Upload Project" dialogue:

```bash
bash report/overleaf/pack_overleaf.sh runs/<run_id>
# → runs/<run_id>/overleaf_<variant>.zip
```

Drop the resulting `.zip` into Overleaf → New Project → Upload Project,
then click **Recompile**.  No further edits are required to get a PDF
identical to the one `make all` produces locally.

---

## Per-variant editing notes

### Focus (`template_focus.tex`)
- Title and abstract are scoped to the **626 train + 347 canonical
  test** evaluation.
- The GT-OCR-rulebased baseline is intentionally absent — Task-3 ships
  KIE entities only, no GT box files, so the baseline cannot be
  measured.  Any `\VAR{gtocr_rulebased_*}` left in section sources
  will render as `\MissingCell{...}` and is safe to delete during
  Overleaf editing.
- Reported numbers ARE comparable to public SROIE Task-3 leaderboard
  entries (every leaderboard entry uses the same 347-image set).

### Baseline (`template_baseline.tex`)
- Title and abstract are scoped to the **500/63/63 internal split**.
- All three arms (DONUT, YOLO+TrOCR+regex, GT-OCR+regex) are present.
- Reported numbers are **not** directly comparable to the public
  leaderboard — for that, switch to the focus variant.

---

## Reproducibility checklist (both variants)

- `MANIFEST.json` lists every artefact with sha256 + size + producer
  stage.  Include it in the Overleaf upload so reviewers can verify
  every figure traces back to the original run.
- `metrics/unresolved_vars.json` enumerates any `\VAR{}` keys that
  failed to resolve.  An empty list (`{"unresolved": [], "count": 0}`)
  is the contract for a clean release.
- The bib file uses IEEEtran style; do not change `\bibliographystyle`
  unless the target venue requires otherwise.
