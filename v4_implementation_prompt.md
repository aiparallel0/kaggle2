# V4 Implementation Prompt — KIE Paper Revision

You are an implementing AI agent. Your task is to take the v3 revision of
the paper *"End-to-End vs. Pipeline Receipt KIE: DONUT Against
YOLO+TrOCR+Attention on SROIE"* and produce a v4 revision that lifts it
from **workshop-grade (current ~6.5/10)** to **ICDAR/DAS conference-grade
(target ~7.5/10)**.

This is your fourth pass. Read this entire prompt before writing any code.
Three prior reviews exist (v1, v2, v3); their conclusions are folded into
this single prompt so you do not need to re-read them.

---

## Section 0 — Read the prior trajectory before you begin

You have shipped three revisions. Their trajectory was:

- **v1 (3c3d8550)**: 5.2/10. Six tables and three figures shipped with
  missing data; McNemar p="0.0000" contradicted bootstrap CI claim;
  Fig. 13 (qualitative samples) was unreadable red 4pt text.
- **v2 (bc4e13c1)**: 4.9/10 — *regressed*. You added four new sections
  (RAG, GAT, KD ablation, controlled-bug-ablation harness) without
  running any of them, shipping ~70 em-dash cells. Fig. 11 was still
  broken. Two new dangling references (`[?]`, `[figure absent]`).
- **v3 (77d8c158)**: 6.5/10 — recovery. Tables IV/V/VI/IX/XIII
  populated; bootstrap CIs printed (`[−0.0383, 0.0603]`); McNemar
  printed cleanly (`p = 4.2 × 10⁻⁷`); per-bug ΔF1 measurements appeared
  in Table I. Fig. 11 still broken. Fig. 1 still schematic. `[?]`
  and `Sec. ??` still in the rendered PDF.

**The lesson the v3 score teaches**: the fastest path forward is *not*
new sections. It is closing every remaining em-dash, every dangling
reference, and every broken figure that has been broken since v1.

In v4, you will do exactly six things, in the order given. **Do not add
new sections. Do not invent new ablations. Do not change the
methodology.** If you find yourself writing a new \section{}, stop and
re-read this paragraph.

---

## Section 1 — Final venue target

You are aiming for **ICDAR 2026 or DAS 2027** as the realistic
publication target. ICDAR is the natural home for SROIE-benchmarked
work; DAS accepts well-engineered system papers. Both venues:

- expect comparison to the canonical SROIE 347-image test set
- expect at least one strong contemporary baseline (LayoutLMv3 / BROS /
  DocFormer)
- accept n=3 seeds as a floor; n=5 preferred
- read figures professionally — they will not accept Fig. 11 in its
  current form
- have one-pass review (no revise-and-resubmit), so the artefact must
  be presentable on submission

You are explicitly **not** aiming for top-tier conferences (CVPR,
NeurIPS, ICML). Those require methodological novelty beyond what this
paper has, plus 3+ datasets. That work is 3 months out, not in scope
for v4.

You are also explicitly **not** going down to workshop tier. The bug
catalogue alone could be a workshop paper, but this whole paper has
the data and engineering quality for a full conference paper if the
six items below are closed.

---

## Section 2 — The six items, in priority order

Each item is a checklist with acceptance criteria. Complete an item
fully before starting the next. Do not interleave.

### Item 1 — Run on canonical SROIE 347-image test set (highest leverage)

**Why this is item 1.** The single biggest weakness in v3 is the
non-canonical 500/63/63 split. ICDAR reviewers know SROIE's standard
626/347 split. Without a number on the 347, they cannot compare your
DONUT F1 of 0.8306 to leaderboard numbers (~0.94+). This is the
fastest single fix that adds the most value.

**What to do:**

1. Locate or download the canonical SROIE 2019 test set (347 receipt
   images + ground-truth field labels). The standard source is the
   ICDAR 2019 SROIE competition page; failing that, the
   `naver-clova-ix/synthdog` and related HuggingFace mirrors carry
   it. Persist the path to `config.canonical_sroie_test_path`.

2. Add an evaluation flag: `config.eval.canonical_sroie = true`. When
   set, the eval stage runs each trained model (DONUT, pipeline) on
   the 347-image canonical test set in addition to the 63-image held-out
   test split.

3. Add **one new table**, Table IV-bis (or rename Table IV to "Headline
   results — 63-image held-out and 347-image canonical splits").
   Schema: `Architecture | Split | F1 | NED | EM`. Two splits × three
   architectures = 6 rows.

4. **Do not retrain** — use the same trained DONUT and pipeline
   checkpoints that produced v3's headline numbers. This is purely an
   inference pass.

5. Add a sentence in Section VII (Results) right after the existing ΔF1
   discussion: *"On the canonical 347-image SROIE test set, DONUT
   reaches F1 = X and the pipeline reaches F1 = Y; both fall below
   the published SROIE leaderboard (~0.94+) because we trained on only
   500 images vs. the leaderboard's 626. The gap to leaderboard
   reflects training-set size, not architectural choice."*

6. Update the Limitations bullet about non-canonical split: *previously*
   "our F1 values are not directly comparable to SROIE leaderboard
   numbers" → *now* "our headline 63-image numbers are not directly
   comparable; the 347-image canonical numbers in Table IV-bis can be
   compared, with the caveat that we trained on 500 receipts rather
   than the leaderboard standard 626."

**Acceptance criteria:**
- Table IV-bis exists with 6 populated rows.
- Section VII has the new sentence.
- Limitations bullet updated.
- `config.eval.canonical_sroie` is documented in Appendix A.
- Compute cost: one inference pass per model on 347 images. ~5 min
  per model on RTX 4090. ~$0.05 total.

---

### Item 2 — LayoutLMv3 baseline

**Why this matters.** ICDAR reviewers will look at your headline table,
see DONUT and pipeline only, and ask "where is LayoutLMv3?" without
reading further. One row in the headline table for LayoutLMv3
preempts that.

**What to do:**

1. Use the public HuggingFace checkpoint
   `microsoft/layoutlmv3-base`. Do not train from scratch; fine-tune
   the public checkpoint on your 500-receipt training split.

2. LayoutLMv3 is BIO-tagging-based. Convert your 500 training receipts
   from `(image, field_dict)` format into the `(image, OCR_tokens,
   bounding_boxes, BIO_tags)` format LayoutLMv3 expects. Use the
   following conversion:
   - Run YOLOv8n + TrOCR (your trained checkpoints) over each training
     image to get `(token, box)` pairs.
   - For each token, assign a BIO tag: `B-COMPANY`, `I-COMPANY`,
     `B-DATE`, `I-DATE`, ..., `O`.
   - Token-to-field assignment uses the same rule-based heuristic that
     produces your `Rule-based (YOLO+TrOCR)` baseline. This means
     LayoutLMv3 inherits some rule-based assignment noise; that is a
     fair representation of how LayoutLMv3 is typically used.

3. Train LayoutLMv3 with the same training budget as DONUT (35 epochs,
   batch 4, bf16). Save the resulting checkpoint to
   `runs/<run_id>/checkpoints/layoutlmv3/`.

4. Evaluate on both the 63-image held-out test set and the 347-image
   canonical test set. Report F1, NED, EM.

5. Add LayoutLMv3 as a row to Table IV (and Table IV-bis from item 1),
   Table V (per-field F1), and Table VIII (per-field %). Do NOT add
   it to every table — the headline tables are sufficient.

6. Add a one-paragraph subsection §VI.D "LayoutLMv3 baseline":
   describe the BIO conversion, the training setup (matching DONUT's
   budget), and the resulting numbers. Two sentences in §VII discuss
   how it compares to DONUT and to the pipeline.

**Acceptance criteria:**
- LayoutLMv3 row appears in Tables IV, IV-bis, V, VIII.
- §VI.D exists and is one paragraph (~150 words).
- Limitations bullet "No LayoutLMv2/v3, TILT, UDOP, StrucTexT, PICK, or
  zero-shot VLM baseline is reimplemented here" → *trim to* "No TILT,
  UDOP, StrucTexT, or PICK baseline is reimplemented here; LayoutLMv3
  appears in Tables IV/V/VIII."
- Compute cost: one full fine-tuning run (~35 min on RTX 4090) +
  inference. ~$0.30.

**If LayoutLMv3 fine-tuning fails for any reason** (BIO conversion
issues, HF checkpoint compatibility): do *not* ship a row of em-dashes.
Either fix the issue or drop the baseline entirely. An empty
LayoutLMv3 row is worse than no LayoutLMv3 row.

---

### Item 3 — Run n_trials=3

**Why this matters.** The paper claims a "seed-parametric harness" but
has shipped n=1 for three revisions. ICDAR reviewers will note this.
n=3 is the floor for a credible mean ± std on F1.

**What to do:**

1. Set `config.seeds = [42, 7, 123]` and `config.n_trials = 3` in
   `config.json`. Verify the harness reads both lists and runs 3
   independent training runs per architecture.

2. Run all three training runs for DONUT, the pipeline (YOLO + TrOCR
   + assigner), and LayoutLMv3 (from item 2). This is 3 × (1 DONUT
   run + 1 pipeline run + 1 LayoutLMv3 run) = 9 training runs.

3. On the 63-image held-out test set, report **mean ± std** over the 3
   seeds for F1, NED, EM in:
   - Table IV (headline)
   - Table V (per-field F1)
   - Table VIII (per-field %)
   - Table IX (P, R, F1, EM)

4. The bootstrap CI on ΔF1 should now be computed from the **per-seed
   point estimates** rather than from a single seed. Update Section
   VII: *"On three seeds (42, 7, 123) the bootstrap-95% CI on ΔF1 is
   [X, Y] (n_trials=3, 1000 resamples per seed, then averaged
   per-image)"*.

5. The McNemar test should also be run per-seed. Report the **median
   p-value across seeds**, not a single seed's value. Caveat in
   Section VII: *"per-seed p-values were [p₁, p₂, p₃]; we report the
   median."*

6. Update Table III (training summary): the "best epoch" cell becomes
   "best epoch (mean ± std across seeds)".

7. Update the abstract: *"With n_trials = 3 ... DONUT reaches a mean
   global token-F1 of X (std Y) and the pipeline reaches Z (std W),
   ..."* The point estimate stays roughly the same; the abstract now
   has a credibility floor.

**Acceptance criteria:**
- All headline tables show mean ± std.
- Section VII discusses across-seed variance.
- Abstract has mean and std.
- The single-seed limitation bullet is removed.
- Compute cost: 9 training runs × ~30 min = ~4.5 hours wall clock,
  or ~1.5 hours on 3 parallel RTX 4090s. ~$2.50.

---

### Item 4 — Redo Fig. 11 (qualitative samples) — third strike, must close

**Why this matters.** Fig. 11 has been broken across three revisions
and is the single most-mentioned issue in every prior review. Reviewers
who skim the figures-only will form their initial impression from this
figure. Right now that impression is "the authors do not care."

**What to do — total replacement, not iteration.**

1. **Pick exactly 4 receipts** (not 12, not 9). Curate by failure mode:
   - **Receipt A** — both correct on all four fields (validation that
     systems work).
   - **Receipt B** — DONUT correct on address (multi-line), Pipeline
     wrong (illustrates the TrOCR-per-crop weakness).
   - **Receipt C** — Pipeline correct on total, DONUT wrong (DONUT
     confuses subtotal vs. total).
   - **Receipt D** — both wrong, in instructive ways (e.g. company
     misread by both, but in different ways — DONUT lexical, Pipeline
     OCR).
   
   Sample IDs: pick from receipts where you have predictions stored
   for both systems. Suggested: receipt 005 (both partial), 022
   (DONUT-wins-address case), 077 (pipeline-wins-total case), 074
   (both struggle).

2. **Layout** — one receipt per page section, 4 receipts total, in a
   2×2 grid:
   - Top of each cell: **the actual receipt image**, ~200px wide,
     readable at print resolution. Use `\includegraphics` from
     `data/sroie/img/<id>.jpg`.
   - Below the image, three labelled blocks: `GT:`, `DONUT:`,
     `Pipeline:`. Each block shows the four fields stacked
     (`company:`, `date:`, `address:`, `total:`).
   - **Black text, 9pt monospace** (`\ttfamily \footnotesize`).
   - Per-field correctness markers from v3 (the red `x`) preserved,
     but black for the field name and red only for the `x` itself.
   - No horizontal overflow — each block is its own minipage with
     `\linewidth` and text wraps cleanly.

3. **Caption** — replace v3's "12 receipts" with "4 receipts curated
   by failure mode (correct / DONUT-wins / pipeline-wins / both-fail);
   full 63-receipt qualitative dump in supplementary." Then move the
   12-receipt full grid to a supplementary appendix figure. Don't
   delete it; just demote it.

4. **Producer**: rewrite `figures/fig_qualitative.py` from scratch.
   Inputs: list of 4 receipt IDs from `config.qualitative_sample_ids`,
   predictions JSON, ground-truth JSON, image directory. Output: a
   single PDF with the 2×2 grid. Use `matplotlib` + `PIL` or pure
   LaTeX `\begin{figure}` — either works as long as receipt images
   are visible.

5. **Test it**. Render the figure, open it at print resolution
   (Acrobat at 100% zoom), confirm:
   - All four receipt images are visible and recognizable as receipts.
   - All text is readable without zooming past 100%.
   - No horizontal text overflow.
   - The red `x` markers are visible.

**Acceptance criteria:**
- Fig. 11 shows 4 receipt images, each with GT/DONUT/Pipeline columns
  beneath in black 9pt monospace.
- Caption matches the figure (no "12 receipts" claim).
- Full 12-receipt grid relocated to supplementary appendix.
- A reviewer can read the figure on a printed page without a magnifying
  glass.

**This is non-negotiable.** If you find yourself writing
"add more annotations to the existing format", stop. The task is
*replacement*, not iteration.

---

### Item 5 — Redo Fig. 1 (architecture diagram)

**Why this matters.** The paper is about reading receipts and contains
zero readable receipts in its first 6 pages. Fig. 1 is currently a
chain of `\node[box]` rectangles — schematic, uninformative. A reader
who hasn't read DONUT learns nothing from "Receipt image → Swin
encoder → BART decoder → XML tokens".

**What to do:**

1. Pick one representative receipt (ideally Receipt A from Fig. 11 —
   the one where both systems get everything right). Use this same
   receipt throughout Fig. 1 for both DONUT and pipeline panels.

2. **DONUT panel (a)**:
   - Real receipt thumbnail (~150px wide) on the left.
   - Arrow → Swin encoder block (with caption "Hierarchical patches,
     4×4")
   - Arrow → BART decoder block.
   - Arrow → **the actual XML output string**: `<s_company>SUPER
     MART</s_company><s_date>03/01/2019</s_date>...` — show the first
     ~80 chars at 8pt monospace, truncate with "..." if needed.
   - Arrow → final parsed JSON dict, also at 8pt monospace.

3. **Pipeline panel (b)**:
   - Same receipt thumbnail.
   - Arrow → YOLOv8 detector block, with **the receipt re-rendered
     with detected bounding boxes overlaid in colored rectangles**
     (~100px wide).
   - Arrow → TrOCR per crop, showing 3 example crops with their
     decoded strings underneath.
   - Arrow → Attention assigner, showing the 4×N attention matrix as
     a small heatmap (~80px wide).
   - Arrow → final parsed JSON dict.

4. Use TikZ + `\includegraphics` for the receipt thumbnails. The
   bounding-box overlay can be pre-rendered in
   `figures/fig1_assets/receipt_with_boxes.png` and included.

5. The figure should be **wide** — span both columns of the IEEE
   two-column template using `\begin{figure*}`. It is currently
   one-column and that is too cramped.

**Acceptance criteria:**
- Fig. 1 shows the same receipt visibly flowing through both
  pipelines.
- A reader who has not read DONUT can understand "this model takes a
  receipt image and outputs structured XML" from the figure alone.
- Spans two columns.
- Print-readable.

**Compute cost:** zero. This is layout + asset preparation work.

---

### Item 6 — Close the long-tail bugs

These are quick fixes that have lingered. Do them in one pass.

#### 6a. Resolve `[?]` (LayoutLLM citation)

In §IV.E (RAG-KIE), the line *"This mirrors the 4-shot in-context
learning scheme of LayoutLLM [?]"* has had a missing citation since v2.

- Either add the actual citation (LayoutLLM paper: Liu et al., 2024,
  arXiv:2404.05225 or similar — verify the actual reference) and add
  it to the BibTeX file.
- Or, if you cannot identify the exact paper, rewrite the sentence to
  cite RAG (Lewis et al., 2020) or in-context learning (Brown et al.,
  2020 / GPT-3) directly and remove the LayoutLLM reference.

#### 6b. Resolve `Sec. ??`

In Appendix A, *"decoder uses η_d = 5 × 10⁻⁴ — see Sec. ??"* is a
broken `\ref{sec:donut_lr}` or similar. Either:
- Add a proper `\label{sec:donut_lr}` to §IV.B's "Differential learning
  rate" subsection and update the `\ref`.
- Or rewrite the sentence to say "see §IV.B" directly without the
  `\ref`.

#### 6c. Populate Table X (latency)

Currently 12/12 cells em-dash. Do this:
1. Add a flag `config.eval.measure_latency = true`.
2. In the eval loop, for each test image, time the forward pass
   (DONUT: full image-to-XML; Pipeline: detect + crop OCR + assign).
   Use `torch.cuda.synchronize()` before and after.
3. Compute mean / p50 / p95 / p99 over 63 images. Throughput =
   1000 / mean_ms. USD/img = (cost_per_run / 63).
4. Run once per system. ~5 minutes.

If this fails for any reason, **delete Table X entirely** rather than
ship em-dashes again.

#### 6d. Populate Table XIV (env snapshot)

Currently 8/8 cells em-dash since v1. The `core.env_snapshot.py`
producer should write `env/hostinfo.json` and `env/git_sha.txt`.
Verify the producer runs and `report.inject` reads from those paths.

The fields are trivial:
- `git SHA`: `git rev-parse HEAD`
- `config sha256`: SHA of `config.json`
- `torch`: `torch.__version__`
- `CUDA`: `torch.version.cuda`
- `GPU`: `torch.cuda.get_device_name(0)`
- `driver`: from `nvidia-smi --query-gpu=driver_version --format=csv`
- `seed`: `config.seeds[0]`
- `run_id`: `config.run_id`

This should be 30 minutes of work. If the inject layer is genuinely
broken, hard-code these strings in the LaTeX template — they don't
change between runs of the same revision.

#### 6e. Fix Table XI column overflow

In v3, Table XI's last column "CO₂eq" is truncated to "CO" by column
width. Use `\resizebox{\linewidth}{!}{...}` around the tabular, or
shorten the column header to "CO₂" or use a smaller font
(`\footnotesize`). Verify by re-rendering.

#### 6f. Fix Fig. 4 train-above-val anomaly

The training-trajectory figure shows train loss above val loss for
all 39 epochs. Add **one sentence to the caption**: *"Training loss
remains above validation loss throughout because the train set
includes higher OCR-noise variance than val (val is sampled from
cleaner receipts in the deterministic seed-42 split). This is not a
data-leakage indicator; the split assertion in `core.config` confirms
zero overlap (see Bug 7)."*

If you cannot confirm this is the actual cause, run a quick
diagnostic: compute the average OCR character-error rate per receipt
on train vs val. If train > val on noise, the caption is correct.
If not, add an honest "we are investigating this anomaly" line and
do not gloss over it.

#### 6g. Update assigner architecture description to match 1157K

§IV.C says "2-layer pre-norm Transformer encoder, 8 heads, d=128".
With 161K → 1157K parameters, the architecture must have changed.
Either:
- Restore the 161K-parameter assigner if the v1 numbers are still
  reachable (rerun the trained-from-scratch flow), and revert all
  "1157K" mentions to "161K". Cleanest.
- Or, identify what the 1157K assigner actually contains (likely
  d=384 or L=6 instead of d=128, L=2), and update §IV.C accordingly.
  Add a sentence: *"The shipped assigner uses d=384 and L=6 totaling
  ~1.16M parameters; an earlier 161K-parameter d=128, L=2 variant
  reached comparable F1 and is preserved as `models/assigner_small.py`
  for ablation."*

The current paper has a description that doesn't match the parameter
count. This will be caught by a careful reviewer.

#### 6h. Demote Table XII (zero-shot ceiling) honesty

Currently the caption admits the row is a fixture, not a measurement,
in language designed to be skipped. Either:
- **Run the API call**. 63 receipts × one Anthropic Sonnet API call
  per receipt at ~$0.005 = ~$0.32 total. Cache the responses by
  SHA-256 of image bytes (the producer already supports this per the
  caption). One afternoon of work. Replace fixture values with
  measured.
- **Or remove Table XII entirely** and replace with a future-work
  bullet: *"Future work: zero-shot foundation-model ceiling. Based on
  related work, frontier VLMs are expected to reach ~0.6 F1 on SROIE
  zero-shot; we leave the empirical measurement to a follow-up."*

A fixture row in an evaluation table is not acceptable for a
conference submission. This is a credibility issue, not just a
completeness issue.

#### 6i. Build-system hardening

Add to the build pipeline (Makefile or CI):

```makefile
check_artefacts:
	@grep -l '\\?\\?' report/*.tex && echo "ERROR: undefined references" && exit 1 || true
	@grep -l '\[\?\]' report/*.tex && echo "ERROR: undefined citations" && exit 1 || true
	@grep -l 'figure absent' report/*.tex && echo "ERROR: missing figure" && exit 1 || true
	@find runs/$(RUN_ID)/figures -name '*.pdf' -size 0 -print -quit | grep . && echo "ERROR: empty figure" && exit 1 || true

all: check_artefacts <existing rule>
```

Promote em-dashes from a soft warning to a hard build error. The
`report.inject` layer should fail the build when a `\VAR{}`
substitution returns an em-dash unless that variable is on an
explicit allow-list of "intentionally not measured."

This was suggestion S16 in the original review and it has been
ignored across three revisions. Do it now. It will catch every issue
in items 6a–6e automatically and prevent future regressions.

**Acceptance criteria for item 6:**
- No `[?]`, `??`, `figure absent`, or em-dash cell remains in the
  rendered PDF (except cells explicitly marked `n/a`).
- Table X populated or removed.
- Table XIV populated.
- Table XI column header readable.
- Fig. 4 caption explains the train-above-val pattern.
- §IV.C architecture description and parameter count agree.
- Table XII either measured or removed.
- `make check_artefacts` exists and passes.

---

## Section 3 — What you must NOT do

These are anti-patterns. Each was tried in v2 and made the paper
worse.

1. **Do not add new methodology sections.** No new GAT variant. No new
   RAG arm. No KD ablation that you don't run. The paper has enough
   methodology; it needs more *evidence* for what it already has.

2. **Do not paper over missing data with em-dashes.** If a producer
   doesn't run, fix the producer or delete the table. An em-dash in
   a published paper is a credibility leak. The build-system change
   in 6i exists to enforce this mechanically.

3. **Do not reorganize sections.** §V (bug catalogue) stays where it
   is. §IV.A–C stays. §VI–XII stays. Any reorganization burns
   reviewer goodwill ("the paper looks different than the abstract
   describes").

4. **Do not add the bug catalogue's controlled-ablation 2D heatmap.**
   v2 promised this, v3 silently dropped it. That is the right
   choice. Don't reintroduce.

5. **Do not change the seed list mid-revision.** Once you commit to
   `seeds = [42, 7, 123]`, those three are the three. Don't drop one
   that gives bad numbers. Don't add a fourth that gives a better
   mean. Reproducibility requires committed seeds.

6. **Do not add datasets beyond canonical SROIE.** No CORD, no
   WildReceipt. Those are noted as future work and that's fine.

7. **Do not exceed 12 pages.** ICDAR limit is typically 15; DAS is 14.
   You are at ~13 in v3. After items 1–6 the paper grows by ~1 page
   (LayoutLMv3 row + canonical-SROIE table + better Fig. 1) and
   shrinks by ~0.5 page (Fig. 11 collapsing 12→4 receipts). Net
   change: small. Stay under 14 pages.

---

## Section 4 — Suggested execution order

Do the items in numerical order (1, 2, 3, 4, 5, 6) but parallelize
within each item where possible.

**Day 1 (compute):**
- Morning: kick off the canonical SROIE eval (item 1) on the existing
  trained checkpoints. ~10 min wall time.
- Morning: kick off LayoutLMv3 fine-tuning (item 2). ~35 min wall time.
- Morning: kick off n_trials=3 retrains for DONUT and pipeline
  (item 3). ~3 hours wall time.
- Afternoon: while compute runs, start Fig. 11 redesign (item 4) and
  Fig. 1 redesign (item 5). These are layout work, no compute.

**Day 2 (closing):**
- Morning: collate item 1 + 2 + 3 results into tables.
- Morning: complete Fig. 11 and Fig. 1 redesigns.
- Afternoon: item 6 long-tail (6a–6i) in one pass.
- Evening: full `make all` build, verify `make check_artefacts`
  passes, render PDF, eyeball every figure and every table.

**Total time:** 2 days. Total compute cost: ~$3.

---

## Section 5 — How to know you're done

Before submitting v4, run this self-check:

1. **Print the PDF and read it on paper.** Every figure must be
   readable without a magnifying glass. Every table cell must contain
   a number or a meaningful label (not em-dash).

2. **Open the PDF in Acrobat at 100% zoom.** Search for `?`, `--`,
   `figure absent`. Zero hits expected.

3. **Read the abstract to a non-expert.** They should be able to
   identify: (a) what the paper does, (b) what was measured, (c) what
   the headline number is, with std. If they can't, rewrite the
   abstract.

4. **Take Fig. 11 to a colleague who has never seen the paper.** Ask
   them to identify the four cases (correct / DONUT-wins / pipeline-
   wins / both-fail) without your help. If they can, the figure works.
   If not, redo.

5. **Check every limitation bullet against the body.** v3 still had
   "Single training seed" as a bullet after items 1–3 are complete,
   that bullet must be removed.

6. **Check every "future work" claim.** If you fixed it in v4, remove
   it from future work.

7. **Pull the LayoutLMv3 row out of every table by eye.** It should
   appear in Tables IV, IV-bis, V, VIII. Confirm presence in all four.

8. **Verify the bootstrap CI in the abstract matches Section VII**
   matches Appendix B's four-decimal numbers. All three should be
   the same number to within rounding.

9. **Run `make check_artefacts` from item 6i.** Must exit 0.

10. **Compare v3.pdf and v4.pdf side-by-side.** Specifically look at
    Fig. 11 — if it looks similar, you have not done item 4.

If all 10 checks pass, ship.

---

## Section 6 — What target score you should expect

If you complete items 1–6 cleanly:

- Methodology rigor: 6 → 7 (LayoutLMv3 baseline, n=3 seeds)
- Statistical honesty: 8 → 9 (canonical SROIE, n=3 CIs)
- Figure quality: 4 → 7 (Fig. 1 + Fig. 11 redone)
- Use of metrics: 8 → 9 (LayoutLMv3 in headline tables, latency
  populated)
- Use of images: 3 → 7 (real receipts in Fig. 1 and Fig. 11)
- Tables: 7 → 9 (Tables X, XII, XIV, XI all closed)
- Writing & structure: 6 → 7 (no dangling refs, honest captions)

**Aggregate target: 7.5/10. ICDAR/DAS borderline-accept territory.**

If you complete items 1–4 but skip 5–6, target ~7.0. Still a
significant improvement over v3's 6.5, but Fig. 1 and the long-tail
bugs are noticeable enough that one item is unlikely to flip a
reviewer who otherwise wants to accept.

If you only complete items 1–3, target ~6.8. The headline
table will look right but the visual presentation will still drag
the paper down.

**Do all six.**

---

## Section 7 — One paragraph that captures the spirit of v4

v4 is a closing pass. It is not a research pass and it is not an
exploration pass. It is the pass where the paper stops embodying the
silent-failure pathology it documents. Every figure must say what its
caption says. Every table must contain numbers, not em-dashes. Every
reference must resolve. Every claim in the abstract must be backed by
a measurement in the body. The work is done; v4 makes the work
visible. Ship the closing pass; do not start a new investigation.

---

## Section 8 — If something goes wrong

If during execution you discover that one of the six items is genuinely
infeasible in 2 days (e.g. LayoutLMv3 has a HF compatibility issue you
can't resolve, or the canonical SROIE 347-image labels can't be
located), do not silently drop the item. Instead:

1. Document what you tried in a `v4_issues.md` file in the repository.
2. Decide between (a) skipping the item with a clear note in
   limitations, or (b) shipping a fallback that is still
   measurement-grade (e.g. BROS instead of LayoutLMv3 if
   LayoutLMv3 fails).
3. Update the limitations section honestly.
4. Continue with the remaining items.

A v4 with 5 of 6 items closed is still a meaningful improvement over
v3. A v4 that fakes the missing item is a regression to v2 mode.

---

End of prompt. Begin with item 1. Do not write any code until you
have read this entire document and confirmed you understand the
priority order.
