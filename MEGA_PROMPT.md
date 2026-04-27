# Final repair pass — kaggle2 advanced paper rendering & data correctness

You are working in `aiparallel0/kaggle2`. The goal of this task is to land
ONE pull request that fixes every audit finding listed below, in a single
coherent change set, against the reference run
`runs/20260427T071206Z-fd9d7b0`.

Do not ask for clarifications. Do not split into multiple PRs. Do not
remove the 13 F1-destroying-bug guards (README §"hard invariants"). Honour
the 18-file core cap and the 166-LOC-per-file cap (AGENTS.md). Every
public function stays 2-in/1-out and `mypy --strict` clean.

## Reference run — known facts (use as the test fixture)

`runs/20260427T071206Z-fd9d7b0/combined_metrics.json` (83 keys):

| Key                                   | Value                  |
|---------------------------------------|------------------------|
| `test_set_kind`                       | `"canonical_347"`      |
| `test_set_size`                       | `347`                  |
| `seeds_used` / `n_trials`             | `[42]` / `1`           |
| `donut_f1`                            | `0.8216`               |
| `pipeline_f1`                         | `0.8149`               |
| `gtocr_rulebased_f1` / `rulebased_f1` | `null` (canonical-strip) |
| `donut_f1_company`                    | `0.8818`               |
| `donut_f1_company_mean`               | `0.8818`               |
| `donut_f1_company_ci_lo` / `_ci_hi`   | `null` (n=1)           |

So this is a **canonical-SROIE-Task-3, single-seed (42), advanced-template**
build. Every fix below is conditioned on this reality — anything labelled
"missing on a canonical run" is already on `MISSING_OK_PREFIXES` in
`report/missing.py` and must NOT raise; anything labelled "missing
unconditionally" is a real producer bug.

`metrics/unresolved_vars.json` for this run:
```json
{"unresolved": ["gtocr_rulebased_em","gtocr_rulebased_f1","gtocr_rulebased_ned",
                "rulebased_f1","rulebased_f1_address","rulebased_f1_company",
                "rulebased_f1_date","rulebased_f1_total","rulebased_ned"], "count": 9}
```
Every one of these is on the allow-list, yet they render as red
`\MissingCell{}` in the PDF (Audit Issues 1–4). Fix the allow-list bypass.

## Issues to resolve (audit numbering preserved)

### Tier A — paper renders mock-up data

**A1 (Issues 1–4 + 17 partial). `\MissingCell{}` for canonical-mode
rulebased keys.** On `test_set_kind=="canonical_347"` the rulebased and
gtocr_rulebased arms cannot run (no GT boxes), so their `\VAR{}` calls
should be elided, not rendered as red `\MissingCell{}`. Three changes:

- `report/inject.py`: `inject_results()` must consult
  `report.missing.is_missing_ok(key)` for every unresolved `\VAR{}` and
  emit either `\textit{n/a}` (canonical-strip prefixes) or remove the
  enclosing tabular row/cell entirely. Use the existing `is_missing_ok`
  classifier — do NOT re-implement.
- `report/sections/results.tex`, `report/sections/appendix.tex`, and any
  `report/sections/*advanced*.tex` referencing `\VAR{rulebased_…}` /
  `\VAR{gtocr_rulebased_…}`: wrap the rulebased rows in a guard macro
  `\IfRulebased{<row contents>}` defined in `report/template.tex` /
  `report/template_advanced.tex`. The macro reads
  `\VAR{test_set_kind}` and expands to nothing on `canonical_347`.
- `report/inject_tables.py`: when rendering the headline-F1 / per-field /
  ablation tables, drop the rulebased row entirely on canonical. Don't
  emit the row with `\MissingCell{}` cells.

**A2 (Issues 5, 6, 9). Tables VIII, IX, XII rendered as raw
`\begin{tabular}…`.** Caused by `report/inject_tables.py` writing a
`\VAR{table_*}` value whose content contains an unbalanced `{`/`%` /
`\cite{}` that tectonic refuses to compile inside the template's
`\begin{tabular}` slot. Two-step fix:
- In `inject_tables.py`, every emitted tabular MUST go through a
  `_sanitise_tabular(s: str) -> str` that escapes nothing it shouldn't,
  but explicitly verifies (a) balanced braces, (b) every `\cite{}` is
  resolvable from `report/references.bib`, (c) no stray `\VAR{}`. Raise
  `EvalError` if any check fails — that's a build failure, not a render
  silent-corrupt.
- Move the Table-VIII/IX/XII bodies INTO `inject_tables.py` (data-
  driven), drop the static `\begin{tabular}` blocks from the section
  files, and have the section files reference `\VAR{table_canonical_perf}`
  / `\VAR{table_canonical_extended}` / `\VAR{table_competitor_compare}`.

**A3 (Issues 7, 8). Tables X & XI legitimately have `\MissingCell{}` for
latency / assigner-sub-stage rows.** Latency is scoped-out (already on
allow-list `donut_latency_*` etc.). Assigner sub-stage cost rows are not.
Two changes:
- Add `assigner_train_minutes`, `assigner_peak_vram_gb`,
  `assigner_cost_usd`, `assigner_energy_kwh`, `assigner_co2_kg` to
  `MISSING_OK_KEYS` in `report/missing.py` with a comment explaining the
  assigner training is sub-minute and cost telemetry is not collected.
- The latency-table cells should render `\textit{n/a}` (already do); the
  assigner sub-stage cells the same. Once allow-listed, A1 will route
  them through the same elision path.

### Tier B — statistical / numerical errors

**B1 (Issue 10). Table XIII CIs do not bracket point estimates.** The
column header reads `F1 [95% CI]`, but the CI bounds come from
`core/metrics_extended.py::summarise_extended` (per-image bootstrap of
**per-image token-F1**) while the point estimate is the global token-F1
written by `build_combined()`. Two different estimators in one cell.

Fix the producer side:
- `core/metrics_extended.py::summarise_extended`: change the F1 cell
  computation so the bootstrap is over the **same** statistic the point
  estimate represents. Replace the per-image `token_f1` mean with a
  bootstrap whose statistic-of-interest is the global aggregate
  (resample images, recompute global token-F1 over the resample). The
  `_bootstrap_field` helper currently averages per-image F1 — change it
  to a "global-statistic bootstrap" pattern, parameterised by a
  `statistic_fn: Callable[[Sequence[float]], float]`.
- Document in the function's docstring that the returned `(lo, hi)` is
  the CI of the **global token-F1** estimator, not of the
  per-image-mean estimator.
- In `assert_ci_bounds_valid` (which we previously discussed deleting),
  re-enable it as the regression test for THIS fix: it must verify
  `ci_lo ≤ <sys>_f1_<field> ≤ ci_hi` on the reference run. If it
  doesn't pass after the producer fix, the producer is still wrong.

**B2 (Issues 11, 12). Table I `bug_timeline.json` reports pre-fix
absolute F1 in the "ΔF1" column.** Open `results/bug_timeline.json` and
inspect the schema. Two cases:
- If the JSON stores `pre_fix_f1` and `post_fix_f1`, change the
  emitter (`report/figures_bugs.py` AND any `inject_tables.py` row
  generator) to compute and render `delta = ceiling - pre_fix_f1`,
  with the column header staying as "ΔF1".
- If the JSON stores raw `delta` values that are wrong for bugs 7, 10,
  13, treat the JSON as the source of truth and fix it. (`results/` is
  fixtures-only and git-tracked — a one-off correction is allowed; do
  it as part of this PR with an explanation in the commit message.)
- For bug_7 (val/test leakage): this is an *over-reporting* bug, so the
  pre-fix F1 (`0.8500`) is HIGHER than the ceiling (`0.8216`). The
  delta is negative. Render as `−0.0284` and add a footnote in the
  caption explaining "negative ΔF1 ⇒ bug inflated F1 above the
  fixed-pipeline measurement."

### Tier C — architecture / leaderboard / title

**C1 (Issue 13). d=192,L=3 vs d=128,L=2 contradiction.** The shipped
assigner is one configuration; the paper text disagrees with itself.
Open `models/attention_assign.py` (or wherever the assigner is built)
and read the actual `nn.TransformerEncoder` ctor args. Then:
- Update `report/sections/method_pipeline.tex` (Section IV-C / Fig. 3
  caption) and `report/sections/training_setup.tex` (Section VI-B) so
  they BOTH cite the same `\VAR{assigner_d_model}` /
  `\VAR{assigner_n_layers}` / `\VAR{assigner_params_k}` keys.
- Add those three keys to `core/combined_metrics.py` schema and to the
  `build_combined` writer (or `merge_assigner_diag` if it's a
  diagnostic). They must be sourced from the actual model object, not
  from a hard-coded literal.
- Drop every numeric literal for the assigner architecture from the
  .tex files. Test: grep the section files for `1157`, `161`, `192`,
  `128`, `L=3`, `L=2` — must return no matches.

**C2 (Issue 14). "One-Third" vs actual 25.2%.** Title and abstract
overstate. Two acceptable fixes — pick (a):
- (a) Change "One-Third" → `\VAR{param_ratio_phrase}` in the title and
  abstract. The phrase resolver in `report/inject_format.py` already
  has the hook (`param_ratio_phrase` is on `MISSING_OK_KEYS`); make it
  emit `"one-quarter"` when ratio ∈ [0.20, 0.30], `"one-third"` for
  [0.30, 0.40], etc. Bands in the resolver must be a single source of
  truth.
- Also have the resolver write `param_ratio_numeric` (e.g., `"25.2\%"`)
  for use in the conclusion paragraph instead of the current literal.

**C3 (Issue 15). Title says "Beating DONUT" but only beats own
re-implementation.** Two changes:
- Title in `report/template_advanced.tex`: replace "Beating" with
  "Matching" (or "Approaching"). The Conclusion already says "match
  DONUT within ε"; align the title.
- In Table XII competitor comparison, add a footnote distinguishing
  "this work — DONUT (re-impl)" from "DONUT (Kim et al., ECCV 2022)"
  and a 1-line discussion paragraph in Section X explaining the
  re-implementation gap (training data scale / preprocessing / SROIE
  Task-3 split definition). Do NOT silently replace the literal
  competitor numbers — they're cited from the original paper.

### Tier D — minor polish

**D1 (Issue 16).** Algorithm 1 box visually collides with Fig. 3
caption. Either (a) move the algorithm to a separate `figure*` env, or
(b) wrap the algorithm in `\begin{algorithm}[!t] \caption{…} \label{alg:…}
\end{algorithm}` so it gets its own caption space. Pick (b); it's
local to `report/sections/method_pipeline.tex`.

**D2 (CI bounds dead code).** PR #100 left `assert_ci_bounds_valid`
referenced from `stages/paper.py:193` — currently disabled by my local
edit. After B1 lands, re-enable the call and the check; the producer
fix will make it actually pass. Remove the local-edit comment.

## Implementation order (do them in this order; each step is testable)

1. B1 (producer-side bootstrap fix) → re-enable `assert_ci_bounds_valid`.
2. A1 (allow-list bypass + canonical row-elision macro) → 9 unresolved
   keys go to 0 on the reference run.
3. A2 (`inject_tables` becomes the single source of truth for VIII / IX /
   XII).
4. A3 (allow-list additions).
5. B2 (bug_timeline ΔF1 semantics + bug_7 footnote).
6. C1 (assigner architecture single source of truth).
7. C2 (param-ratio phrase resolver).
8. C3 (title + Table XII footnote + Section X discussion).
9. D1, D2.

## Tests to add

- `tests/test_paper_audit_smoke.py` (NEW): for the canonical fixture
  metrics dict, after `expand_inputs` + `collect_unresolved` +
  `is_missing_ok` filter, the residual unresolved set MUST be empty.
- `tests/test_ci_bounds.py`: extend with a regression case using the
  reference run's actual `donut_f1_company` / `_ci_lo` / `_ci_hi`
  triple (from a check-in `tests/fixtures/canonical_347_n1.json`).
- `tests/test_inject_tables_sanitise.py` (NEW): every `table_*` value
  emitted by `inject_tables` is balanced-brace, has zero stray
  `\VAR{}`, and resolves every `\cite{}` against `references.bib`.
- `tests/test_bug_timeline_delta.py` (NEW): for each of bugs 7, 10, 13,
  the emitted ΔF1 ∈ [−1, +1] AND |ΔF1| ≤ |ceiling − floor| + 1e-6,
  AND bug_7's ΔF1 < 0.
- `tests/test_no_arch_literals_in_sections.py` (NEW): grep every
  `report/sections/*.tex` for `1157`, `161`, `192`, `128`, `d=192`,
  `d=128`, `L=2`, `L=3` — assert zero matches outside comments.
- `tests/test_canonical_no_rulebased_rows.py` (NEW): on the canonical
  fixture, `inject_tables` emits zero rulebased rows in the headline,
  per-field, and ablation tables.

## Verification protocol

After all changes:

```
make check              # mypy --strict + ruff
pytest -x tests/        # full suite
make paper              # against runs/20260427T071206Z-fd9d7b0
jq '.unresolved | length' \
   runs/20260427T071206Z-fd9d7b0/metrics/unresolved_vars.json   # ⇒ 0
pdftoppm -r 150 runs/20260427T071206Z-fd9d7b0/paper/paper_filled.pdf \
   runs/20260427T071206Z-fd9d7b0/paper/page -png
```

The rasterised PDF must show:
- Zero red `\MissingCell{}` markers on any page.
- Zero raw `\begin{tabular}` blocks visible to the reader.
- Every Table-XIII row's F1 point estimate inside its `[ci_lo, ci_hi]`.
- Title says "Matching DONUT" (or equivalent) and parameter phrase
  matches the conclusion (`one-quarter` / `≈25%`).
- Algorithm 1 has its own caption box, not colliding with Fig. 3.
- Table I's bug_7 ΔF1 column shows a negative value with footnote.
- Table I's bug_10 and bug_13 ΔF1 columns show values consistent with
  `ceiling − pre_fix_f1`.
- Section IV-C and Section VI-B agree on the assigner architecture.

## Out-of-scope for this PR

- Producing latency / throughput / per-image-USD measurements (still
  scoped-out per AGENTS.md; `\textit{n/a}` is the correct render).
- Re-running training for additional seeds. Single-seed n=1 is the
  reference run; CI bounds across seeds remain correctly absent.
- Changing the conclusion's pipeline-vs-DONUT framing beyond C3.

## Hard rules — reject any patch that violates these

- File-count cap ≤ 18 in core dirs (`core/`, `data/`, `models/`,
  `report/`, `stages/`, `main.py`). Prefer modifying existing files.
- LOC cap ≤ 166 per file in those dirs.
- 2-in/1-out signatures preserved for every public function.
- `# type: ignore` only with a same-line justification comment.
- Never write to `results/` at runtime. The bug_7 footnote / fixture
  edit in B2 is a one-off, hand-authored commit-time change, not a
  runtime write.
- Every `\VAR{}` used by the advanced template must be on the
  allow-list OR populated by the canonical-347 single-seed producer
  path. No exceptions.

Open the PR with title:

> Final paper-render repair pass: rulebased elision, table sanitisation, CI estimator alignment, bug_timeline ΔF1 semantics, assigner architecture single-source, title/phrase truthing
