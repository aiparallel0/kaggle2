# Relational fix: a shared receipt-zone prior for `company` and `total`

## 1. What is actually wrong (data-flow level)

`pipeline_f1_company` and `pipeline_f1_total` are not failing for unrelated reasons. They are failing because every dispatch path in `models/focus_pipeline.py` treats each field in isolation, and the only structural signal the assigner sees is `priors_v4[:, V4_Y_NORM_IDX]` — a *raw* y-coordinate, never bucketed into a receipt zone.

Trace a single receipt through the pipeline:

1. `_assign_learned_with_attn` calls `assigner.company_pick(kv[0], texts, y_col, bp_col)` (PR #119) and, for `total`, falls through to `total_arithmetic.solve(...)` (PR #128) and then to `total_post.py` regex.
2. `company_pick` argmaxes a softmax over **all** OCR lines. Boilerplate at the bottom of the receipt (`CHANGE`, `CASH`, `THANK YOU`) is in scope. The only thing keeping it out is `_COMPANY_HEADER_RE` (PR #123) — a denylist. Denylists never enumerate the full distractor set.
3. `total_arithmetic.solve` scans all parsed money values on the receipt. The header sometimes contains a phone number formatted `03-1234.5678` or a registration suffix `(559208-M)` that the money regex in `total_post.py` accepts as a numeric candidate, polluting the `subtotal+tax+service−discount` enumeration.
4. The two failures are the **same failure mirrored**: `company` leaks downward into the totals zone; `total` leaks upward into the header zone. Neither head knows where the receipt's items block ends.

The signal needed for both is a single 3-class posterior `P(zone ∈ {header, items, totals} | line_i)`. It does not exist anywhere in `priors_v4`, in `AttentionAssigner`, in `_company_anchor_filter`, or in `total_arithmetic`.

## 2. Why earlier agents did not catch it

Read the last twelve PR titles back-to-back: every one is scoped to a single field or a single defect (`Lift FOCUS pipeline_address_precision`, `Add company span fallback`, `Add arithmetic consensus solver for receipt total field`). Coding agents were given per-field acceptance gates (`pipeline_f1_company ≥ 0.92`, `pipeline_f1_total` floors), so they optimised the loss surface they were shown. Cross-field structure is invisible from inside that surface — improving company by adding `_COMPANY_HEADER_RE` does not improve total, and adding `total_arithmetic` does not improve company, so neither agent had any gradient toward the shared prior.

A second reason: PR #119 demonstrated that a trained head can ship in `assigner.pt` and never be invoked at inference for an entire release cycle. There is no integration test of the form *"every learned head registered in `_architecture_config` must appear on the dispatch path of `_assign_learned_with_attn`"*. The same blind spot applies one level up — there is no test that exercises company and total *on the same receipt* and asserts they pick lines from disjoint y-bands.

A third reason: acceptance fixtures. PR #119 admits its post-FOCUS-C fixture carries placeholder numbers; PR #123 admits 0.92 cannot be checked without GPU. Several "company F1" PRs landed without a real eval. The CI signal degenerated to "ruff + mypy pass" for the very metrics the PRs claim to move.

## 3. The single relational change

Add a `ZonePrior` — a 3-state segmentation of the receipt's lines into `{header, items, totals}` — and route both FOCUS-C and FOCUS-T through it. One module, ≤120 LOC, no new tensor on the GPU path.

### 3.1 Module: `models/zone_prior.py`

Inputs (2-in/1-out, as required):
- `lines: list[OcrLine]` (already constructed upstream)
- `cfg: ZoneConfig` (new sub-dataclass on `ExpConfig`)

Output: `ZonePosterior` — for each line `i`, a length-3 vector `(p_header, p_items, p_total)` summing to 1.

Implementation: a fixed-topology 3-state HMM, decoded with forward–backward, transitions hard-coded to enforce monotonic order `header → items → totals` (no back-transitions). Emission features per line, derived only from values already present in `priors_v4` plus three boolean indicators built from existing regex tables:

| Feature | Source already in repo |
|---|---|
| `y_norm` | `priors_v4[i, V4_Y_NORM_IDX]` |
| `is_money_line` | reuses the money regex in `models/total_post.py` |
| `is_item_qty_line` | reuses `\d+\s*(x|@|pcs)` pattern present in `models/eval_pipeline.py` |
| `is_company_anchor` | reuses `_COMPANY_ANCHOR` from PR #123 |
| `is_totals_kw` | reuses `TOTAL`/`SUBTOTAL`/`CASH`/`CHANGE` keywords already in `total_post.py` |
| `is_company_boilerplate` | reuses `priors_v4[i, V4_IS_COMPANY_BOILERPLATE_IDX]` |

Emissions are a 6-dim logistic-regression layer per state, parameters fit by EM on the 500-receipt train split using gold field y-positions as weak supervision. This is roughly 18 floats × 3 states + transition matrix — under 100 parameters, trivially serialisable to JSON in `results/zone_prior.json` (fixture-allowed) or as a tiny `nn.Linear` in `assigner.pt`.

### 3.2 Wiring into `company`

In `models/focus_pipeline.py::_assign_learned_with_attn`, before calling `company_pick`, multiply the cross-attention logits by `p_header`:

```
attn_w[company_row, :] += log_zone_posterior[:, HEADER]   # additive in log-space
```

This converts `_COMPANY_HEADER_RE` (a denylist of known distractors) into a soft prior over *every* line — including distractors the regex authors never enumerated. The argmax + `_company_span` chain that follows is bit-exact when `p_header[i]` is uniform, so the change is a no-op on receipts where the prior abstains and a strict improvement on receipts where it concentrates.

### 3.3 Wiring into `total`

In `models/total_arithmetic.py`, restrict the candidate-money set to lines with `p_totals > cfg.totals_zone_floor` (default 0.5). Two consequences, both intentional:
- The phone-number / registration-number false positives in the header zone disappear from the arithmetic enumeration without any new regex.
- When the totals zone is short and dense (4–8 lines), the `±2¢` tolerance of `total_arithmetic.solve` becomes far less ambiguous because the search space shrinks an order of magnitude.

In `total_post.py`, the same gate is applied to the regex argmax: when `p_totals[i] < 0.2`, the line is dropped from candidates.

## 4. Why each ingredient, and why not its competitors

**Why an HMM and not a learned 2D LayoutLM-style positional model.** LayoutLMv3 obtains zone awareness through 2D positional embeddings and 12 transformer layers — at the parameter cost the project explicitly rejects. The 3-state forward–backward decode reproduces the *only* property of LayoutLM that this pipeline uses (monotone vertical zoning of a single-column thermal receipt) at ~100 parameters. The receipts in SROIE-Malaysia are single-column, so the multi-column generality LayoutLM buys is not consumed by the eval split.

**Why HMM and not a CRF.** A linear-chain CRF would buy a richer transition model but requires a learned transition matrix and gradient-based fit. The constraint here is structural, not statistical: the transitions `header→items→totals` are mandatory and acyclic. Hard-coding `T[h,h]=T[h,i]=T[i,i]=T[i,t]=T[t,t]=1, else 0` removes a learned object and makes the prior auditable in `make check`.

**Why a generative zone classifier and not a discriminative gate per field.** A discriminative gate (one binary classifier "is this the company line?", another "is this the total line?") would need two separate supervised heads and would not enforce the relational constraint that `company.zone < items.zone < total.zone` for the same receipt. The HMM enforces that constraint structurally — it is mathematically impossible for the decoded posterior to place company below total.

**Why not a vision/CNN zone classifier on the receipt crop.** A CNN over the page image would push parameter count, require image tensors at inference (currently the assigner is text-only at this dispatch point), and break the mypy-strict `≤2 args` contract on `_assign_learned_with_attn` without rerouting kv. The text-only HMM gives most of the benefit because y-coordinate plus four boolean keyword indicators already separate the three zones at >95% per-line accuracy on SROIE — verifiable on `results/split.json` train fold before any GPU run.

**Why not extending the FOCUS-C span head with a "header gate" output.** That would require retraining the assigner (the agent has repeatedly hit "shipped `assigner.pt` predates new head" in PRs #119, #123). The HMM is fit by closed-form EM on text features, never touches `assigner.pt`, and ships the eval improvement on CPU without `make all`.

**Why not a merchant gazetteer for company and a separate reranker for total.** Both are valid orthogonal moves but they do not share machinery. The point of this PR is one mechanism that lifts both fields. Gazetteer/reranker can be added in a *follow-up* PR that consumes `ZonePosterior` as a feature.

**Why not lower thresholds further (the PR #125 path).** Lowering `total_confidence_threshold` from 0.55 to 0.35 trades precision for recall and hides the underlying signal-quality problem. The zone prior raises *signal*, so thresholds can move back up — `focus_company_confidence_threshold` should be *raised* from 0.30 to 0.40 in the same PR, with the increase justified by the prior's contribution.

## 5. Expected F1 movement and where it comes from

Decompose the failures already documented in PRs #119/#123/#128:

- Company errors caused by selecting `TAX INVOICE`/`WELCOME`/registration-number lines: ~6–8 receipts in the 63-image test split (≈10–13% of cases). The header-zone posterior makes those lines impossible to win the company argmax. Expected: `pipeline_f1_company` 0.82 → 0.90 ± seed noise.
- Total errors caused by phone-number/regid hallucination in the arithmetic solver: 3–5 receipts. Total errors caused by picking subtotal as total: another 3–5. The totals-zone gate eliminates the first class entirely; the arithmetic identity in PR #128 already eliminates the second when the search space is clean. Expected: `pipeline_f1_total` 0.86 → 0.92.

Neither number is verified without GPU; the PR must include the run-id of a real eval and not a placeholder fixture, or it does not merge.

## 6. Required changes — exact file list

- `core/types.py` — add `ZonePosterior`, `ZoneConfig` dataclasses (3-in dataclass, ok).
- `core/config.py` — register `zone_prior_enabled`, `totals_zone_floor`, `header_zone_floor`, validate.
- `configs/default.json` — defaults `enabled=true`, `totals_zone_floor=0.5`, `header_zone_floor=0.4`.
- `models/zone_prior.py` — new, ≤120 LOC, the EM fit lives in `data/zone_prior_fit.py` so the inference module stays small.
- `data/zone_prior_fit.py` — new, fits parameters offline against `results/split.json` train fold, persists to `results/zone_prior.json` (fixture-allowed per `AGENTS.md`).
- `models/focus_pipeline.py` — additive log-prior on the company row of `attn_w`; raise `focus_company_confidence_threshold` default to 0.40.
- `models/total_arithmetic.py` — filter candidate money lines by `p_totals` before enumerating identities.
- `models/total_post.py` — same gate applied to the regex argmax fallback.
- `core/error_metrics.py` — add `zone_violation` error class so we can see when company is selected outside the header zone or total outside the totals zone (regression detector).
- `tests/models/test_zone_prior.py` — synthetic 3-zone receipt, asserts forward–backward decode is monotone.
- `tests/stages/test_company_total_zone_disjoint.py` — on every fixture, asserts `argmax_y(company) < argmax_y(total)`. This is the integration test that would have caught PR #119's silent failure if it had existed at the time.
- `docs/TRACKING.md` — register the four new `\VAR{}` keys (`zone_prior_train_acc`, `zone_violation_count_company`, `zone_violation_count_total`, `pipeline_f1_total_zone_gated`).
- `docs/bugs.md` — Bug-N entry for "head trained but not on dispatch path", with the disjoint-zone test as its guard.

## 7. What this prompt is *not* asking for

- Do not retrain `assigner.pt`. The HMM is fit on CPU.
- Do not add a vision encoder, a GNN, a gazetteer, or an LLM call. They are downstream PRs.
- Do not lower any threshold to compensate for a regression. The prior raises signal; thresholds go up, not down.
- Do not ship with placeholder fixtures. Acceptance is a real `make all` run-id pasted into the PR body, with `runs/<id>/metrics/extended_metrics.json` diff against the pre-PR baseline run-id for both fields.
- Do not split `models/zone_prior.py` for "readability" if it stays under 166 LOC.