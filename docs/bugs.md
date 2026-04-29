# kaggle2 — F1-destroying bug ledger

This is the append-only ledger of subtle bugs that destroy F1 on SROIE
KIE pipelines, each guarded in code. Do not remove or relax these
guards; reroute them when callers are renamed.

## Bug 1: lm_head weight deduplication

safetensors drops tied weights — the `lm_head` must be explicitly
re-initialised after model resize when `tie_word_embeddings=False`.

## Bug 2: Wrong decoder_start_token_id

String-form tokeniser produces incorrect decoder start token; the
integer ID must be resolved explicitly from the tokeniser vocabulary.

## Bug 3: token2json list return

CORD-style multi-page output returns lists; merge prefers longest
non-empty value per field to avoid silent truncation.

## Bug 4: fp16 gradient overflow

bf16 on Ampere+, else fp16 + max_grad_norm to prevent gradient
explosion during DONUT fine-tuning.

## Bug 5: YOLO imgsz mismatch

Inference default ≠ training size; train/eval parity asserted via
`pipeline_meta.json` to prevent silent degradation.

## Bug 6: TrOCR undertrained

<5 epochs produces all-empty outputs; minimum epoch guard ensures
convergence.

## Bug 7: Val == Test leakage

Physically separate splits, persisted to disk in `results/split.json`
to prevent accidental data leakage across runs.

## Bug 8: YOLO project path resolution

ultralytics ≥8.3 relative-path bug; paths are resolved to absolute
before YOLO instantiation.

## Bug 9: Stale generation_config on reload

eval_F1 ≡ 0 with healthy eval_loss; `Seq2SeqTrainer(predict_with_generate=True)`
reads the snapshot, not live overrides on `model.config`. Guard is now
symmetric across DONUT and TrOCR trainers via
`models/gen_config.py::_persist_generation_config`, with a disk-assertion
that raises `TrainError` on any mismatch.

## Bug 10: tie_word_embeddings=False subtlety

Post-resize `lm_head` must be re-initialised explicitly when
`tie_word_embeddings=False`; the trainer asserts this invariant.

## Bug 11: num_items_in_batch kwargs leak

transformers ≥4.48 passes `num_items_in_batch` to `SwinModel.forward`;
guarded via `accepts_loss_kwargs=False` in trainer config.

## Bug 12: Outer wrapper flattening in token2json

Outer `<s_sroie>` wrapper in `token2json` causes per-field F1 = 0 with
healthy eval_loss; `_flatten_token2json` recursively unwraps the root tag.

## Bug 13: Warmup-steps-vs-ratio precedence

HF `Trainer` warmup_steps overrides warmup_ratio; we force
`warmup_steps=0` when `warmup_ratio > 0` to get the expected behaviour.

## Bug 14: canonical-SROIE entities/ poisoned by Task-1 box files

(PR #90): The HuggingFace `Metric-AI/icdar_sroie` mirror's `entities/`
field accidentally contained Task-1 OCR-box payloads instead of Task-3
KIE entity strings on a subset of records, silently inflating
canonical-347 F1 in early runs. The loader in `data/sroie_canonical.py`
now validates `entities` is a JSON object with the four KIE field keys
(`company`, `date`, `address`, `total`) before accepting a record,
raising `DataError` on shape mismatch.

## Bug 18: One-sided MIL pos-mass loss + disabled FOCUS flags

**Mechanism.** The legacy assigner loss is
``L_f = -log Σ_{i∈T_f} softmax(A)_{f,i}``. It rewards mass on positives
but never penalises mass on boilerplate, so the assigner over-merges
``INV NO`` / ``CASHIER`` / ``TEL`` / ``BRN`` / ``GST`` / ``TAX INVOICE``
into address, and ``SUBTOTAL`` / ``TAX`` / ``CHANGE`` / ``ROUNDING``
into total. PRs #106 (FOCUS-A span head) and #107 (FOCUS-T/C +
priors_v4) shipped the architectural fix as opt-in, but
``configs/default.json`` left every ``focus_*`` flag and ``priors_v4``
at False — the shipped run never invoked the new heads. Two failure
modes stacked: (a) the loss had no "stop-here" signal, (b) the
configured paper variant was ``focus`` but the architecture was off.

**F1 impact.** Address P=0.386, R=0.613, F1=0.708, EM=0.029
(P≪R is the over-merge signature). Total F1=0.703, EM=0.703.
Pipeline F1 sits below DONUT (0.818).

**Guard.** Three guards, all enforced in code:
1. `core/config.py::_validate_focus_flags` raises
   :class:`core.errors.ConfigError` at load time when
   ``paper_variant=='focus'`` and any ``focus_*`` sub-flag is False, or
   when ``focus_total_enabled`` and not ``priors_v4``, or when
   ``focus_enabled`` and ``n_priors < 20``. This is the AGENTS.md
   "no silent placeholders" invariant applied to architecture flags.
2. `models/assigner_loss.py::composite_field_loss` replaces the
   one-sided NLL with ``L_pos + λ_ctkr·L_ctkr + λ_iou·L_iou_attn``.
   CTKR is contrastive top-K repulsion against the *weakest* gold line
   — sparse, adaptive, and margin-referenced to ``a_min``, so long
   addresses with thin per-line mass still get a usable margin and
   short ones get a tighter one. Soft-IoU on the row-max-normalised
   attention is the differentiable analogue of token-F1 at line-mask
   granularity, closing the loss/metric gap that plain pos-mass NLL
   has by construction. Priors_v3 distractor bits act as a tie-breaker
   inside top-K only — never as a stacked penalty.
3. `models/postprocess_address.py::normalize_address_focus` is applied
   symmetrically to pred AND gold inside `eval_pipeline._nt`,
   collapsing whitespace, stripping ``,.:;`` from non-numeric tokens
   (postcodes / phones / lot numbers preserved), and casefolding so
   comma/period drift does not destroy F1 and EM on receipts whose
   line set is already correct.

The guard is `bug_flags["bug_18"]`; flipping it False is the
documented escape hatch for replay runs that need the legacy loss.

## Bug 19: Per-field heads optimised in isolation — no relational zone prior

**Mechanism.** Every dispatch path in `models/focus_pipeline.py` treated
each field independently: `company_pick` argmaxed over *all* OCR lines
with only a denylist (`_COMPANY_HEADER_RE`) keeping bottom-of-receipt
boilerplate (`CHANGE`, `CASH`, `THANK YOU`) out of scope, and
`total_arithmetic.solve` enumerated every parsed money value on the
receipt — including phone numbers (`03-1234.5678`) and registration
suffixes (`(559208-M)`) sitting in the *header* zone that the regex
in `total_post.py` accepted as numeric. The two failures were the
same failure mirrored: `company` leaked downward into the totals zone;
`total` leaked upward into the header zone. The signal needed for both
— a single 3-class posterior `P(zone ∈ {header, items, totals} | line_i)`
— did not exist anywhere in `priors_v4`, `AttentionAssigner`, or
`total_arithmetic`. A second, structural reason this stayed undetected:
no integration test ever exercised `company` and `total` *on the same
receipt* and asserted they pick lines from disjoint y-bands, so a head
trained but never invoked at inference (the PR #119 silent-failure
class) could ship for an entire release cycle.

**F1 impact.** ~6–8 receipts in the 63-image test split land
`company` on a header-distractor line (`TAX INVOICE`, `WELCOME`, regid);
3–5 receipts land `total` on a header-zone phone/regid number, plus
3–5 on `subtotal`. Estimated `pipeline_f1_company` 0.82 → 0.90 and
`pipeline_f1_total` 0.86 → 0.92 once the prior is wired.

**Guard.** Three guards, all enforced in code:
1. `models/zone_prior.py::decode_zone_posterior` is a fixed-topology
   3-state HMM (header → items → totals) decoded by forward–backward;
   transitions are hard-coded for monotonicity, so back-transitions
   are *mathematically impossible* and the relational invariant
   `argmax_y(p_header) < argmax_y(p_total)` holds at the posterior
   level on every receipt the prior touches.
2. `models/focus_pipeline.py::_assign_learned_with_attn` adds
   `log p_header` to the company row of `attn_w` (additive in
   log-space; bit-exact when the prior is uniform) and abstains on
   `company_pick` whose `p_header < zone_cfg.header_zone_floor`. The
   total dispatch filters `total_arithmetic.solve` candidates by
   `p_total >= zone_cfg.totals_zone_floor` and routes the regex-argmax
   fallback through `models.total_post.apply_zone_gate`, dropping
   header-zone numerics from the candidate set.
3. `tests/stages/test_company_total_zone_disjoint.py` is the
   integration test that pins the disjoint-zone contract: on every
   fixture-shaped receipt, `argmax_y(p_header) < argmax_y(p_total)`.
   This is the test that would have caught PR #119's silent failure
   if it had existed at the time, and it now fires before any GPU
   work as part of `make test`.

The guard is implicit (no `bug_19` flag): the zone HMM is fit by
closed-form EM in `data/zone_prior_fit.py`, ships its parameters as
`results/zone_prior.json` (fixture-allowed), and never touches
`assigner.pt`. The escape hatch is `zone_prior_enabled=false` in
`configs/default.json`, which restores the legacy bit-for-bit
dispatch.
