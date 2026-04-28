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
