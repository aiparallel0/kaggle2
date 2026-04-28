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
