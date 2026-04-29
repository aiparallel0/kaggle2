# Bug 9 Postmortem — Stale `generation_config.json` Driving Pipeline F1 → 0.0

Project: **kaggle2** — *End-to-End vs. Pipeline Receipt KIE on SROIE* (IEEE/ICDAR).
Scope: explains what "Bug 9" is, why it keeps relapsing, what PR #49 fixed, what
gaps remain, and how to unstick a run without retraining using
[`scripts/repin_generation_config.py`](../scripts/repin_generation_config.py).

---

## 1. Symptom

The guardrail in `models/eval_pipeline.py` fires with:

> `Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5), TrOCR undertrained (Bug 6), or stale generation_config decoder_start_token_id (Bug 9)`

while each component in isolation looks healthy:

| Component           | Metric                     | Value    |
| ------------------- | -------------------------- | -------- |
| DONUT end-to-end    | `eval_f1`                  | 0.8191   |
| TrOCR (crop-level)  | `eval_f1` (training peak)  | 0.967    |
| YOLO detector       | `mAP50`                    | 0.987    |
| **Pipeline E2E**    | `f1`                       | **0.0**  |

So DONUT, YOLO, and TrOCR-as-a-unit all learned. The stitched pipeline
still produces empty or garbage sequences — the classic Bug 9 signature.

## 2. Root cause

`Seq2SeqTrainer(predict_with_generate=True)` reads the *snapshot*
`model.generation_config` at eval time, **not** live overrides on
`model.config`. When `load_best_model_at_end=True`, the trainer restores
the best epoch's checkpoint *in place* — including its stale
`generation_config` that was serialised before our Bug-1/6/7 patches
mutated the token IDs. The ordering hazard is subtle:

```
trainer.train()             # best ckpt restored → gc reset to pre-patch values
trainer.save_model(out_dir) # writes generation_config.json with the stale ids
```

`Hugging Face` writes `generation_config.json` as part of `save_model`, so
any re-pin done *before* `save_model` can be silently overwritten unless
the in-memory `generation_config` object itself already has the right
IDs. PR #49 handles this by re-pinning **before** `save_model` and then
writing again **after** `save_model` (belt-and-braces).

### The three IDs that matter

| Checkpoint | `decoder_start_token_id`       | `eos_token_id`                 | `pad_token_id`           |
| ---------- | ------------------------------ | ------------------------------ | ------------------------ |
| DONUT      | `<s_sroie>` (id from tokenizer) | `</s_sroie>` (id from tokenizer) | `proc.tokenizer.pad_token_id` |
| TrOCR      | `cls_token_id` (typically 0)   | `sep_token_id` (typically 2)   | `pad_token_id` (typically 1) |

Numbers depend on the tokenizer present in `results/{donut,trocr}/` —
**always resolve IDs from the on-disk tokenizer**, never hard-code them.
(The frequently-cited `2/1/2` triple is the DONUT/mBART default, not the
TrOCR/RoBERTa one.)

In addition to the three IDs above, mBART-style `forced_bos_token_id`
and `forced_eos_token_id` leak into the `generation_config` and must be
explicitly nulled; otherwise generation is constrained to a token that
is not in the SROIE target vocabulary.

## 3. Relapse history

| PR   | Title                                                                       | Coverage                                    |
| ---- | --------------------------------------------------------------------------- | ------------------------------------------- |
| #29  | Fix generation config mismatch in Donut model training and eval             | DONUT only                                  |
| #34  | …fix TrOCR Bug 9, 384-dim hardcode, YOLO class misnomer                     | TrOCR (single-site re-pin)                  |
| #45  | Fix Bug 9: stale generation_config causing eval F1=0.0 in pipeline          | Post-`save_model` re-pin, no disk assertion |
| #49  | Bug 9: symmetric generation_config re-pin + TrainError assertion            | Symmetric + `TrainError` disk round-trip    |

PR #49 introduced `models/gen_config.py::_persist_generation_config`
which is the canonical fix for the training path.

## 4. Residual gaps (not covered by #49)

Even with #49 merged, a pipeline run can still land at F1=0.0 through
paths that aren't a literal "wrong `decoder_start_token_id` on disk":

1. **Tokenizer drift at eval load.** `models/detect.py` passes
   `decoder_start_token_id=trocr_proc.tokenizer.cls_token_id` to
   `model.generate()`. If the processor saved to `results/trocr/` does
   not match the tokenizer the model was trained with, the IDs disagree
   at inference time even though `generation_config.json` itself is
   valid. There is no load-time assertion for this.
2. **Silent exception swallowing** in `eval_pipeline.py`
   (`except (OSError, RuntimeError, ValueError): ...`) masks per-receipt
   crashes and produces F1=0.0 with zero log trace.
3. **Checkpoint-path drift.** A stale `results/trocr/` left over from an
   earlier run can shadow the newest write (e.g. when `save_model` wrote
   into a `checkpoint-best/` subdirectory rather than the top-level).

These are tracked separately from this postmortem.

## 5. Diagnostic decision tree

Before another speculative "Bug 9 fix", open the crashed run's
`results/pipeline_metrics.json` and branch on what you see:

| Condition                                                                 | Root cause         |
| ------------------------------------------------------------------------- | ------------------ |
| `per_receipt_error_fraction == 1.0`                                       | §4.2 silent excepts (not Bug 9) |
| `empty_detection_fraction` high and `rulebased_f1 == 0`                   | Bug 5 (YOLO imgsz) |
| `rulebased_f1 > 0` but `assigner_f1 == 0`                                 | Assigner bug       |
| `rulebased_f1 == 0`, errors low, detections fine, **and** `trocr_empty_prediction_fraction` ≈ 1.0 | Bug 9 — run the repair script (§6) |

## 6. Workaround — repair a corrupted checkpoint without retraining

When the root cause **is** Bug 9 on a checkpoint you already trained
(i.e. `results/trocr/generation_config.json` has the wrong
`decoder_start_token_id`), do not retrain. The checkpoint weights are
correct; only the JSON sidecar is wrong. Use:

```bash
# Dry-run: show what would change for the TrOCR checkpoint.
python scripts/repin_generation_config.py results/trocr --dry-run

# Repair in place (auto-detects TrOCR vs DONUT from the tokenizer).
python scripts/repin_generation_config.py results/trocr
python scripts/repin_generation_config.py results/donut

# Force a specific kind or manual IDs if auto-detection is wrong.
python scripts/repin_generation_config.py results/trocr \
  --kind trocr \
  --decoder-start-token-id 0 --eos-token-id 2 --pad-token-id 1
```

The script:

1. Loads the tokenizer from the *same* checkpoint directory so the IDs
   come from ground truth, not a hard-coded table.
2. Writes a timestamped backup to
   `generation_config.json.bak-YYYYMMDDHHMMSS` before touching the
   original (idempotent: repeated runs keep adding backups).
3. Rewrites `generation_config.json` with the correct
   `decoder_start_token_id`, `eos_token_id`, `pad_token_id`,
   `bos_token_id`, and **nulls** `forced_bos_token_id` and
   `forced_eos_token_id`.
4. Re-reads the file and raises if the round-trip disagrees — the same
   contract as `models/gen_config.py::_persist_generation_config`.
5. Prints a before/after diff so the fix is auditable in logs.

After repair, re-run the eval stage only:

```bash
python main.py --stage eval --config config.json
```

No retraining required.

## 7. Related files

* `models/gen_config.py` — canonical train-time guard (PR #49).
* `models/donut_train.py`, `models/trocr_train.py` — call sites.
* `models/eval_pipeline.py` — where the Bug-9 guardrail fires.
* `tests/models/test_donut_eval_diag_artifact.py` — regression tests.
* `scripts/repin_generation_config.py` — post-hoc repair tool (this PR).
