# kaggle2

Context-window-first receipt KIE. DONUT vs YOLO+TrOCR+Attention on SROIE.
18 files, ≤166 LOC cap, 2-in/1-out contracts, mypy-as-test-suite.
Replaces 34K-line Python monolith. Trains both architectures, evaluates on
63-image split, generates IEEE-regional LaTeX paper.

## Expected F1 — honest range, no guarantees

Published DONUT-on-SROIE results land in **0.73 – 0.90** depending on
epochs, resolution, auxiliary data, and seed. The tightened configuration
in this repo (15 epochs, cosine schedule, beam-search decoding, best-F1
checkpoint selection) typically lands in **0.78 – 0.88** on an RTX 4090 /
A6000. **No specific F1 number can be guaranteed by code changes alone**:
F1 is a stochastic training outcome that depends on GPU availability,
Hugging Face weight snapshots at download time, and SROIE label noise.

`validate_f1()` in `main.py` enforces two levels:

| level            | threshold                           | behaviour |
|------------------|-------------------------------------|-----------|
| hard floor       | DONUT < 0.50 / pipeline == 0.0      | raises `TrainError` (indicates a bug, not underperformance) |
| soft expectation | `config.expected_f1_warn` (def 0.75) | logs a WARNING, does not fail the run |

## Architecture

Two KIE architectures are compared on the
[SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) receipt dataset:

| Architecture | Components | Assignment |
|---|---|---|
| **DONUT** (end-to-end) | Swin encoder → BART decoder | Implicit (seq2seq) |
| **Pipeline** (detect-then-read) | YOLOv8 → TrOCR → AttentionAssigner | Learned cross-attention |

The pipeline also evaluates a **rule-based spatial baseline** (position +
regex heuristics) to quantify the assigner's contribution.

### Module layout

```
core/         config, types, errors, shared metrics, seed_everything
data/         SROIE download, split (persisted), crop extraction
models/       donut_train, donut_eval, yolo_train, trocr_train,
              assigner_train, attention_assign, pipeline_eval
report/       LaTeX template injection, references
scripts/      vastai_bootstrap.sh — one-shot install+check
main.py       orchestrator (--stage train | eval | paper | all)
```

## Quick start (vast.ai — three copy-pastes)

1. Rent a **PyTorch** instance with **≥24 GB GPU** (RTX 4090, 3090, A6000,
   A100) and **≥50 GB disk**. Use the vast.ai web UI; do not bother with
   the CLI / SSH.
2. Open the instance's built-in terminal or Jupyter terminal.
3. Paste:

   ```bash
   cd /workspace
   git clone https://github.com/aiparallel0/kaggle2.git && cd kaggle2
   bash scripts/vastai_bootstrap.sh
   make all
   ```

`make all` runs `check → test → train → eval → paper`. On a single RTX
4090 the DONUT stage takes ≈ 45 min at 15 epochs; YOLO+TrOCR+Attention
takes ≈ 30 min. Intermediate artefacts land in `./results/`, the final
paper is `report/paper_filled.pdf`.

Partial runs:

```bash
make check         # ruff + mypy --strict + import smoke
make test          # pytest (no GPU needed)
python main.py --stage train
python main.py --stage eval
python main.py --stage paper
```

The split (500 / 63 / 63) is persisted to `results/split.json` on the
first train run, so a later `--stage eval` in a separate shell sees the
exact same test set — no silent drift.

## Local (non-vast.ai) run

```bash
pip install -r requirements.txt
make all
```

Or via Docker:

```bash
docker build -t kaggle2 .
docker run --gpus all -v $(pwd)/results:/app/results kaggle2
```

## Configuration

All hyperparameters live in `config.json`. F1-affecting knobs:

| Parameter | Default | Effect on expected F1 |
|---|---|---|
| `epochs_donut` | 15 | Longer training → higher F1, diminishing returns past 15. |
| `image_size` | [1280, 960] | Higher resolution → better address/total recognition; more VRAM. |
| `num_beams` | 4 | Beam search typically gains 2–4 F1 points over greedy. |
| `lr_scheduler_type` | cosine | Cosine > linear for short SROIE runs. |
| `warmup_ratio` | 0.1 | Stabilises early loss on small data. |
| `gradient_checkpointing` | true | Lets batch 8 × 1280 × 960 fit in 24 GB. |
| `patience` | 3 | EarlyStopping on plateau of eval F1. |
| `precision` | bf16 | bf16 on Ampere+; fp16 with grad-clip otherwise (Bug 4). |
| `yolo_img_size` | 512 | MUST match training and inference (Bug 5). |
| `epochs_trocr` | 10 | Floor of 5 enforced in config.py (Bug 6). |
| `expected_f1_warn` | 0.75 | Soft WARN threshold (non-fatal). |

## F1-destroying bugs (all guarded in code)

1. lm_head weight deduplication (safetensors drops tied weights)
2. Wrong decoder_start_token_id (string-form tokeniser)
3. token2json list return (CORD-style multi-page output); merge prefers
   longest non-empty value per field
4. fp16 gradient overflow (bf16 on Ampere+, else fp16 + max_grad_norm)
5. YOLO imgsz mismatch (inference default ≠ training size)
6. TrOCR undertrained (<5 epochs produces all-empty outputs)
7. Val == Test leakage (physically separate splits, persisted to disk)

## Testing

```bash
make test        # pytest (split persistence, metrics, configs, etc.)
make check       # ruff + mypy --strict + import smoke
```

## Reproducibility

`core.seed.seed_everything(config.seed)` is called at startup and seeds
`random`, `numpy`, `torch`, and CUDA (with `cudnn.deterministic=True`).
Combined with HF `Trainer(seed=config.seed, data_seed=config.seed)`, a
persisted data split, and a deterministic DataLoader
(`worker_init_fn` + seeded `torch.Generator`), runs are bit-for-bit
reproducible on identical hardware.

## License

See [LICENSE](LICENSE).
