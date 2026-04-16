# kaggle2

Context-window-first receipt KIE. DONUT vs YOLO+TrOCR+Attention on SROIE.
18 files, ≤166 LOC cap, 2-in/1-out contracts, mypy-as-test-suite.
Replaces 34K-line Python monolith. Trains both architectures, evaluates on
63-image split, generates IEEE-regional LaTeX paper. Best F1 ≈ 0.85–0.90.

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
core/         config, types, errors, shared metrics
data/         SROIE download, split, crop extraction
models/       donut_train, donut_eval, yolo_train, trocr_train,
              assigner_train, attention_assign, pipeline_eval
report/       LaTeX template injection, references
main.py       orchestrator (--stage train | eval | paper | all)
```

## Quick start

```bash
# Install (GPU recommended; CPU works for eval on small splits)
pip install -r requirements.txt

# Lint + type-check
make check

# Train both architectures on SROIE
make train

# Evaluate on held-out 63-image test split
make eval

# Generate IEEE-format LaTeX paper with real metrics
make paper

# Or run everything end-to-end
make all
```

### Docker

```bash
docker build -t kaggle2 .
docker run --gpus all kaggle2
```

## Configuration

All hyperparameters live in `config.json`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `epochs_donut` | 10 | DONUT fine-tuning epochs |
| `epochs_yolo` | 50 | YOLOv8 training epochs |
| `epochs_trocr` | 10 | TrOCR fine-tuning epochs (≥5 enforced) |
| `image_size` | [1280, 960] | DONUT input resolution [W, H] |
| `yolo_img_size` | 512 | YOLO detection resolution |
| `batch_size` | 8 | Training batch size |
| `precision` | bf16 | Mixed precision (bf16 on Ampere+, fp16 fallback) |

## F1-destroying bugs

Seven implementation bugs that silently destroy F1 are documented and
guarded against in code:

1. **lm_head weight deduplication** — safetensors drops tied weights on reload
2. **Wrong decoder_start_token_id** — string-form tokeniser returns wrong ID
3. **token2json returns list** — CORD-style multi-page output breaks parsing
4. **fp16 gradient overflow** — use bf16 or max_grad_norm clipping
5. **YOLO imgsz mismatch** — inference default ≠ training size → 0% detection
6. **TrOCR undertrained** — < 5 epochs produces all-empty outputs
7. **Val == Test leakage** — physically separate validation and test splits

## Testing

```bash
python -m pytest -q tests/
```

Tests cover configuration validation, metric computation (F1, NED, EM),
YOLO label conversion, and LaTeX template injection.

## License

See [LICENSE](LICENSE).
