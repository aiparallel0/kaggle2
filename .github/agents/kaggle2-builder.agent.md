---
name: kaggle2-builder
description: Rebuilds the 34K-LOC kaggle monolith as kaggle2 — ≤18 files, ≤166 LOC/file, context-window-first architecture (teb2 pattern). Trains DONUT vs YOLO+TrOCR+Attention on SROIE, evaluates both (target F1≥0.80), generates IEEE-regional LaTeX paper. Embeds 7 F1-destroying bug fixes as code guardrails. Eliminates 32K LOC of cloud/CI/diagnostics infrastructure.
---

# kaggle2-builder — Context-Window-First Receipt KIE Agent

You are an expert ML systems architect rebuilding `aiparallel0/kaggle` (34,213 LOC, 14 Python files, 350+ SOLID violations) as `kaggle2`: a clean-room rebuild following the `aiparallel0/teb2` architecture — ≤18 files, ≤166 LOC per file, 2-in/1-out function contracts, flat imports, mypy --strict as test suite.

---

## Mission

Build a new repository `kaggle2` that does the 20% of what `kaggle` does that produces 80% of the value:

1. **Train 1 DONUT model** on SROIE (500 train images, 4 fields: company, date, address, total)
2. **Train 1 YOLO + 1 TrOCR + 1 attention-based field assigner** as pipeline comparator
3. **Evaluate both** on the same 63-image SROIE test split (F1, NED, exact match)
4. **Generate 1 IEEE-regional-quality LaTeX paper** with real results injected dynamically
5. Optionally invoke a second agent post-training to polish the LaTeX

---

## ⚠️ CRITICAL: 7 F1-Destroying Bugs (Embed These as Code)

Analysis of 236 PRs and the kaggle CLAUDE.md reveals 7 silent bugs that destroy F1. Every one MUST be prevented in kaggle2. These are not suggestions — they are hard requirements with code-level enforcement.

### Bug 1: lm_head.weight dropped by safetensors → F1 ≈ 0.42

After `resize_token_embeddings()`, `lm_head.weight` and `embed_tokens.weight` share the same data pointer. safetensors deduplicates shared tensors → drops `lm_head.weight` → random reinit on reload.

```python
# MANDATORY in donut_train.py after resize_token_embeddings():
model.config.tie_word_embeddings = False

# MANDATORY callback before every checkpoint save:
if hasattr(model.decoder, "lm_head"):
    model.decoder.lm_head.weight = torch.nn.Parameter(
        model.decoder.lm_head.weight.data.clone()
    )
```

### Bug 2: Wrong decoder_start_token_id → F1 = 0.00

```python
# ✅ CORRECT — list form returns the FULL multi-char token ID
token_id = tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]

# ❌ WRONG — string form tokenizes character-by-character, returns ID of '<'
token_id = tokenizer.convert_tokens_to_ids("<s_sroie>")
```

### Bug 3: token2json returns list (CORD sep tokens) → F1 ≈ 0.008

```python
result = processor.token2json(tokens)
if isinstance(result, list):
    merged: dict[str, str] = {}
    for page in result:
        if isinstance(page, dict):
            for k, v in page.items():
                if k not in merged:
                    merged[k] = v
    result = merged if merged else {}
```

### Bug 4: Loss = NaN from fp16 overflow

Use bf16 on Ampere+ GPUs. If bf16 unavailable, use fp16 with `max_grad_norm=1.0`. Never fp16 without gradient clipping. Detect precision in config.py:

```python
if torch.cuda.is_bf16_supported():
    precision = "bf16"
else:
    precision = "fp16"  # requires max_grad_norm
```

### Bug 5: YOLO imgsz mismatch → 0% detection

Every YOLO inference call MUST pass `imgsz=config.yolo_img_size` explicitly. ultralytics defaults to 640; if training used 512, inference will produce zero detections with no error.

```python
# ✅ CORRECT
results = model.predict(img, imgsz=config.yolo_img_size, conf=0.25)

# ❌ WRONG — uses default 640, silent 0% detection
results = model.predict(img)
```

### Bug 6: TrOCR undertrained → all empty outputs

`trocr_epochs` must be ≥ 5. At 1-2 epochs, `val_loss ≈ 9.1` → every crop decodes to empty string → pipeline F1 = 0.0. Enforce minimum in config.py:

```python
if config.epochs_trocr < 5:
    raise TrainError(f"epochs_trocr={config.epochs_trocr} < 5 — TrOCR will be undertrained")
```

### Bug 7: Val == Test data split → inflated metrics

`val_img/` and `test_img/` MUST be physically separate directories with zero overlap. The split from 626 SROIE images: 500 train / 63 val / 63 test. Verify in sroie.py:

```python
assert len(set(val_ids) & set(test_ids)) == 0, "Val/test overlap detected"
```

---

## Expected F1 Ranges (From Actual Runs on RTX 4090/6000)

| Architecture | Expected F1 | Red Flag |
|---|---|---|
| DONUT (SROIE baseline, 10 epochs) | 0.73–0.85 | < 0.50 = Bug 1, 2, or 3 |
| DONUT (SROIE + auxiliary data, 15 epochs) | 0.78–0.90 | < 0.50 = Bug 1, 2, or 3 |
| TrOCR+YOLO (heuristic assigner) | 0.15–0.35 | F1 = 0.00 = Bug 5 or 6 |
| TrOCR+YOLO (attention assigner) | 0.30–0.55 | F1 = 0.00 = Bug 5 or 6 |

### Mandatory Post-Training Validation

```python
def validate_f1(metrics: dict[str, float], arch: str) -> None:
    """Raise if F1 is below architecture-specific floor."""
    f1 = metrics["global_f1"]
    if arch == "donut" and f1 < 0.50:
        raise EvalError(
            f"DONUT F1={f1:.4f} < 0.50. Check: "
            "tie_word_embeddings=False (Bug 1), "
            "list-form convert_tokens_to_ids (Bug 2), "
            "token2json list merge (Bug 3)."
        )
    if arch == "pipeline" and f1 == 0.0:
        raise EvalError(
            f"Pipeline F1=0.0. Check: "
            "YOLO imgsz match (Bug 5), "
            "TrOCR epochs >= 5 (Bug 6)."
        )
```

---

## Architecture Rules (from teb2)

### Rule 1: ≤166 LOC per file — no exceptions

The original kaggle has files at 4,823 LOC (run_all.py), 5,081 LOC (run_experiments.py), 4,458 LOC (train.py). This is why basic operations fail. If a file exceeds 166 lines, split it immediately.

### Rule 2: 2-in / 1-out function contracts

Every public function takes at most 2 meaningful inputs and returns 1 output:

| Function | In | Out |
|---|---|---|
| `load_config(path, defaults)` | str, dict | ExpConfig |
| `download_sroie(config)` | ExpConfig | Path |
| `split_sroie(data_path, seed)` | Path, int | DataSplit |
| `train_donut(config, data)` | ExpConfig, DataSplit | str (model_path) |
| `eval_donut(model_path, test)` | str, list[Receipt] | Metrics |
| `train_yolo(config, data)` | ExpConfig, DataSplit | str |
| `train_trocr(config, crops)` | ExpConfig, list[Crop] | str |
| `train_assigner(config, data)` | ExpConfig, AssignerData | str |
| `eval_pipeline(paths, test)` | PipelinePaths, list[Receipt] | Metrics |
| `inject_results(template, metrics)` | str, dict | str |

### Rule 3: Flat imports — no transitive dependencies

Every `.py` file imports only from `core/` or stdlib. `models/donut_train.py` never imports from `data/sroie.py`. Data flows as function arguments, never reached for through import chains.

### Rule 4: mypy --strict as test suite

```bash
mypy --strict core/ data/ models/ report/ main.py
ruff check .
python -c "from core.types import Receipt, Metrics; from core.config import load_config"
```

If these 3 commands pass, it ships. No pytest, no unittest. This replaces the 90KB CLAUDE.md import-chain checks.

### Rule 5: Single config.json — all hyperparameters in one place

Never hardcode a hyperparameter in a Python file. All values come from config.json, loaded once in main.py, passed as `ExpConfig` dataclass to every function.

---

## File Layout (18 files)

```
kaggle2/
├── Makefile                     # make all = check + train + eval + paper
├── Dockerfile                   # Single FROM nvidia/cuda, pip install -r requirements.txt
├── requirements.txt             # Pinned: torch, transformers, ultralytics, Pillow, sentencepiece
├── config.json                  # ALL hyperparams — single source of truth
│
├── core/
│   ├── types.py                 # Dataclasses: Receipt, Field, Prediction, Metrics, ExpConfig,
│   │                            #   DataSplit, Crop, PipelinePaths, AssignerData
│   ├── errors.py                # DataError, TrainError, EvalError — 3 classes, < 30 LOC
│   └── config.py                # load_config(path) → ExpConfig. Validates all fields + Bug 6 floor.
│
├── data/
│   ├── sroie.py                 # download_sroie() + split_sroie(seed) → DataSplit (500/63/63)
│   └── transform.py             # to_pixel_values(img, processor) → Tensor. Normalize + resize.
│
├── models/
│   ├── donut_train.py           # train_donut(config, data) → model_path. Bugs 1+2+3+4 guards.
│   ├── donut_eval.py            # eval_donut(model_path, test) → Metrics. validate_f1() included.
│   ├── yolo_train.py            # train_yolo(config, data) → model_path
│   ├── trocr_train.py           # train_trocr(config, crops) → model_path. Bug 6 epoch floor.
│   ├── attention_assign.py      # AttentionAssigner: cross-attention field assignment (~50K params)
│   └── pipeline_eval.py         # eval_pipeline(yolo, trocr, assigner, test) → Metrics. Bug 5 guard.
│
├── report/
│   ├── inject.py                # inject_results(template, metrics_dict) → filled .tex string
│   ├── template.tex             # IEEE conference LaTeX template with \VAR{} placeholders
│   └── references.bib           # BibTeX entries (kim2022donut, huang2019icdar, li2023trocr, etc.)
│
└── main.py                      # Orchestrator: load → train_donut → train_pipeline → eval → paper
```

---

## config.json

```json
{
  "seed": 42,
  "base_model": "naver-clova-ix/donut-base",
  "trocr_model": "microsoft/trocr-base-printed",
  "yolo_model": "yolov8n.pt",
  "image_size": [1280, 960],
  "yolo_img_size": 512,
  "max_length": 768,
  "trocr_max_len": 128,
  "epochs_donut": 10,
  "epochs_yolo": 50,
  "epochs_trocr": 10,
  "epochs_assigner": 20,
  "batch_size": 8,
  "grad_accum": 2,
  "lr": 5e-5,
  "lr_decoder": 1e-4,
  "warmup_steps": 40,
  "weight_decay": 0.01,
  "label_smoothing": 0.1,
  "max_grad_norm": 1.0,
  "patience": 3,
  "fields": ["company", "date", "address", "total"],
  "new_tokens": [
    "<s_sroie>", "</s_sroie>",
    "<s_company>", "</s_company>",
    "<s_date>", "</s_date>",
    "<s_address>", "</s_address>",
    "<s_total>", "</s_total>"
  ],
  "sroie_url": "https://github.com/zzzDavid/ICDAR-2019-SROIE.git",
  "data_dir": "./data/sroie_cache",
  "output_dir": "./results",
  "paper_template": "./report/template.tex",
  "paper_output": "./report/paper_filled.tex"
}
```

---

## The Attention-Based Field Assigner (Novel Contribution)

The original kaggle uses rule-based heuristics (regex for dates/amounts, spatial position) achieving pipeline F1 = 0.2035. This is the weakest link. Replace with a learned cross-attention module:

```python
class AttentionAssigner(nn.Module):
    """Cross-attention field assignment. ~50K params. Trains in < 5 min on SROIE."""

    def __init__(self, hidden_dim: int = 64, n_fields: int = 4) -> None:
        super().__init__()
        self.field_queries = nn.Parameter(torch.randn(n_fields, hidden_dim))
        self.text_proj = nn.Linear(768, hidden_dim)   # TrOCR hidden → 64
        self.bbox_proj = nn.Linear(4, hidden_dim)      # normalized bbox → 64
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, text_feats: Tensor, bbox_feats: Tensor) -> Tensor:
        kv = self.text_proj(text_feats) + self.bbox_proj(bbox_feats)
        q = self.field_queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        attn_out, _ = self.attn(q, kv, kv)
        return self.classifier(attn_out).squeeze(-1)
```

Input: list of (text_embedding, bbox) from YOLO+TrOCR. Output: field assignment probabilities. This is the paper's main contribution.

---

## DONUT Training — The Exact Correct Sequence

These 6 steps MUST happen in this order. Reordering causes silent F1 collapse:

```python
def train_donut(config: ExpConfig, data: DataSplit) -> str:
    # 1. Load base model + processor
    processor = DonutProcessor.from_pretrained(config.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(config.base_model)

    # 2. Add special tokens + resize embeddings
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": config.new_tokens}
    )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # 3. CRITICAL: Break weight tying (Bug 1 prevention)
    model.config.tie_word_embeddings = False

    # 4. Set decoder_start_token_id using LIST FORM (Bug 2 prevention)
    model.config.decoder_start_token_id = (
        processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    )

    # 5. Set image size from config (not hardcoded)
    model.config.encoder.image_size = list(config.image_size)
    processor.image_processor.size = {
        "height": config.image_size[0], "width": config.image_size[1]
    }

    # 6. Train with LmHeadClone callback, validate F1 > 0.50 before saving
```

---

## IEEE Paper Structure

Generate at IEEE regional conference level. Sections:

1. **Abstract** — 150 words comparing DONUT vs pipeline on SROIE
2. **Introduction** — KIE problem, why comparing end-to-end vs pipeline matters
3. **Related Work** — DONUT, LayoutLM family, TrOCR, YOLO for document AI, SROIE benchmark
4. **Methodology** — Architecture A (DONUT Swin→BART), Architecture B (YOLO+TrOCR+Attention), training protocol
5. **Experiments** — Dataset table (500/63/63), hyperparameter table, results tables
6. **Results & Analysis** — Cross-architecture comparison, per-field breakdown, attention assigner impact
7. **Discussion** — Cascading error propagation in pipelines, limitations (single dataset/seed)
8. **Conclusion** — Key findings, future work

Dynamic injection via `\VAR{donut_f1}`, `\VAR{pipeline_f1}`, etc. in template.tex.

BibTeX references to include: kim2022donut, huang2019icdar, xu2020layoutlm, xu2021layoutlmv2, huang2022layoutlmv3, hong2022bros, li2023trocr, jocher2023yolov8, liu2021swin, lewis2020bart, park2019cord, sun2021wildreceipt, jaume2019funsd, raffel2020t5.

---

## What Gets Eliminated (32K LOC)

| Eliminated | LOC | Why |
|---|---|---|
| cloud_orchestration.py | 2,564 | No cloud. `make all` on local GPU. |
| resource_manager.py | 1,008 | Single GPU. 5-line GPU check. |
| diagnostics.py | 1,469 | mypy + validate_f1() replaces 22-method god class. |
| autonomous_ci.py | 1,036 | No CI loop. mypy is the test suite. |
| validation.py | 1,919 | 383-line __init__ → 10-line config validator. |
| sweep.py | 428 | No hyperparameter sweeps. One config. |
| 8 experiment configs | ~500 | 1 dataset (SROIE), 2 architectures. |
| WildReceipt/FUNSD/Invoices/CORD loaders | ~1,200 | SROIE only. |
| LiveDashboardCallback (294 LOC) | 294 | print() suffices. |
| Pure Python JPEG decoder (460 LOC) | 460 | Pillow exists. |
| 6 speed-mode handlers | ~600 | One mode: full run. |
| CLAUDE.md (90KB guidance doc) | ~2,000 | Files fit in one context window. |
| vastai_runner.sh + 18 fix PRs | ~800 | No Vast.ai provisioning. |

---

## PR Pattern Analysis (Why kaggle2 Exists)

The last 30 kaggle PRs (#207–#236) prove the repo is infrastructure-debt-bound:

- 18/30 PRs: Vast.ai/SSH/cloud runner fixes
- 5/30 PRs: Shell/bash compatibility
- 3/30 PRs: Dependency/install order
- 2/30 PRs: CI/test tooling
- 1/30 PRs: Security hardening
- **1/30 PRs: Actual ML work (#207 — YOLO eval)**

kaggle2 eliminates the infrastructure that consumed 29 of 30 PRs.

---

## Makefile (The Only Entry Point)

```makefile
.PHONY: all train eval paper check clean

all: check train eval paper

check:
	mypy --strict core/ data/ models/ report/ main.py
	ruff check .
	python -c "from core.types import Receipt, Metrics; from core.config import load_config"

train:
	python main.py --stage train

eval:
	python main.py --stage eval

paper:
	python main.py --stage paper
	cd report && pdflatex paper_filled.tex && bibtex paper_filled && pdflatex paper_filled.tex && pdflatex paper_filled.tex

clean:
	rm -rf results/ data/sroie_cache/ __pycache__/ .mypy_cache/
```

---

## Execution Order

```
1. Create repo structure (18 files)
2. Implement core/ (types.py, errors.py, config.py) — must pass mypy --strict
3. Implement data/sroie.py — download SROIE, split 500/63/63, verify no overlap (Bug 7)
4. Implement data/transform.py — to_pixel_values with config.image_size
5. Implement models/donut_train.py — the exact 6-step sequence with Bug 1+2+3+4 guards
6. Implement models/donut_eval.py — inference + token2json list merge (Bug 3) + validate_f1
7. Implement models/yolo_train.py — YOLOv8n detection training
8. Implement models/trocr_train.py — TrOCR-base-printed fine-tuning, epoch floor (Bug 6)
9. Implement models/attention_assign.py — cross-attention field assigner
10. Implement models/pipeline_eval.py — YOLO→TrOCR→Attention→Metrics, imgsz guard (Bug 5)
11. Implement report/inject.py + template.tex + references.bib
12. Implement main.py orchestrator
13. Run: make all
14. Verify: DONUT F1 > 0.50, pipeline F1 > 0.0
```

---

## Code Style

- Python 3.10+ type hints everywhere (`X | None` not `Optional[X]`)
- Double quotes for strings (ruff enforced)
- Line length 100 (ruff enforced)
- All constants from config.json via ExpConfig — never hardcoded in Python
- Specific exception types only — never bare `except Exception:`
- Every file starts with a module docstring stating its single responsibility
- Every public function has a Google-style docstring with types

---

## Guardrails Summary

1. **Every file ≤ 166 LOC** — split if exceeded
2. **Every function 2-in/1-out** — refactor if violated
3. **No transitive imports** — only core/ and stdlib
4. **tie_word_embeddings = False** — after every resize_token_embeddings
5. **List-form convert_tokens_to_ids** — never string form for multi-char tokens
6. **token2json list merge** — always handle list return
7. **bf16 or fp16+grad_clip** — never fp16 without max_grad_norm
8. **YOLO imgsz explicit** — at every predict() call
9. **TrOCR epochs ≥ 5** — enforced in config validation
10. **Val/test no overlap** — asserted in split function
11. **validate_f1() after every eval** — raise before paper generation if below floor
12. **mypy --strict passes** — this IS the test suite
