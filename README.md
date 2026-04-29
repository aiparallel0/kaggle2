# kaggle2

Context-window-first receipt KIE. DONUT vs YOLO+TrOCR+Attention on SROIE.
18 files, ≤166 LOC cap, 2-in/1-out contracts, mypy-as-test-suite.
Replaces 34K-line Python monolith. Trains both architectures, evaluates on
63-image split, generates IEEE-regional LaTeX paper.

## Expected F1 — honest range, no guarantees

Published DONUT-on-SROIE results land in **0.73 – 0.90** depending on
epochs, resolution, auxiliary data, and seed. The tightened configuration
in this repo (30 epochs, cosine schedule, differential encoder/decoder LR,
beam-search decoding, best-F1 checkpoint selection) typically lands in
**0.78 – 0.85** on an RTX 4090 /
A6000. The pipeline (YOLO+TrOCR+Attention) typically achieves
**0.50 – 0.60** F1, with YOLO text-line detection at **0.98 mAP@0.5**.
**No specific F1 number can be guaranteed by code changes alone**:
F1 is a stochastic training outcome that depends on GPU availability,
Hugging Face weight snapshots at download time, and SROIE label noise.

### Two paper variants

The repo bifurcates into two **paper variants** that share one codebase:

| variant     | test set                                               | arms                                                | leaderboard-comparable | template                       |
|-------------|--------------------------------------------------------|-----------------------------------------------------|------------------------|--------------------------------|
| `focus`  | 347 official ICDAR-2019 Task-3 (auto-downloaded)       | DONUT vs YOLO+TrOCR+**Attention assigner**           | **yes**                | `report/template_focus.tex` |
| `baseline`     | 63-image internal (from 500/63/63 of 626 train)        | DONUT vs YOLO+TrOCR+**regex** vs **GT-OCR+regex**    | no                     | `report/template_baseline.tex`    |

The `focus` variant is the **default**.  `data/sroie_canonical.py`
sha256-verifies and extracts the 347 Task-3 images on first run
(primary: `rrc.cvc.uab.es`; fallback: docTR mirror).  Switch variants
via CLI: `python main.py --paper-variant baseline --stage all`.  See
[`report/overleaf/README.md`](report/overleaf/README.md) for one-shot
Overleaf upload instructions.

`validate_f1()` in `main.py` enforces two levels:

| level            | threshold                           | behaviour |
|------------------|-------------------------------------|-----------|
| hard floor       | DONUT < 0.50 / pipeline == 0.0      | raises `EvalError` (indicates a bug, not underperformance) |
| soft expectation | `config.f1_warn_threshold` (def 0.75) | logs a WARNING, does not fail the run |

Set `KAGGLE2_F1_WARN_ONLY=1` to downgrade the DONUT hard floor
to a `WARNING` so forensic artifacts (`donut_eval_diag.json`,
`combined_metrics.json`) are written even when F1 < 0.50:

```bash
KAGGLE2_F1_WARN_ONLY=1 python main.py --stage eval
```

For a targeted one-shot DONUT re-eval that always terminates and writes
`<run_dir>/donut/donut_eval_diag.json`, use the dedicated console script:

```bash
# targets the most-recently-modified run under runs/ automatically
python -m scripts.donut_eval_only

# target a specific run directory
python -m scripts.donut_eval_only configs/default.json --run-dir runs/<run_id>

# inspect the diag artifact
jq '.lm_head_out_features, .tokenizer_vocab_size, .decoder_start_token_id' \
    runs/<run_id>/donut/donut_eval_diag.json
jq '.samples[0]' runs/<run_id>/donut/donut_eval_diag.json
```

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
              focus_train, focus_inference, eval_pipeline
report/       LaTeX template injection, references
scripts/      vastai_bootstrap.sh — one-shot install+check
main.py       orchestrator (--stage train | eval | paper | all)
```

## Demo — see it work in a browser

One command, no training required:

```bash
pip install -r requirements.txt
make serve            # → http://localhost:8000
```

Drop a receipt image onto the page; DONUT returns `company`, `date`,
`address`, `total`. If `results/donut/` exists the fine-tuned checkpoint
is used; otherwise the server falls back to `config.base_model` and the
page shows a clear banner that predictions will be poor until training
finishes. `GET /health` reports which is loaded.

Endpoints:

| Route | Purpose |
|---|---|
| `GET  /`        | Drag-and-drop upload UI (single HTML page, no build step). |
| `POST /predict` | multipart `file` → `{"fields": {...}, "model_source": ...}`. |
| `GET  /health`  | `{"model_loaded": bool, "model_source": "finetuned"\|"base", ...}`. |

### Production deployment (nginx sub-path `/teb2/`)

The demo is deployed at `https://portearchive.com/teb2/` behind nginx.
`deploy/` contains the required configuration files.

**1. Install the app**

```bash
sudo cp -r . /var/www/kaggle2
cd /var/www/kaggle2
sudo pip install -r requirements.txt
```

**2. Enable the systemd service** (keeps uvicorn alive across reboots)

```bash
sudo cp deploy/kaggle2.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kaggle2
sudo systemctl status kaggle2   # should show "active (running)"
```

**3. Add the nginx location block**

```bash
sudo cp deploy/nginx-teb2.conf /etc/nginx/snippets/kaggle2.conf
# then inside your server { } block add:  include snippets/kaggle2.conf;
sudo nginx -t && sudo systemctl reload nginx
```

The `deploy/kaggle2.service` binds uvicorn to `127.0.0.1:8000` with
`--root-path /teb2`; `deploy/nginx-teb2.conf` proxies
`location /teb2/ → http://127.0.0.1:8000/`.

**4. Enable auto-deploy (GitHub Actions → push to `main` → server restarts)**

`.github/workflows/deploy.yml` triggers automatically after CI passes on
`main`. It SSHes into the server, does a `git pull`, and restarts the
service. Add four secrets in the repository's **Settings → Secrets and
variables → Actions**:

| Secret | Value |
|---|---|
| `DEPLOY_HOST` | Server IP or hostname (e.g. `portearchive.com`) |
| `DEPLOY_USER` | SSH user (e.g. `www-data` or `deploy`) |
| `DEPLOY_SSH_KEY` | Private SSH key (paste the full contents of `~/.ssh/github_deploy`) |
| `DEPLOY_PORT` | SSH port — omit if `22` |

Generate a dedicated key pair on the server:

```bash
ssh-keygen -t ed25519 -C "github-deploy" -f ~/.ssh/github_deploy
cat ~/.ssh/github_deploy.pub >> ~/.ssh/authorized_keys
# then paste the contents of ~/.ssh/github_deploy into the DEPLOY_SSH_KEY secret
```

Also allow the deploy user to restart the service without a password prompt
(replace `deploy` with whatever username you use):

```bash
echo "deploy ALL=(ALL) NOPASSWD: /bin/systemctl restart kaggle2" \
  | sudo tee /etc/sudoers.d/kaggle2-deploy
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
takes ≈ 30 min.

### Outputs — "open folder, select all, download"

After `make all` finishes, **every file worth keeping lives under a
single folder**:

```
runs/<run_id>/                          # one folder per run
├── metrics/
│   ├── combined_metrics.json           # headline F1/NED/EM
│   ├── extended_metrics.json           # per-field P/R + bootstrap CI
│   ├── assigner_metrics.json           # assigner diagnostics
│   ├── pipeline_metrics.json           # pipeline sidecar
│   ├── cost_{donut,pipeline}.json      # USD + Wh + CO2
│   ├── pipeline_meta.json
│   ├── gtocr_rulebased_metrics.json
│   └── unresolved_vars.json            # audit: \VAR{} keys that did NOT resolve
├── predictions/
│   ├── donut_preds.jsonl               # one row per test image (gt, pred, per-field)
│   ├── pipeline_preds.jsonl
│   ├── donut_errors.jsonl
│   ├── pipeline_errors.jsonl
│   └── per_field_errors.jsonl          # merged 8-category miss classifier
├── env/
│   ├── git_sha.txt
│   ├── pip_freeze.txt
│   ├── nvidia_smi.txt
│   ├── config_snapshot.json            # exact configs/default.json used
│   └── hostinfo.json                   # CPU, RAM, GPU model, driver, CUDA
├── figures/                            # every PDF the paper cites
├── paper/
│   ├── paper_filled.tex
│   └── paper_filled.pdf                # the final submission
├── attention_samples.json              # attention-heatmap source
├── training_log.json                   # per-epoch scalars
├── bug_timeline.json                   # copied fixture (paper cites)
└── MANIFEST.json                       # relpath + sha256 + size + producer-stage
```

**Nothing lands in `./results/` (fixtures-only) or `./report/` (template-only).**
The Copilot round-trip contract is:

```bash
# On vast.ai:
make all                                 # runs/<run_id>/ is self-contained
make pack                                # optional: tar the run into a single archive

# In your browser:
# → open the folder runs/<run_id>/
# → select all, download
# → upload back to Copilot
```

`MANIFEST.json` is the definitive index — every download target appears
in it with sha256 so `scripts/unpack_run.sh` can verify integrity
end-to-end.

### "No placeholders" contract

After a successful `make all`, every `\VAR{}` in the compiled PDF
resolves to a real measured value (not a fillers `---`).  The
`metrics/unresolved_vars.json` sidecar lists any key that did NOT
resolve on the current run — a successful full run writes
`{"unresolved": [], "count": 0}`.

Which metrics have producers (and therefore real values) is enumerated
in `docs/TRACKING.md`; any `\VAR{}` key not covered by the producer
matrix will surface in `unresolved_vars.json` for audit rather than
rendering as a silent placeholder in the PDF.

Partial runs:

```bash
make check         # ruff + mypy --strict + import smoke
make test          # pytest (no GPU needed)
python main.py --stage train
python main.py --stage eval
python main.py --stage eval_rule_gtocr  # real F1 on GT-OCR stream, no HF needed
python main.py --stage paper
```

### Offline / no-HF eval path

`--stage eval_rule_gtocr` runs the rule-based assignment head over
SROIE's GT-OCR box-file text (bypassing YOLO+TrOCR by feeding ground-truth
bboxes/text directly) and writes real F1 / NED / EM into
`results/gtocr_rulebased_metrics.json` plus a paper-ready
`results/combined_metrics.json` (DONUT and pipeline rows are zero-padded
and tagged `artifact_mode: gtocr_rulebased_only`; the paper's results
table honestly reflects that). This path needs only SROIE (GitHub) —
no Hugging Face Hub access, no GPU, ~1 second on CPU.

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

All hyperparameters live in `configs/default.json`. F1-affecting knobs:

| Parameter | Default | Effect on expected F1 |
|---|---|---|
| `epochs_donut` | 30 | Longer training → higher F1, diminishing returns past ~30 on 500 SROIE receipts. |
| `image_size` | [1280, 960] | Higher resolution → better address/total recognition; more VRAM. |
| `num_beams` | 4 | Beam search typically gains 2–4 F1 points over greedy. |
| `lr_scheduler_type` | cosine | Cosine > linear for short SROIE runs. |
| `warmup_ratio` | 0.1 | Stabilises early loss on small data. |
| `gradient_checkpointing` | true | Lets batch 8 × 1280 × 960 fit in 24 GB. |
| `patience` | 3 | EarlyStopping on plateau of eval F1. |
| `precision` | bf16 | bf16 on Ampere+; fp16 with grad-clip otherwise (Bug 4). |
| `yolo_image_size` | 1024 | MUST match training and inference (Bug 5). |
| `epochs_trocr` | 12 | Floor of 5 enforced in config.py (Bug 6). |
| `f1_warn_threshold` | 0.75 | Soft WARN threshold (non-fatal). |
| `canonical_sroie_enabled` | true | When true, `data.sroie_canonical.ensure_canonical_test_set` auto-downloads the official ICDAR-2019 SROIE Task-3 test set (347 images + KIE GT) from `rrc.cvc.uab.es` (sha256-pinned docTR mirror as fallback) and replaces the 500/63/63 internal test split with it.  Numbers become directly comparable to the public SROIE leaderboard.  Set `false` for the basic 500/63/63 study. |
| `canonical_sroie_test_url` | `https://rrc.cvc.uab.es/downloads/SROIE_test_images_task_3.zip` | Primary download URL for the 347 test images. |
| `canonical_sroie_gt_url` | `https://rrc.cvc.uab.es/downloads/SROIE_test_gt_task_3.zip` | Primary download URL for the Task-3 GT entities. |
| `canonical_sroie_mirror_url` | docTR | Fallback mirror used when the RRC primary fails (sha256 verified before extraction). |
| `paper_variant` | `focus` | `focus` → `report/template_focus.tex` (DONUT vs YOLO+TrOCR+Assigner on 626 train + 347 canonical test); `baseline` → `report/template_baseline.tex` (DONUT vs YOLO+TrOCR+regex vs GT-OCR+regex on 500/63/63 internal split).  CLI override: `--paper-variant {focus,baseline}` — flips `canonical_sroie_enabled` to match. |

### Assigner-fix knobs (strategies B / C / E / F / G / I from `docs/assigner_fix_plan.md`)

Added under `extra` in `configs/default.json`. All default to `0.0` (disabled) so existing
training runs reproduce bit-exact; set non-zero to opt into the fresh-train regime.

| Parameter | Default | Strategy | Effect |
|---|---|---|---|
| `focus_hardneg_weight` | 0.0 | B | λ for the listwise hinge over SUBTOTAL/CASH/CHANGE/TAX/header distractors. 0.5 is a reasonable starting point. |
| `assigner_kd_weight` | 0.0 | C | λ for KL-divergence KD from the rule-based teacher (`_score_money` softmax). 0.1 is a reasonable starting point. |
| `priors_v3` | false | E | Upgrade priors from 9-d to 14-d (adds `is_subtotal / cash / change / tax / rounding` bits). Requires a fresh train; v2 checkpoints still load. |
| `priors_v4` | false | FOCUS | Upgrade priors from 14-d to 20-d (adds 6 FOCUS-T/C dims: `is_subtotal_kw, is_tax_kw, is_company_boilerplate, line_y_normalised, money_value_normalised, arithmetic_witness_self`). Required by `focus_total_enabled` / `focus_company_enabled`. Requires a fresh assigner train. |
| `focus_enabled` | false | FOCUS-A | Master toggle for the FOCUS framework. Adds the address-span head (PR #106). Sub-flags below gate FOCUS-T / FOCUS-C independently. |
| `focus_total_enabled` | false | FOCUS-T | Add the relational total head (3× `Linear(d,1)`: score + sigmoid-gated arithmetic-witness term). Requires `focus_enabled=True` and `priors_v4=True`. |
| `focus_total_witness_weight` | 1.0 | FOCUS-T | Coefficient on the arithmetic-witness gate term in the FOCUS-T `final` logits. |
| `focus_company_enabled` | false | FOCUS-C | Add the positional company head (2× `Linear(d,1)`: score + y-bias + boilerplate-penalty). Requires `focus_enabled=True` and `priors_v4=True`. |
| `focus_company_y_weight` | 1.0 | FOCUS-C | Coefficient on the `-y_norm` top-of-receipt bias in the FOCUS-C `final` logits. |
| `focus_company_boilerplate_weight` | 1.0 | FOCUS-C | Coefficient on the `-is_company_boilerplate` penalty (pushes "SDN BHD" / "BERHAD" / "PTE LTD" suffix lines down). |
| `focus_ctkr_k` | 4 | Bug 18 | Contrastive Top-K Repulsion fanout — covers the observed BRN+INV+TEL+GST confuser cluster per receipt. |
| `focus_ctkr_margin` | 0.05 | Bug 18 | Margin between the most-attended non-gold line and the *weakest* gold line. Uniform softmax over 30 lines ≈ 0.033, so 0.05 is just above the no-info floor. |
| `focus_ctkr_weight` | 1.0 | Bug 18 | λ for the CTKR term in the composite loss. The plain Σ-neg-mass term originally proposed in the parent prompt's step C is deleted, not stacked — the two push in the same direction but CTKR is sparse + adaptive while plain neg-mass is dense + constant. |
| `negative_mass_weight` | 0.5 | Bug 18 | Back-compat alias for the deprecated plain Σ-neg-mass term. Active loss is the CTKR + soft-IoU composite; this knob is read by triage scripts for ablation comparisons only. |
| `focus_synth_subtotal` | 0.0 | I | Per-receipt probability of injecting a synthetic `SUBTOTAL: RM xx.yy` line before the true TOTAL. 0.3–0.5 suggested. |
| `assigner_ocr_noise` | 0.0 | F | Per-receipt probability of re-deriving priors from OCR-noised text (digit split, O↔0, trailing-zero drop). 0.2 suggested. |
| `focus_hidden_dim` | 384 | G | Backbone width. Plan recommends 192 for fresh trains with B/C/E enabled (~1.4M params, better match for 500 receipts). |
| `focus_n_layers_level2` | 6 | G | Backbone depth. Plan recommends 3 for fresh trains. |

Inference-side strategies L (additive attn×rule ensemble) and H (confidence-gated
delegation to rule-based) are always on — they need no retrain and no checkpoint
change; see `models/consensus.py`.

## F1-destroying bugs

See [docs/bugs.md](docs/bugs.md) for the full bug ledger (Bugs 1–14, 18, 19) and code guards.

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
