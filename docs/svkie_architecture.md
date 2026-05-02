# SVKIE — how the system actually works

A focused technical reference for drawing presentation and paper
visuals.  Concrete tensor shapes, decision points, and a worked
numeric example.  No proofs, no related work, no methodology
debates — just enough to draw the architecture correctly.

---

## 1. End-to-end flow at a glance

```
    ┌─────────────────────┐
    │   Receipt image     │   I ∈ ℝ^(H × W × 3)
    └──────────┬──────────┘
               │
       ┌───────▼────────┐
       │  YOLOv8n       │   line detection
       │  (Stage 1)     │
       └───────┬────────┘
               │   N bounding boxes  {b₁, …, b_N},  bᵢ ∈ ℝ⁴
               │
       ┌───────▼────────┐
       │  TrOCR-small   │   per-line text
       │  (Stage 2)     │
       └───────┬────────┘
               │   N text strings  {t₁, …, t_N}
               │
       ┌───────▼────────────────────────────────┐
       │  Per-line feature construction         │   priors_v4 ∈ ℝ^(N × 20)
       │  (text embeds + bbox normalisation     │   text_feats ∈ ℝ^(N × 768)
       │   + rule-derived priors)               │
       └───────┬────────────────────────────────┘
               │
   ┌───────────┴───────────┬───────────────────────┐
   │                       │                       │
┌──▼──────┐         ┌──────▼─────┐         ┌───────▼──────┐
│ FOCUS-T │         │   GAT      │         │  Frozen      │   three parallel
│ cross-  │         │  graph-    │         │  ResNet-18   │   field-assignment
│ attn    │         │  attn      │         │  CNN visual  │   heads
│  H₁     │         │   H₂       │         │   H₃         │
└──┬──────┘         └──────┬─────┘         └───────┬──────┘
   │                       │                       │
   │   each emits softmax over N lines per field   │
   │   Pᶠⁱ = softmax(scoreᶠⁱ),  shape (4 × N)      │
   │                                               │
   └────────┬───────────────┬─────────────────────┘
            │               │
            │       ┌───────▼────────┐
            │       │  Zone-prior    │   3-state HMM (header/items/totals)
            │       │   HMM          │   summary z ∈ ℝ³
            │       └───────┬────────┘
            │               │
   ┌────────▼───────────────▼─────────┐
   │  Gating MLP                      │   weights heads by per-field
   │  in:  (Pᵗᵒᵖ_H₁, Pᵗᵒᵖ_H₂,         │   confidence + zone summary
   │        Pᵗᵒᵖ_H₃, z)               │
   │  out: w ∈ ℝ^(4 × 3) (softmax     │
   │       over heads, per field)     │
   └────────┬─────────────────────────┘
            │   fused per-field softmax
            │   P_fused = Σₖ wₖ · P_Hₖ,   shape (4 × N)
            │
   ┌────────▼─────────────────────────┐
   │  Argmax candidate per field      │   ŷᶠⁱ = argmax_i P_fused[f, i]
   └────────┬─────────────────────────┘
            │   four candidate values:
            │     ŷ_company, ŷ_date, ŷ_address, ŷ_total
            │
   ┌────────▼────────────────────────┐
   │  FOCUS-Σ verifier               │   only on ŷ_total
   │  W(ŷ_total) ∈ {0, 1, 2, 3}      │   witness count
   └────────┬────────────────────────┘
            │
            │  if W ≥ 2:  commit
            │  if W == 1:  commit (single witness ok)
            │  if W == 0:  try Hamming-1 OCR-drift recovery
            │              if no recovery: confidence cascade
            │
   ┌────────▼────────────────────────┐
   │  Confidence-gated cascade       │   if FOCUS-T softmax max < 0.55
   │  (epistemic prior)              │   AND verifier silent:
   │                                 │     fall back to rule-based pick
   └────────┬────────────────────────┘
            │
   ┌────────▼────────────────────────┐
   │  Output                         │   final structured fields
   │  {company, date, address,       │
   │   total}                        │
   └─────────────────────────────────┘
```

---

## 2. Tensor shapes at every stage

| Stage | Input shape | Output shape | Notes |
|---|---|---|---|
| YOLOv8n | `(H, W, 3)` | `(N, 4)` bboxes | N typically 15–60 lines |
| TrOCR | `(N, 3, 64, 384)` line crops | `N` text strings | greedy decode |
| Per-line text feature | `N` strings | `(N, 768)` | TrOCR encoder pooled |
| Per-line priors | `N` strings + bboxes | `(N, 20)` | priors_v4: y_norm, money flag, keyword flags, … |
| FOCUS-T head | `(N, 768) + (N, 20)` | `(4, N)` softmax | 4 field queries × N lines |
| GAT head | `(N, 768) + (N, 4)` bboxes + kNN graph | `(4, N)` softmax | k=6 neighbours by bbox-centre distance |
| CNN visual head | `(N, 3, 224, 224)` line crops | `(4, N)` softmax | frozen ResNet-18 → projection |
| Zone-prior HMM | `(N, 6)` features | `(N, 3)` posterior | header / items / totals per line |
| Gating MLP | `(3·4 + 3,) = (15,)` | `(4, 3)` softmax | per-field weights over 3 heads |
| Fused softmax | weighted sum over heads | `(4, N)` | row-stochastic |
| FOCUS-Σ verifier | candidate value `w` + line money values | `W(w) ∈ {0,1,2,3}` | witness count |

---

## 3. Stage-by-stage detail

### Stage 1 — YOLO line detection

Input: receipt image.
Output: list of axis-aligned bboxes `{(x₀, y₀, x₁, y₁)ᵢ}` for text lines.
Trained: YOLOv8n on weak-supervision boxes derived from SROIE GT-OCR.
Image size: 1024 × 1024.  Confidence threshold: 0.25.

For visualisation:
- **Show bboxes coloured by detection confidence** on the receipt image.
- **Show NMS in action**: cluster of overlapping boxes → single survivor.

### Stage 2 — TrOCR text recognition

Input: line crops (each padded by 6 px).
Output: ASCII transcript per line.
Trained: TrOCR-small-printed on SROIE crops, 12 epochs.
Decoding: greedy, max 64 new tokens.

For visualisation:
- **Show one crop → token-by-token decode** with attention over crop pixels.
- **Show OCR-error mode**: e.g. "RM 15.50" → "RM 15.5O" (digit-O confusion).

### Stage 3 — Per-line feature construction

For each detected line `i` we build:
- `text_feats[i]` ∈ ℝ⁷⁶⁸ — TrOCR encoder last hidden state, mean-pooled
- `bbox[i]` ∈ ℝ⁴ — `(x₀, y₀, x₁, y₁)` normalised to `[0, 1]`
- `priors[i]` ∈ ℝ²⁰ — rule-derived flags (priors_v4):
  - `y_norm`: vertical position
  - `is_money_line`: `\$|RM|\d+\.\d{2}` matched
  - `is_subtotal_kw`, `is_tax_kw`, `is_total_kw`, …
  - `is_company_boilerplate`: `(SDN BHD|PTE LTD|BHD|...)` matched
  - `arithmetic_witness_self`: line value satisfies `subtotal + tax = total` for some pair
  - 14 more boolean / scalar features

For visualisation:
- **Show priors as a colour-coded matrix** `(N × 20)` — one row per
  line, columns coloured by which flag fired.

### Stage 4 — Three-headed assigner ensemble (the centerpiece)

#### 4a — FOCUS-T cross-attention head (H₁)

```
Q ∈ ℝ^(4 × d)           4 learnable field-query embeddings
K, V ∈ ℝ^(N × d)        per-line key/value from text_feats + priors

attn[f, i] = softmax_i (Q[f] · Kᵀ[i] / √d)
score[f, i] = attn[f, i] + α · prior[i, field_specific_idx]
P_H1[f, i] = softmax_i(score[f, i])
```

For visualisation:
- **Heatmap (4 × N)**: rows are fields (company/date/address/total), columns are lines, cell colour = `P_H1[f, i]`.  Bright spots show where the head puts mass.
- **Multi-line address case**: show how the company query peaks on one line, address query smears across 3–5 contiguous lines.

#### 4b — GAT graph-attention head (H₂)

```
G = kNN-graph over bbox centres, k=6
For each edge (i, j): 
    e[i, j] = LeakyReLU(a · [W·xᵢ ‖ W·xⱼ])
    α[i, j] = scatter-softmax over neighbours of i
    h'[i] = Σⱼ α[i, j] · W · xⱼ
```

H₂ output: per-line embedding refined by 2 layers of GAT, then a
field-query attention identical in shape to H₁.

For visualisation:
- **Show the kNN graph as edges between line bboxes** drawn directly on the receipt image.
- **Compare H₁ vs H₂ heatmaps**: H₂ tends to spread mass across spatially-adjacent lines; H₁ does not.

#### 4c — Frozen-CNN visual head (H₃)

```
For each line bbox bᵢ:
    crop_i = letterbox(image[bᵢ], 224×224)
    feat_i = resnet18(crop_i)        ∈ ℝ⁵¹²    (frozen)
    proj_i = Linear(512 → d)(feat_i)            (trainable, ~50K params)
score[f, i] = Q_visual[f] · proj_iᵀ
P_H3[f, i] = softmax_i(score[f, i])
```

For visualisation:
- **Show crop → ResNet feature → projection → score** as a left-to-right pipeline diagram for ONE line.
- **Show the visual signal**: company crops fire on H₃ via font weight; total crops fire via the printed currency-symbol cluster.

#### 4d — Gating MLP fusion

```
zone_summary = mean over lines of zone_prior_posterior  ∈ ℝ³

per-field per-head top-prob:
    Pᵗᵒᵖ_Hₖ[f] = max_i P_Hₖ[f, i]                 ∈ ℝ⁴

gate_input = concat(Pᵗᵒᵖ_H₁, Pᵗᵒᵖ_H₂, Pᵗᵒᵖ_H₃, zone_summary)  ∈ ℝ¹⁵
gate_logits = MLP_2_layer_GELU(gate_input).reshape(4, 3)        # 4 fields × 3 heads
weights[f] = softmax_k(gate_logits[f])                          ∈ ℝ³

P_fused[f, i] = Σₖ weights[f, k] · P_Hₖ[f, i]                   ∈ ℝᴺ
```

The gate has ~3K trainable parameters.

For visualisation:
- **Show three input bars (top-prob per head per field) → MLP → output bar** (head weights per field).
- **Show how a head with degenerate output (all-zero P_top) is masked from the softmax denominator** — graceful degradation.

### Stage 5 — FOCUS-Σ verifier

For the candidate `ŷ_total`, count witnesses across three arithmetic identities:

```
M = {i : t_i parses to a money value}
v_i = parsed money value of line i  (in cents)
τ = sum of tax + service − discount
I = {i ∈ M : t_i has no distractor keyword}        # candidate item lines

I₁:  ŷ_total = v_cash − v_change                              ± 0.02
I₂:  ŷ_total = v_subtotal + τ                                 ± 0.02
I₃:  ∃ S ⊆ I, |S| ≥ 2  s.t.  ŷ_total = Σᵢ∈S vᵢ + τ           ± 0.02

W(ŷ_total) = 𝟙[I₁ holds] + 𝟙[I₂ holds] + 𝟙[I₃ holds]
```

I₃ is computed by a **bounded subset-sum DP** over per-item cents:
```
T = boolean array of length max_sum (≤10⁶)
T[0] = True
for v in {v_i : i ∈ I}:
    T |= T << v
ŷ_total ∈ T  →  I₃ holds
```

Cost: `O(|I| · max_sum_cents)` ≈ 30 × 10⁶ = 3 × 10⁷ ops ≈ 35 ms in pure Python.

For visualisation:
- **Show an actual receipt** with the lines highlighted that participate in each identity.
- **Show the subset-sum DP table filling row by row** as item values are absorbed.
- **Show the abstain case**: a receipt with no SUBTOTAL keyword and no CASH/CHANGE — only I₃ fires.

### Stage 6 — Hamming-1 OCR-drift recovery

When `W(ŷ_total) = 0` and the candidate string carries the `TOTAL` keyword:

```
y_str = OCR'd cent string for ŷ_total          # e.g. "15580" cents
N₁(y) = all Hamming-1 neighbours of y         # |N₁| ≤ 9 · D
T = subset-sum target set (computed in stage 5)
recover:
    candidates = N₁(y) ∩ T
    if |candidates| == 1:  return that candidate
    else:                  abstain
```

For visualisation:
- **Show the digit-substitution lattice** for one OCR'd value (e.g., "15580" → 9 single-digit neighbours per position).
- **Highlight the unique neighbour that lands in T**.

### Stage 7 — Confidence-gated cascade (epistemic prior)

```
if W(ŷ_total) ≥ 1:
    commit ŷ_total
elif Hamming-1 recovery succeeds:
    commit recovered value
elif max_i P_fused[total, i] < 0.55:
    fall back to rule-based regex pick
else:
    commit ŷ_total with low-confidence flag
```

For visualisation:
- **Draw a 4-state state machine**: `verified → recovered → rule-fallback → low-confidence`, with the transitions labelled by W and softmax-max conditions.

---

## 4. Worked numeric example — one receipt

Imagine a receipt with these OCR'd lines (vertical order):

```
i=1   "MAYDAY MART SDN BHD"                   company
i=2   "20, JALAN BUKIT MERAH"                 address
i=3   "01/05/2024"                            date
i=4   "BREAD                  3.50"
i=5   "MILK                   5.00"
i=6   "EGGS                   4.20"
i=7   "SUBTOTAL              12.70"
i=8   "GST 6%                 0.76"
i=9   "TOTAL                 13.46"
i=10  "CASH                  20.00"
i=11  "CHANGE                 6.54"
```

Per-line money values (cents):
```
v_4 = 350,  v_5 = 500,  v_6 = 420,
v_7 = 1270 (subtotal), v_8 = 76 (tax),
v_9 = 1346 (total),    v_10 = 2000 (cash), v_11 = 654 (change)
```

Feature construction:
- `priors_v4[7].is_subtotal_kw = 1`
- `priors_v4[8].is_tax_kw = 1`
- `priors_v4[9].is_total_kw = 1`
- `priors_v4[10].is_cash_kw = 1, priors_v4[11].is_change_kw = 1`
- Zone-prior posterior: lines 1–3 → header, 4–6 → items, 7–11 → totals

FOCUS-T candidate for `total`: line 9 (the strong `TOTAL` keyword).
ŷ_total = 1346 cents.

FOCUS-Σ verifier on `ŷ_total = 1346`:
```
I₁:  v_cash − v_change = 2000 − 654 = 1346  ✓   (witness 1)
I₂:  v_subtotal + τ = 1270 + 76 = 1346      ✓   (witness 2)
I₃:  S = {4, 5, 6}, sum = 350+500+420 = 1270; +τ = 1346  ✓   (witness 3)

W(1346) = 3
```

All three identities fire.  The verifier commits with maximum
confidence.

Now an OCR-drift case: imagine line 9 was OCR'd as `"TOTAL  13.16"` (the "4" dropped):
```
ŷ_total = 1316 cents

I₁:  2000 − 654 = 1346 ≠ 1316   ✗
I₂:  1270 + 76 = 1346 ≠ 1316    ✗
I₃:  no subset of {350, 500, 420} + 76 sums to 1316   ✗

W(1316) = 0
```

Hamming-1 recovery on "1316":
```
N₁("1316") = {2316, 3316, ..., 1916, 1006, ...}      # 36 neighbours
T (subset-sum target set) = {76, 350, 426, 500, 576, 850, 920, 1346, 1426, ...}
N₁("1316") ∩ T = {1346}                              # unique match
recover ŷ_total = 1346
```

The verifier commits the recovered value.

For visualisation:
- **Draw the receipt** with each line's contribution labelled.
- **Show the three identity checks lighting up green / red** as the verifier evaluates them.
- **Show the OCR-drift case** with the lattice of Hamming-1 neighbours and the unique recovery in `T`.

---

## 5. Visualisation recipes

### Recipe A — Full pipeline architecture diagram (1 figure)

Boxes: YOLO → TrOCR → Per-line features → 3 parallel head boxes
(FOCUS-T, GAT, CNN) → Zone-prior HMM (sidecar input to the gate) →
Gating MLP → P_fused → Argmax → FOCUS-Σ verifier → Output.

Arrows labelled with tensor shapes.

### Recipe B — Three-headed ensemble heatmap (1 figure)

3 panels, each (4 × N) heatmap for a single receipt:
- Left: H₁ (FOCUS-T cross-attention)
- Centre: H₂ (GAT)
- Right: H₃ (CNN visual)

Below: the gate's per-field per-head weight matrix `(4 × 3)`.
Below that: the fused (4 × N) heatmap.

Caption: "Each head emphasises different lines.  H₁ sharp on
lexical anchors; H₂ smears across spatially-adjacent lines; H₃
fires on visually salient regions.  The gate combines them."

### Recipe C — FOCUS-Σ verifier flowchart (1 figure)

Three identity boxes (I₁, I₂, I₃) feeding into a witness-count
node, branching to `commit` / `recover` / `cascade`.  Annotate
each identity box with the actual receipt lines that participate.

### Recipe D — Subset-sum DP table animation (1 figure or animation)

Show the bit-mask `T` filling row by row as item values are
absorbed.  Highlight the cell that ŷ_total lands in (or doesn't,
in the OCR-drift case).

### Recipe E — Five-priors radar chart (presentation slide)

5-axis radar: arithmetic, spatial, visual, lexical, epistemic.
Shade the radar for each system: Paper 2 baseline (only lexical),
+I₂+I₃ (lexical + arithmetic), +zone+GAT (lexical + arithmetic +
spatial), +CNN (lexical + arithmetic + spatial + visual), +cascade
(all five).  The shaded area grows with each prior added — and
each radar's apex F1 score is annotated next to the panel.

### Recipe F — Wrapper-Δ matrix (1 figure)

3 × 2 grid: 3 architectures (DONUT, LayoutLMv3, FOCUS-T) × 2
configurations (Bare, +SVKIE).  Each cell coloured by F1 value;
arrows showing the wrapper-Δ lift per architecture.

### Recipe G — Confidence cascade state machine (1 figure)

4 nodes: `[verified]`, `[recovered]`, `[rule-fallback]`,
`[low-confidence-flagged]`.  Arrows labelled by W = 0/1/2/3 and
softmax-max thresholds.  Annotate each state with the % of
canonical SROIE Task-3 receipts that land in it.

---

## 6. What you need from the code to draw these

| Recipe | Code source |
|---|---|
| A (full pipeline) | this doc + paper3/report/sections/method.tex |
| B (heatmap) | paper3/models/focus_attention_paper3.py + focus_gat_paper3.py + focus_cnn_paper3.py |
| C (verifier flowchart) | paper3/models/total_arithmetic_paper3.py |
| D (subset-sum DP) | paper3/models/total_arithmetic_paper3.py::_dp_solve |
| E (priors radar) | paper3/results/ablation_focus_sigma.json (when populated) |
| F (wrapper-Δ matrix) | paper3/results/wrapper_delta_metrics.json (when populated) |
| G (cascade state machine) | paper3/models/consensus_paper3.py |

---

## 7. What the visuals must communicate

For the **paper** figures:
- Architecture is principled, not ad-hoc.  Each module has a clear
  role tied to a structural prior.
- Each ablation increment is interpretable: "we added arithmetic
  witnesses, F1 went up by X" — not "we added neural capacity".
- The verifier is rule-based and unaffected by upstream
  compression / training noise.

For the **presentation** slides:
- The 5-prior framework is the headline.  Each prior is one slide.
- The wrapper-Δ matrix proves architecture-agnosticism.
- The worked numeric example makes the FOCUS-Σ identity-3
  contribution concrete.

Use these descriptions as the prompt input to whatever diagramming
tool you prefer (TikZ for paper, draw.io / excalidraw / Lectic for
slides).
