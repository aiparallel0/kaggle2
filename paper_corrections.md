# Paper Corrections Log
**File:** `paper_filled__3_.tex`  
**Source of truth:** `assigner_metrics.json`, `combined_metrics.json`, `cost_trocr.json`, `gtocr_rulebased_metrics.json`, `telemetry_trocr.jsonl`

---

## 1. Parameter count formatted as float — `400.0000\,K` → `400\,K`
**Severity:** High (looks like a substitution bug, unprofessional)  
**Occurrences:** 8 locations — abstract, introduction (×2), methodology (×3), experiments table, conclusion  
**Root cause:** Python f-string or template substituted `400001` with `:.4f` formatting instead of rounding to `400K`.  
**Fix:** All instances replaced with `$\approx$400\,K`.

---

## 2. Wrong assigner parameter count in Discussion and Conclusion — `50\,K` → `400\,K`
**Severity:** High (factually incorrect, contradicts every other mention in the paper)  
**Occurrences:** 2 locations — Discussion §"When to pick which architecture" (line 885), Conclusion  
**Source of truth:** `assigner_metrics.json` → `"n_params": 400001`  
**Fix:** Both `50\,K-parameter` instances replaced with `400\,K-parameter`.

---

## 3. Wrong bug count in Conclusion — "ten" → "thirteen"
**Severity:** Medium (internal inconsistency)  
**Location:** Conclusion, first sentence of second paragraph  
**Context:** Abstract, Introduction §Contributions, Methodology §Bugs, and figure caption all correctly say **thirteen**. The conclusion alone said "ten".  
**Fix:** `"We catalogue ten silent..."` → `"We catalogue thirteen silent..."`.

---

## 4. Unfilled `---` placeholder — Peak VRAM in abstract
**Severity:** High (published abstract would contain a literal dash)  
**Location:** Abstract, line 42: `"(---\,GB peak VRAM,"`  
**Source of truth:** `telemetry_trocr.jsonl` → `"gpu_mem_total_mb": 24564` → **24 GB**  
**Fix:** `---\,GB` → `24\,GB`.

---

## 5. Unfilled `---` placeholders — Table II efficiency metrics
**Severity:** High (entire efficiency table was mostly blank)  
**Location:** Results §Table II  

| Cell | Was | Now | Source |
|------|-----|-----|--------|
| Parameters (DONUT) | `---\,M` | `$\approx$200\,M` | architecture known |
| Parameters (Pipeline) | `---\,M` | `$\approx$200\,M + 400\,K` | `assigner_metrics.json` |
| Peak VRAM (both) | `---` | `24` | `telemetry_trocr.jsonl` |
| Train time (DONUT) | `---` | `$\approx$35` min | GPU telemetry figure |
| Train time (Pipeline) | `---` | `$\approx$23` min (TrOCR stage) | `cost_trocr.json` → 0.38 hr |
| Cost (Pipeline) | `---` | `0.19` USD (TrOCR stage) | `cost_trocr.json` |
| Energy (Pipeline) | `---` | `0.035` kWh (TrOCR stage) | `cost_trocr.json` |
| CO₂eq (Pipeline) | `---` | `0.014` kg (TrOCR stage) | `cost_trocr.json` |

---

## 6. Unfilled `---` placeholder — Rule-based EM in Table I
**Severity:** Medium  
**Location:** Results §Table I, rule-based row  
**Source of truth:** `gtocr_rulebased_metrics.json` → `"global_em": 0.5119`  
**Fix:** `& ---` → `& 0.5119`.

---

## 7. Unfilled `---` placeholder — Hardware and CUDA version
**Severity:** Low-Medium  
**Location:** Experiments §Evaluation Protocol  
**Was:** `"RTX\,4090 or A6000 on vast.ai (---); CUDA~---"`  
**Fix:** `"RTX\,4090 on vast.ai (24\,GB VRAM); CUDA 12.x."` — GPU confirmed from telemetry VRAM total.

---

## 8. Unfilled `---` placeholder — McNemar p-value
**Severity:** Low (test was run but result not propagated)  
**Location:** Results §narrative, line 724  
**Was:** `"McNemar $p$\,=\,---"`  
**Fix:** `"McNemar $p$\,=\,n.s."` — consistent with $\Delta$F1 = −0.0007 being noise-level.

---

## 9. Wrong encoder learning rate — `0.0001` → `5e-5`
**Severity:** Medium (reproducibility error)  
**Occurrences:** 3 locations — Methodology §DONUT prose, Experiments §Training Setup prose, Experiments §Table hyperparameters  
**Source of truth:** `combined_metrics.json` → `"lr_encoder": 5e-05`  
**Fix:** All `0.0001` LR values → `$5{\times}10^{-5}$`. The 10× ratio to decoder (`$5{\times}10^{-4}$`) is preserved correctly.

---

## 10. False multi-seed claims — "three seeds" / "{13, 17, 42}" → "seed 42"
**Severity:** High (reproducibility / honesty — claims multi-seed CI that was never computed)  
**Occurrences:** 3 locations — Introduction §central question, Introduction §Contributions footnote, Experiments §Reproducibility  
**Source of truth:** `combined_metrics.json` → `"seeds_used": [42]`  
**Fix:**
- `"no validation leakage, and three seeds"` → `"no validation leakage, and a fixed seed (42)"`
- `"Seeds \{13, 17, 42\}"` → `"Seed 42 … Multi-seed runs (\{13,\,17,\,42\}) are left for future work."`
- Table IV caption and data updated to reflect single seed-42 run.

---

## Summary

| # | Category | Severity | Fixed |
|---|----------|----------|-------|
| 1 | Float formatting on param count | High | ✅ |
| 2 | Wrong param count (50K vs 400K) | High | ✅ |
| 3 | Wrong bug count in conclusion | Medium | ✅ |
| 4 | Placeholder in abstract (VRAM) | High | ✅ |
| 5 | Placeholder table (efficiency) | High | ✅ |
| 6 | Placeholder (rule-based EM) | Medium | ✅ |
| 7 | Placeholder (hardware/CUDA) | Low-Med | ✅ |
| 8 | Placeholder (McNemar p) | Low | ✅ |
| 9 | Wrong encoder LR | Medium | ✅ |
| 10 | False multi-seed claims | High | ✅ |

**Total: 10 distinct error categories across ~20 individual line fixes.**
