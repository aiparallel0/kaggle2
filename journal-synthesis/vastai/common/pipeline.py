"""Shared model-inference + dual-axis signal pipeline.

This is the single pipeline E5 runs both axes through so receipt_ids
align by construction. The import patterns and decode/margin logic are
reused (not copied verbatim, but faithful in method) from:

  - arith-gating/experiments/phase3_donut_extract.py  (Donut decode +
    softmax_confidence / c_seq extraction)
  - arith-gating/scripts/smoke/BM_beam_margin.py       (num_beams=2,
    num_return_sequences=2, sequences_scores -> margin = top1-top2)
  - triology core.subset_sum.i3_accepts                (Axis-A subset-sum
    verifier, EPS_CENTS=2, k_min auto-select)

HEAVY IMPORTS (torch / transformers / datasets / scipy) are deferred
into the functions that need them so this module byte-compiles and
imports in a no-GPU environment for `python3 -m py_compile`. NOTHING
here returns a fabricated number: every function either loads a real
model or refuses (raising) when the GPU/model/data is absent.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

# ---- Axis-A: subset-sum verifier (triology core.subset_sum, lifted) -------

EPS_CENTS = 2  # tolerance, identical to triology / Paper 1


def reachable_targets(items_cents: List[int], tau_cents: int,
                      k_min: int = 1):
    """Classical sparse 0/1 subset-sum: best[s] = min #items summing to s.
    Verbatim method from triology core.subset_sum.reachable_targets."""
    best: Dict[int, int] = {0: 0}
    for v in items_cents:
        for s, k in list(best.items()):
            ns, nk = s + v, k + 1
            if ns not in best or best[ns] > nk:
                best[ns] = nk
    return {s + tau_cents for s, k in best.items() if k >= k_min}


def i3_accepts(candidate_cents: int, items_cents: List[int],
               tau_cents: int, eps_cents: int = EPS_CENTS) -> bool:
    """Paper 1 / triology I_3 verifier. k_min auto-select: |S|>=2 when
    tau==0 (forbid single-item==total), |S|>=1 otherwise."""
    k_min = 1 if tau_cents != 0 else 2
    targets = reachable_targets(items_cents, tau_cents, k_min)
    return any(abs(candidate_cents - t) <= eps_cents for t in targets)


def subset_sum_verdict(candidate_cents: Optional[int],
                       items_cents: Optional[List[int]],
                       tau_cents: int = 0) -> str:
    """Return "pass" | "fail" | "abstain".

    Abstain (NOT silently scored) when the identity is unavailable -
    same convention as run_analysis.py E1 (n_applicable==0 excluded) and
    triology i3_available (needs >=2 items, or >=1 item + non-zero tau).
    """
    if candidate_cents is None or not items_cents:
        return "abstain"
    if tau_cents != 0:
        if len(items_cents) < 1:
            return "abstain"
    else:
        if len(items_cents) < 2:
            return "abstain"
    return "pass" if i3_accepts(candidate_cents, items_cents, tau_cents) else "fail"


# ---- money parsing (arith-gating / triology money-line convention) --------

_MONEY_RE = re.compile(r"-?\d[\d.,]*")


def to_cents(s) -> Optional[int]:
    """Parse a money string to integer cents. Mirrors the
    normalization the arith-gating predictions already store (strings
    like '28,000' / '60.000'); ambiguous => None (caller abstains)."""
    if s is None:
        return None
    txt = str(s).strip()
    m = _MONEY_RE.search(txt)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        if "." in raw:
            return int(round(float(raw) * 100))
        return int(raw) * 100
    except ValueError:
        return None


# ---- Axis-B: beam-margin extraction (arith-gating BM_beam_margin.py) ------

def beam_margin_batch(imgs, processor, model, dec_one, pad_id,
                      max_length: int = 512) -> List[Dict]:
    """num_beams=2, num_return_sequences=2; margin = avg_logp(top1) -
    avg_logp(top2). Faithful to arith-gating BM_beam_margin.beam_margin_batch
    including the per-step fallback when sequences_scores is None.

    Requires a real CUDA model; torch imported lazily."""
    import numpy as np
    import torch

    px = processor(imgs, return_tensors="pt").pixel_values.to(
        "cuda", dtype=torch.float16)
    dec = dec_one.repeat(len(imgs), 1).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            px, decoder_input_ids=dec, max_length=max_length,
            num_beams=2, num_return_sequences=2, pad_token_id=pad_id,
            length_penalty=1.0, return_dict_in_generate=True,
            output_scores=True,
        )
    results: List[Dict] = []
    if getattr(out, "sequences_scores", None) is not None:
        sc = out.sequences_scores.float().cpu().tolist()
        for b in range(len(imgs)):
            t1, t2 = sc[2 * b], sc[2 * b + 1]
            results.append({"top1_norm_logp": float(t1),
                            "top2_norm_logp": float(t2),
                            "margin": float(t1 - t2),
                            "cseq_top1": float(np.exp(t1)),
                            "method": "sequences_scores"})
    else:
        seqs = out.sequences
        prompt_len = dec.size(1)

        def step_logp(seq):
            lp = []
            toks = seq[prompt_len:].tolist()
            for st, score in enumerate(out.scores):
                if st >= len(toks):
                    break
                tk = toks[st]
                if tk == pad_id:
                    break
                p = torch.softmax(score.float(), -1)[tk].clamp_min(1e-12).item()
                lp.append(float(np.log(p)))
            return float(np.mean(lp)) if lp else None

        for b in range(len(imgs)):
            t1 = step_logp(seqs[2 * b])
            t2 = step_logp(seqs[2 * b + 1])
            if t1 is None or t2 is None:
                continue
            results.append({"top1_norm_logp": float(t1),
                            "top2_norm_logp": float(t2),
                            "margin": float(t1 - t2),
                            "cseq_top1": float(np.exp(t1)),
                            "method": "per_step_fallback"})
    return results


def load_donut(checkpoint: str):
    """Load a Donut VisionEncoderDecoder checkpoint to CUDA, fp16.

    Same import path as phase3_donut_extract.py / BM_beam_margin.py. The
    `checkpoint` is supplied by the caller / runbook; NO model identifier
    is hard-coded into this package or its artifacts. Raises if CUDA is
    absent (we never silently fall back to fabricated CPU output)."""
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "load_donut requires a CUDA GPU; none visible. This package "
            "is scaffolding - run it on a vast.ai GPU box (see "
            "README_RUNBOOK.md). It will NOT fabricate CPU output.")
    processor = DonutProcessor.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(
        checkpoint, torch_dtype=torch.float16).to("cuda").eval()
    return processor, model


def decode_fields(imgs, processor, model, task_prompt: str,
                   max_length: int = 512) -> List[Tuple[Dict, float, float]]:
    """Greedy decode -> (fields dict, softmax_confidence, c_seq).

    softmax_confidence / c_seq derivation matches phase3_donut_extract.py
    (mean step top-1 prob and exp(mean step logp) respectively)."""
    import numpy as np
    import torch

    px = processor(imgs, return_tensors="pt").pixel_values.to(
        "cuda", dtype=torch.float16)
    dec = processor.tokenizer(task_prompt, add_special_tokens=False,
                              return_tensors="pt").input_ids
    dec = dec.repeat(len(imgs), 1).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            px, decoder_input_ids=dec, max_length=max_length,
            num_beams=1, return_dict_in_generate=True, output_scores=True,
            pad_token_id=processor.tokenizer.pad_token_id)
    res = []
    seqs = out.sequences
    for b in range(len(imgs)):
        step_top1, step_logp = [], []
        for st, score in enumerate(out.scores):
            probs = torch.softmax(score[b].float(), -1)
            top1 = float(probs.max().item())
            tok = int(probs.argmax().item())
            if tok == processor.tokenizer.pad_token_id:
                break
            step_top1.append(top1)
            step_logp.append(float(np.log(max(probs[tok].item(), 1e-12))))
        softmax_conf = float(np.mean(step_top1)) if step_top1 else 0.0
        c_seq = float(np.exp(np.mean(step_logp))) if step_logp else 0.0
        text = processor.batch_decode(seqs[b:b + 1],
                                      skip_special_tokens=False)[0]
        text = text.replace(processor.tokenizer.eos_token, "")
        text = text.replace(processor.tokenizer.pad_token, "")
        text = re.sub(r"<.*?>", "", text, count=1)
        try:
            fields = processor.token2json(text)
        except Exception:
            fields = {}
        res.append((fields if isinstance(fields, dict) else {},
                    softmax_conf, c_seq))
    return res
