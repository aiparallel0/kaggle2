#!/usr/bin/env python3
"""SEVERE TESTS S1-S5 - robustness-check the NEGATIVE result.

S5 (added as a CONFOUND CHECK, not a thesis-rescue) asks whether the
"sequence length / c_seq separates in-dist from shift better than
beam_margin" finding is merely a CORPUS-LENGTH ARTIFACT, by recomputing
the separation AUROC of length / c_seq / beam_margin both full and
RESTRICTED to an overlapping-length stratum. Like S1-S4 its binning rule
and verdict rule are fixed before the data is read and written into the
JSON; no choice here can manufacture a margin 'rescue'. Honest scope: it
runs on whatever pair is decoded on the box (cord_dev vs wildreceipt),
which is NOT the paper's CORD->SROIE pair (no SROIE fetcher here).

PURPOSE (stated honestly, up front): the clean decode-once run already
produced a NEGATIVE for the journal thesis (H1 composed WORSE than every
single signal at matched cost; H3 margin-variance tracks DIFFICULTY not
SHIFT; H2 error-association weak and SIGN-UNSTABLE across runs). These
four tests exist ONLY to check whether that negative is ROBUST or merely
a small-n / OCR-cord_dev / difficulty-confound artifact. They are
explicitly designed to let the negative STAND if it is real. There is
NO tuning, threshold-picking, or stratification choice here that can
manufacture a positive: every decision rule is fixed before the data is
read and the interpretation rule is written into the JSON next to the
number so the verdict cannot be cherry-picked after the fact.

COST: zero GPU. This reads ONLY the existing decode-once cache
(results/<label>__<ckpthash>.records.jsonl) via the SAME cache-load
path the other CPU experiments use (common.records._try_load_cache),
never decodes, never imports torch. If a corpus cache is missing it
REFUSES to run (it does not invent records).

REUSE: all metrics come from common.metrics (phi_mcc, perm_p,
bootstrap_ci, wilson, spearman, variance) and the axis mapping is
byte-identical to e1e3_fullscale / e5_integrated_benchmark: Axis-A
error = applicable (subset_sum_verdict != "abstain") AND NOT arith_pass;
confidence-error = below-median c_seq; correctness = common.totals
.is_correct; difficulty proxy = 1 - c_seq. Nothing is reimplemented.

OUTPUT: results/SEVERE.json with a real `computed_on` stamp. Resumable
exactly like the other experiments (a SEVERE.json that already carries a
real computed_on is skipped by run_parallel.sh's already_done()).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# cache-load path ONLY (no decode); metrics reused verbatim.
from common.records import (  # noqa: E402
    _try_load_cache, cache_path, split_corpus_arg,
)
from common.metrics import (  # noqa: E402
    phi_mcc, perm_p, bootstrap_ci, wilson, spearman, median, auroc,
)
from common.totals import (  # noqa: E402
    gold_total_cents, gold_items_cents, gold_tax_cents,
    pred_total_cents, is_correct, subset_sum_verdict_prior,
)

SEED = 12345
TASK_PROMPT_DEFAULT = "<s_cord-v2>"


# ---------------------------------------------------------------------------
# Cache -> per-receipt analysis records (axis mapping byte-identical to
# e5_integrated_benchmark / e1e3_fullscale; NOTHING reinterpreted).
# ---------------------------------------------------------------------------

def _records_from_cache(label, path, checkpoint, task_prompt):
    """Load the existing decode-once cache for ONE corpus and build the
    per-receipt analysis tuples. REFUSES (returns None) if the cache is
    missing/incomplete - it never decodes and never invents records."""
    cpath = cache_path(os.path.join(HERE, "results"), label, checkpoint)
    prim = _try_load_cache(cpath, label, path, checkpoint, task_prompt)
    if prim is None:
        return None
    out = []
    for p in prim:
        fields = p["fields"] if isinstance(p.get("fields"), dict) else {}
        gt = p.get("gold")
        c_seq = p.get("c_seq")
        bm = p.get("beam_margin")
        gold_total = gold_total_cents(gt)
        items = gold_items_cents(gt)
        tau = gold_tax_cents(gt)
        pred_total = pred_total_cents(fields)
        verdict = subset_sum_verdict_prior(pred_total, items, tau)
        out.append({
            "receipt_id": f"{label}:{p.get('receipt_id')}",
            "corpus": label,
            "c_seq": c_seq,
            "beam_margin": bm,
            "arith_pass": (verdict == "pass"),
            "verdict": verdict,
            "correct": bool(is_correct(gt, fields)),
            # S5 length proxy (see _derived_pred_length): the cache schema
            # has NO explicit length field and does NOT retain the raw
            # decoded string (only the parsed token2json `fields` dict
            # survives). length is therefore a DERIVED PROXY = whitespace
            # token count of the deterministically serialized predicted
            # `fields`. None if `fields` is absent/empty (skipped, never
            # fabricated).
            "pred_len_proxy": _derived_pred_length(fields),
        })
    return out


def _derived_pred_length(fields):
    """DERIVED length proxy for one receipt.

    HONEST NOTE (also written into SEVERE.json): the decode-once cache
    schema (common/records.py) stores NO explicit per-record length and
    does NOT retain the raw decoded predicted string - only the parsed
    `token2json` `fields` dict survives the cache. So a true predicted
    token length is NOT recoverable here. We therefore use a PROXY:
    the whitespace-token count of the deterministically serialized
    predicted `fields` (sorted keys, stable separators). This is a
    monotone-ish stand-in for "how much structured content the model
    emitted", NOT the model's literal token count. Returns None if
    `fields` is missing/empty so the receipt is SKIPPED (never
    fabricated to a number)."""
    if not isinstance(fields, dict) or not fields:
        return None
    try:
        s = json.dumps(fields, sort_keys=True, ensure_ascii=False,
                        separators=(" ", " "))
    except (TypeError, ValueError):
        return None
    n = len(s.split())
    return n if n > 0 else None


def _axisA_err_event(r):
    """Axis-A error event on an APPLICABLE receipt (verdict != abstain):
    1 iff the subset-sum verifier did NOT pass. Identical to E1/E5."""
    return 0 if r["arith_pass"] else 1


def _conf_err_event_below_median(recs):
    """Confidence-error = below-median c_seq (E1/E5 convention). Returns
    a dict receipt_id -> {0,1} over the receipts that have a c_seq."""
    cvals = [r["c_seq"] for r in recs if r["c_seq"] is not None]
    med = median(cvals) if cvals else None
    ev = {}
    for r in recs:
        v = r["c_seq"]
        ev[r["receipt_id"]] = (
            1 if (v is not None and med is not None and v < med) else 0)
    return ev, med


# ---------------------------------------------------------------------------
# S1  DIFFICULTY-CONTROLLED DECORRELATION
# ---------------------------------------------------------------------------
# Bin receipts by difficulty proxy (1 - c_seq) into quartiles. WITHIN each
# stratum compute phi/MCC between the Axis-A error event and the
# confidence-error event, with permutation p and a bootstrap CI on phi.
# If the H2 association is just the difficulty confound it VANISHES (phi
# -> ~0, p not significant) within strata; if it persists with a STABLE
# SIGN across strata and corpora it is real. We report the sign-stability
# verbatim; we do NOT pick the stratum with the biggest effect.
def s1_difficulty_controlled(recs):
    appl = [r for r in recs if r["verdict"] != "abstain"
            and r["c_seq"] is not None]
    if len(appl) < 8:
        return {"n_applicable_with_cseq": len(appl),
                "status": "INSUFFICIENT (n<8 applicable with c_seq)"}
    # difficulty proxy = 1 - c_seq; quartile edges from THIS corpus's
    # own distribution (fixed rule, not chosen to favour any outcome).
    diff = sorted((1.0 - r["c_seq"]) for r in appl)
    n = len(diff)
    q = [diff[int(n * f)] if int(n * f) < n else diff[-1]
         for f in (0.25, 0.5, 0.75)]

    def stratum_of(r):
        d = 1.0 - r["c_seq"]
        if d <= q[0]:
            return 0
        if d <= q[1]:
            return 1
        if d <= q[2]:
            return 2
        return 3

    strata = []
    phis = []
    for s in range(4):
        sub = [r for r in appl if stratum_of(r) == s]
        if len(sub) < 4:
            strata.append({"stratum": s, "n": len(sub),
                           "status": "too small (n<4)"})
            continue
        a = [_axisA_err_event(r) for r in sub]
        cev, _ = _conf_err_event_below_median(sub)
        b = [cev[r["receipt_id"]] for r in sub]
        phi, counts = phi_mcc(a, b)
        # paired-list bootstrap of phi over receipt indices.
        idx = list(range(len(sub)))

        def phi_stat(sampled_idx, _a=a, _b=b):
            aa = [_a[i] for i in sampled_idx]
            bb = [_b[i] for i in sampled_idx]
            return phi_mcc(aa, bb)[0]

        obs, lo, hi = bootstrap_ci(idx, stat=phi_stat, seed=SEED)
        p = perm_p(a, b, seed=SEED)
        phis.append(phi)
        strata.append({
            "stratum": s, "n": len(sub),
            "phi_mcc": phi, "counts_n11_n10_n01_n00": counts,
            "phi_bootstrap_ci95": [lo, hi],
            "permutation_p_two_sided": p})
    signs = [(1 if x > 0 else -1 if x < 0 else 0) for x in phis if x != 0.0]
    sign_stable = len(set(signs)) <= 1 and len(signs) >= 2
    return {
        "n_applicable_with_cseq": len(appl),
        "difficulty_proxy": "1 - c_seq",
        "quartile_edges": q,
        "strata": strata,
        "sign_stable_within_strata": sign_stable,
        "phi_values_by_stratum": phis,
        "interpretation_rule": (
            "If |phi|->~0 and permutation p NOT significant within EVERY "
            "computable stratum, the H2 association was the DIFFICULTY "
            "CONFOUND (robust-negative). It only 'survives' if phi stays "
            "non-trivial with a STABLE SIGN across strata AND corpora. "
            "Sign-instability (sign flips across strata/corpora) = the "
            "association is an artifact, NEGATIVE STANDS."),
    }


# ---------------------------------------------------------------------------
# S2  PLACEBO-AXIS NEGATIVE CONTROL
# ---------------------------------------------------------------------------
# Replace real Axis-A with a RANDOM gate matched to Axis-A's empirical
# accept rate (seeded; averaged over >=200 placebo draws). Compare
# composed(real A, B) vs composed(placebo, B) on precision-at-matched-
# coverage with a bootstrap CI on the difference. If real-A composition
# is NOT reliably better than placebo composition, the two-axis story is
# illusory (real Axis-A adds nothing a coin flip at the same rate does
# not). We report delta + CI + verdict; no rate is tuned.
def s2_placebo_axis(recs, n_placebo=200):
    rs = [r for r in recs if r["beam_margin"] is not None]
    if len(rs) < 8:
        return {"n_with_beam_margin": len(rs),
                "status": "INSUFFICIENT (n<8 with beam_margin)"}
    margins = [r["beam_margin"] for r in rs]
    m_thr = median(margins)
    accept_B = {r["receipt_id"]: (r["beam_margin"] >= m_thr) for r in rs}
    # real Axis-A accept rate (over the SAME universe) -> placebo rate.
    n_realA = sum(1 for r in rs if r["arith_pass"])
    rate = n_realA / len(rs)
    corr = {r["receipt_id"]: r["correct"] for r in rs}

    def precision_of(accept_map):
        acc = [rid for rid in accept_map if accept_map[rid]]
        k = len(acc)
        if k == 0:
            return None, 0
        c = sum(1 for rid in acc if corr[rid])
        return c / k, k

    real_comp = {r["receipt_id"]: (r["arith_pass"] and accept_B[r["receipt_id"]])
                 for r in rs}
    real_prec, real_k = precision_of(real_comp)

    rng = random.Random(SEED)
    placebo_precs = []
    deltas = []
    for _ in range(n_placebo):
        plac = {}
        for r in rs:
            gate = rng.random() < rate
            plac[r["receipt_id"]] = gate and accept_B[r["receipt_id"]]
        pp, pk = precision_of(plac)
        if pp is not None and real_prec is not None:
            placebo_precs.append(pp)
            deltas.append(real_prec - pp)
    if not deltas:
        return {"n_with_beam_margin": len(rs), "axisA_accept_rate": rate,
                "status": "INSUFFICIENT (no placebo draw yielded accepts)"}
    obs, lo, hi = bootstrap_ci(deltas, seed=SEED)
    mean_plac = sum(placebo_precs) / len(placebo_precs)
    return {
        "n_with_beam_margin": len(rs),
        "axisA_empirical_accept_rate": rate,
        "n_placebo_draws": len(deltas),
        "beam_margin_threshold_median": m_thr,
        "real_composed_precision": real_prec,
        "real_composed_n_accept": real_k,
        "mean_placebo_composed_precision": mean_plac,
        "real_minus_placebo_precision_delta": obs,
        "delta_ci95": [lo, hi],
        "real_A_reliably_beats_placebo": (lo is not None and lo > 0),
        "interpretation_rule": (
            "real_A_reliably_beats_placebo is TRUE only if the bootstrap "
            "CI lower bound on (real - placebo) precision is > 0. If it is "
            "FALSE, real Axis-A composition is NOT distinguishable from a "
            "rate-matched coin flip composed with B: the two-axis story is "
            "ILLUSORY and the H1 NEGATIVE STANDS. A positive delta whose "
            "CI includes 0 is NOT evidence for the thesis."),
    }


# ---------------------------------------------------------------------------
# S3  POWER / MINIMUM DETECTABLE EFFECT
# ---------------------------------------------------------------------------
# Given observed n and the observed composed/best-single precisions,
# simulate the smallest TRUE H1 advantage (composed precision minus best
# single-baseline precision) that this n could detect at 80% power
# (alpha=0.05, paired-bootstrap-CI-excludes-0 decision rule, matching the
# H1 head-to-head test), and report the achieved post-hoc CI width. This
# decides CONCLUSIVE (an effect we'd care about is excluded) vs
# UNDERPOWERED (cannot distinguish "no effect" from "no power").
def s3_power_mde(recs):
    rs = [r for r in recs if r["c_seq"] is not None]
    if len(rs) < 8:
        return {"n": len(rs), "status": "INSUFFICIENT (n<8 with c_seq)"}
    n = len(rs)
    corr = {r["receipt_id"]: r["correct"] for r in rs}
    margins = [r["beam_margin"] for r in rs if r["beam_margin"] is not None]
    m_thr = median(margins) if margins else None
    c_thr = median([r["c_seq"] for r in rs])

    def policy_ids(accept):
        return [r["receipt_id"] for r in rs if accept(r)]

    conf_ids = set(policy_ids(lambda r: r["c_seq"] >= c_thr))
    axisA_ids = set(policy_ids(lambda r: r["arith_pass"]))
    axisB_ids = set(policy_ids(
        lambda r: r["beam_margin"] is not None and m_thr is not None
        and r["beam_margin"] >= m_thr))
    comp_ids = set(policy_ids(
        lambda r: r["arith_pass"] and r["beam_margin"] is not None
        and m_thr is not None and r["beam_margin"] >= m_thr))

    def prec(ids):
        k = len(ids)
        if k == 0:
            return None
        return sum(1 for i in ids if corr[i]) / k

    singles = {"confidence_alone": prec(conf_ids),
               "axisA_alone": prec(axisA_ids),
               "axisB_alone": prec(axisB_ids)}
    valid = {k: v for k, v in singles.items() if v is not None}
    best_single = max(valid.values()) if valid else None
    comp_prec = prec(comp_ids)

    # Post-hoc CI width on the OBSERVED H1 contrast (composed minus best
    # single) using the SAME paired-bootstrap-of-correctness-difference
    # the H1 head-to-head test uses.
    posthoc = None
    if best_single is not None and comp_prec is not None and valid:
        best_name = max(valid, key=valid.get)
        base_ids = {"confidence_alone": conf_ids, "axisA_alone": axisA_ids,
                    "axisB_alone": axisB_ids}[best_name]
        shared = sorted(comp_ids | base_ids)
        diffs = []
        for rid in shared:
            cc = 1 if corr.get(rid) else 0
            diffs.append((cc if rid in comp_ids else 0)
                         - (cc if rid in base_ids else 0))
        if diffs:
            o, lo, hi = bootstrap_ci(diffs, seed=SEED)
            posthoc = {"vs_best_single": best_name,
                       "observed_delta": o, "ci95": [lo, hi],
                       "ci_width": (hi - lo) if (lo is not None
                                                 and hi is not None) else None}

    # Minimum detectable effect: simulate paired Bernoulli outcomes for a
    # composed vs best-single contrast at this n; sweep the true precision
    # gap; the MDE is the smallest gap reaching 80% power under the
    # CI-excludes-0 rule. Base precision = observed best_single (honest
    # operating point); fixed seeds; NOT tuned to a desired answer.
    mde = None
    if best_single is not None:
        base_p = max(0.0, min(1.0, best_single))
        rng = random.Random(SEED)
        n_sim = 300
        for gap in [g / 100.0 for g in range(1, 51)]:
            comp_p = min(1.0, base_p + gap)
            hits = 0
            for _ in range(n_sim):
                diffs = []
                for _ in range(n):
                    bc = 1 if rng.random() < base_p else 0
                    cc = 1 if rng.random() < comp_p else 0
                    diffs.append(cc - bc)
                _o, lo, _hi = bootstrap_ci(diffs, iters=400,
                                           seed=rng.randrange(1 << 30))
                if lo is not None and lo > 0:
                    hits += 1
            if hits / n_sim >= 0.80:
                mde = {"min_detectable_precision_gap_at_80pct_power": gap,
                       "base_precision_used": base_p,
                       "n": n, "n_sim_per_gap": n_sim}
                break
        if mde is None:
            mde = {"min_detectable_precision_gap_at_80pct_power": None,
                   "note": ">0.50 gap still <80% power at this n",
                   "base_precision_used": base_p, "n": n}

    # Verdict rule: the clean H1 negatives were about -0.25..-0.41. If the
    # MDE is SMALLER than the magnitude of an effect we'd care about
    # (we fix the care-about threshold at 0.05 absolute precision, stated
    # here, not chosen post hoc) AND the post-hoc CI EXCLUDES a +0.05
    # advantage, the H1 negative is CONCLUSIVE; else UNDERPOWERED.
    care_about = 0.05
    conclusive = None
    if mde is not None and posthoc is not None:
        g = mde.get("min_detectable_precision_gap_at_80pct_power")
        hi = posthoc["ci95"][1]
        conclusive = (g is not None and g <= care_about
                      and hi is not None and hi < care_about)
    return {
        "n": n,
        "observed_singles_precision": singles,
        "observed_best_single": best_single,
        "observed_composed_precision": comp_prec,
        "posthoc_h1_contrast": posthoc,
        "minimum_detectable_effect": mde,
        "care_about_threshold_abs_precision": care_about,
        "h1_negative_conclusive": conclusive,
        "interpretation_rule": (
            "CONCLUSIVE iff (a) the 80%-power MDE at this n is <= the "
            "pre-stated care-about gap (0.05 abs precision) AND (b) the "
            "post-hoc CI on the observed composed-minus-best-single "
            "contrast EXCLUDES a +0.05 advantage. Otherwise UNDERPOWERED: "
            "the data cannot distinguish 'composition truly does not help' "
            "from 'too few receipts to tell'. UNDERPOWERED does NOT rescue "
            "the thesis; it only forbids calling the negative conclusive."),
    }


# ---------------------------------------------------------------------------
# S4  SPLIT-STABILITY
# ---------------------------------------------------------------------------
# Recompute the H1 matched-cost deltas (composed minus each single
# baseline) and the H3 Spearman(margin-variance, difficulty) and
# Spearman(margin-variance, c_seq) for EACH corpus separately AND pooled,
# each with bootstrap CIs, so we can see whether the negative is stable
# across the cord_dev/wildreceipt split or driven by one corpus / the OCR
# set. margin-variance is computed in fixed c_seq-difficulty bins (the
# only available within-corpus variance proxy in the cache).
def _h1_deltas(recs):
    rs = [r for r in recs if r["c_seq"] is not None]
    if len(rs) < 4:
        return {"status": "INSUFFICIENT (n<4)"}
    corr = {r["receipt_id"]: r["correct"] for r in rs}
    c_thr = median([r["c_seq"] for r in rs])
    margins = [r["beam_margin"] for r in rs if r["beam_margin"] is not None]
    m_thr = median(margins) if margins else None

    def ids(accept):
        return set(r["receipt_id"] for r in rs if accept(r))

    conf = ids(lambda r: r["c_seq"] >= c_thr)
    axA = ids(lambda r: r["arith_pass"])
    axB = ids(lambda r: r["beam_margin"] is not None and m_thr is not None
              and r["beam_margin"] >= m_thr)
    comp = ids(lambda r: r["arith_pass"] and r["beam_margin"] is not None
               and m_thr is not None and r["beam_margin"] >= m_thr)
    out = {}
    for name, base in (("confidence_alone", conf), ("axisA_alone", axA),
                       ("axisB_alone", axB)):
        shared = sorted(comp | base)
        diffs = []
        for rid in shared:
            cc = 1 if corr.get(rid) else 0
            diffs.append((cc if rid in comp else 0) - (cc if rid in base else 0))
        if diffs:
            o, lo, hi = bootstrap_ci(diffs, seed=SEED)
        else:
            o, lo, hi = None, None, None
        out[f"composed_minus_{name}"] = {
            "delta": o, "ci95": [lo, hi],
            "h1_supported": (lo is not None and lo > 0)}
    return out


def _h3_spearmans(recs):
    """Spearman(margin-variance, difficulty) and (margin-variance, c_seq)
    using fixed c_seq deciles as the within-corpus binning (the only
    variance proxy available in the cache: per-bin beam_margin variance).
    Fixed binning, not chosen to favour any sign."""
    rs = [r for r in recs if r["beam_margin"] is not None
          and r["c_seq"] is not None]
    if len(rs) < 12:
        return {"status": "INSUFFICIENT (n<12 with beam_margin & c_seq)"}
    rs_sorted = sorted(rs, key=lambda r: r["c_seq"])
    n = len(rs_sorted)
    nbins = 10
    bins = []
    for b in range(nbins):
        lo = int(n * b / nbins)
        hi = int(n * (b + 1) / nbins)
        chunk = rs_sorted[lo:hi]
        if len(chunk) < 2:
            continue
        ms = [r["beam_margin"] for r in chunk]
        mean = sum(ms) / len(ms)
        var = sum((x - mean) ** 2 for x in ms) / (len(ms) - 1)
        mean_cseq = sum(r["c_seq"] for r in chunk) / len(chunk)
        bins.append({"mean_c_seq": mean_cseq,
                     "difficulty": 1.0 - mean_cseq,
                     "margin_variance": var, "n": len(chunk)})
    if len(bins) < 3:
        return {"status": "INSUFFICIENT (fewer than 3 usable bins)"}
    mv = [b["margin_variance"] for b in bins]
    diff = [b["difficulty"] for b in bins]
    cseq = [b["mean_c_seq"] for b in bins]
    return {
        "n_bins": len(bins),
        "spearman_marginvar_vs_difficulty": spearman(diff, mv),
        "spearman_marginvar_vs_cseq": spearman(cseq, mv),
        "bins": bins,
    }


def s4_split_stability(per_corpus_recs, pooled):
    out = {"per_corpus": {}, "pooled": {}}
    for label, recs in per_corpus_recs.items():
        out["per_corpus"][label] = {
            "n": len(recs),
            "H1_matched_cost_deltas": _h1_deltas(recs),
            "H3_margin_variance": _h3_spearmans(recs)}
    out["pooled"] = {
        "n": len(pooled),
        "corpora": sorted(per_corpus_recs.keys()),
        "H1_matched_cost_deltas": _h1_deltas(pooled),
        "H3_margin_variance": _h3_spearmans(pooled)}
    out["interpretation_rule"] = (
        "The negative is STABLE only if the sign of every H1 "
        "composed-minus-single delta and of the H3 "
        "Spearman(margin-variance, difficulty) is the SAME across "
        "cord_dev-only, wildreceipt-only AND pooled. If a corpus flips "
        "the sign, the pooled negative is DRIVEN by the other corpus / "
        "the OCR set and the conclusion is corpus-specific, NOT a "
        "thesis-level refutation - reported as such, not hidden.")
    return out


# ---------------------------------------------------------------------------
# S5  MATCHED-LENGTH STRATIFIED AUROC  (CONFOUND CHECK; NOT thesis-rescue)
# ---------------------------------------------------------------------------
# Honest question: is "sequence length / c_seq separates in-dist from shift
# better than beam_margin" merely a CORPUS-LENGTH ARTIFACT? Two corpora that
# differ systematically in length will be trivially separable by length (and
# by anything length-correlated such as c_seq) regardless of any
# distribution-shift signal. S5 recomputes the separation AUROC of three
# scores (sequence length, c_seq, beam_margin) for in-dist(0) vs shift(1)
# both on the FULL pair and RESTRICTED to an overlapping-length stratum
# where the two corpora have comparable length. If beam_margin's matched
# AUROC does not overtake length/c_seq, the "length beats margin" finding is
# NOT just a length confound and the prior negative STANDS.
#
# ONE binning rule, fixed before the data is read, NOT searched: pooled-
# length DECILES over the pooled lengths of the ordered pair (10 equal-count
# quantile cut points on the union of both corpora's lengths); the matched
# stratum is exactly the receipts whose length falls in a decile bin that is
# populated by BOTH corpora (>=1 receipt from each). A 1:1 nearest-length-
# matched subsample is reported as a SECOND view only (not the verdict
# driver). ORIENTATION is fixed: AUROC(pos=shift scores, neg=in-dist
# scores) with the raw score (no abs, no sign search); a value <0.5 simply
# means in-dist scores higher and is reported verbatim.
#
# LENGTH SOURCE: prefer an explicit per-record length if the cache schema
# has one; the decode-once cache (common/records.py) has NONE and does NOT
# retain the raw decoded string, so length is the DERIVED PROXY
# `pred_len_proxy` (whitespace-token count of the deterministically
# serialized predicted `fields`; see _derived_pred_length). This is stated
# in the JSON. Receipts with no proxy (empty/absent fields) are SKIPPED,
# never fabricated.
#
# PRE-STATED VERDICT RULE (written next to the numbers, not chosen post
# hoc): the negative STANDS if, at matched length, c_seq AUROC and/or
# length AUROC is >= beam_margin AUROC (CIs considered). beam_margin is
# 'rescued' ONLY if its matched AUROC is the highest of the three AND its
# bootstrap-CI lower bound exceeds the point estimates of BOTH the others.
# AUROCs are folded to separation strength max(a, 1-a) for the comparison
# so an equally-strong but oppositely-oriented separator is not mistaken
# for weak; raw oriented AUROCs are reported too.

def _explicit_record_length(r):
    """Return an EXPLICIT per-record length iff the cache schema exposes
    one. The decode-once cache (common/records.py SCHEMA_VERSION=1) has
    NO such field, so this returns None and S5 falls back to the
    documented derived proxy. Kept as a hook so that if a future schema
    adds e.g. `pred_token_len` S5 uses it automatically and records the
    source as explicit."""
    for key in ("pred_token_len", "n_pred_tokens", "seq_len",
                "pred_len", "token_len"):
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v and v > 0:
            return float(v)
    return None


def _length_of(r):
    """(length, source) for one analysis record. Explicit field if the
    schema has one, else the derived whitespace-token proxy of the
    serialized predicted fields. None if neither is available (skip)."""
    ex = _explicit_record_length(r)
    if ex is not None:
        return ex, "explicit"
    lp = r.get("pred_len_proxy")
    if isinstance(lp, (int, float)) and lp is not None and lp > 0:
        return float(lp), "derived_proxy"
    return None, "none"


def _sep_strength(a):
    """Orientation-free separation strength of an AUROC: max(a, 1-a).
    0.5 == no separation; 1.0 == perfect (either orientation)."""
    if a is None:
        return None
    return a if a >= 0.5 else 1.0 - a


def _auroc_with_ci(pos, neg, seed=SEED):
    """Oriented AUROC(pos=shift, neg=in-dist) plus a percentile bootstrap
    95% CI obtained by resampling EACH class with replacement (the
    correct two-sample AUROC bootstrap). Returns dict or None."""
    if not pos or not neg:
        return None
    obs = auroc(pos, neg)
    if obs is None:
        return None
    rng = random.Random(seed)
    npos, nneg = len(pos), len(neg)
    boots = []
    for _ in range(2000):
        bp = [pos[rng.randrange(npos)] for _ in range(npos)]
        bn = [neg[rng.randrange(nneg)] for _ in range(nneg)]
        v = auroc(bp, bn)
        if v is not None:
            boots.append(v)
    lo = hi = None
    if boots:
        boots.sort()
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots)) - 1]
    return {"auroc": obs, "sep_strength": _sep_strength(obs),
            "ci95": [lo, hi], "n_pos": npos, "n_neg": neg and nneg}


def _score_aurocs(in_recs, shift_recs, with_ci):
    """AUROC of length / c_seq / beam_margin separating in-dist(0) from
    shift(1) over the supplied record subsets. Each receipt contributes
    a score only if that score is present (independent missingness)."""
    out = {}
    for name, getter in (
            ("length", lambda r: _length_of(r)[0]),
            ("c_seq", lambda r: r.get("c_seq")),
            ("beam_margin", lambda r: r.get("beam_margin"))):
        pos = [getter(r) for r in shift_recs]
        pos = [v for v in pos if isinstance(v, (int, float)) and v is not None]
        neg = [getter(r) for r in in_recs]
        neg = [v for v in neg if isinstance(v, (int, float)) and v is not None]
        if with_ci:
            res = _auroc_with_ci(pos, neg)
            out[name] = res if res is not None else {
                "auroc": None, "status": "INSUFFICIENT (a class empty)",
                "n_pos": len(pos), "n_neg": len(neg)}
        else:
            a = auroc(pos, neg) if pos and neg else None
            out[name] = {"auroc": a, "sep_strength": _sep_strength(a),
                         "n_pos": len(pos), "n_neg": len(neg)}
    return out


def _pooled_decile_edges(lengths):
    """9 equal-count quantile cut points (deciles) over the POOLED
    lengths. Fixed rule; not searched. Bin index in [0,9]."""
    s = sorted(lengths)
    n = len(s)
    return [s[min(n - 1, int(n * f))] for f in
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def _bin_of(x, edges):
    b = 0
    for e in edges:
        if x <= e:
            return b
        b += 1
    return b


def s5_matched_length_auroc(in_label, in_recs, shift_label, shift_recs):
    """One ordered pair (in_dist=0, shift=1). Full + matched-length
    stratified AUROC for length / c_seq / beam_margin, with the
    pre-stated verdict rule attached."""
    # length source: established from the records actually present.
    src_counts = {"explicit": 0, "derived_proxy": 0, "none": 0}
    for r in in_recs + shift_recs:
        src_counts[_length_of(r)[1]] += 1
    if src_counts["explicit"] > 0 and src_counts["derived_proxy"] == 0:
        length_source = "explicit_per_record_field"
    elif src_counts["derived_proxy"] > 0 and src_counts["explicit"] == 0:
        length_source = ("DERIVED_PROXY (whitespace-token count of the "
                         "deterministically serialized predicted `fields`; "
                         "the decode-once cache stores NO explicit length "
                         "and does NOT retain the raw decoded string, so "
                         "this is a PROXY for emitted structured content, "
                         "NOT the model's literal token count)")
    elif src_counts["explicit"] > 0 and src_counts["derived_proxy"] > 0:
        length_source = ("MIXED explicit+derived (reported; mixing is a "
                         "limitation - treat with caution)")
    else:
        return {
            "pair": f"{in_label}(0)_vs_{shift_label}(1)",
            "status": ("SKIPPED: no length available - cache has no "
                       "explicit length field and no receipt retained a "
                       "non-empty predicted `fields` to derive a proxy "
                       "from. A length is NOT fabricated."),
            "length_source": "none",
        }

    in_l = [(r, _length_of(r)[0]) for r in in_recs]
    in_l = [(r, L) for r, L in in_l if L is not None]
    sh_l = [(r, _length_of(r)[0]) for r in shift_recs]
    sh_l = [(r, L) for r, L in sh_l if L is not None]
    if not in_l or not sh_l:
        # A whole CLASS has zero usable length (no explicit field, no
        # derivable proxy on that side). A per-pair AUROC needs lengths
        # on BOTH sides; we SKIP rather than fabricate a length.
        empty = [lbl for lbl, lst in
                 ((in_label, in_l), (shift_label, sh_l)) if not lst]
        return {
            "pair": f"{in_label}(0)_vs_{shift_label}(1)",
            "length_source": length_source,
            "status": (f"SKIPPED: no length available for class(es) "
                       f"{empty} - the cache has no explicit length and "
                       f"no receipt on that side retained a non-empty "
                       f"predicted `fields` to derive a proxy from. A "
                       f"length is NOT fabricated."),
        }
    if len(in_l) < 4 or len(sh_l) < 4:
        return {
            "pair": f"{in_label}(0)_vs_{shift_label}(1)",
            "length_source": length_source,
            "status": (f"INSUFFICIENT (n<4 with a length per class: "
                       f"in={len(in_l)} shift={len(sh_l)})"),
        }

    full = _score_aurocs([r for r, _ in in_l], [r for r, _ in sh_l],
                         with_ci=False)

    pooled_lengths = [L for _, L in in_l] + [L for _, L in sh_l]
    edges = _pooled_decile_edges(pooled_lengths)
    in_bins = {}
    sh_bins = {}
    for r, L in in_l:
        in_bins.setdefault(_bin_of(L, edges), []).append((r, L))
    for r, L in sh_l:
        sh_bins.setdefault(_bin_of(L, edges), []).append((r, L))
    shared_bins = sorted(set(in_bins) & set(sh_bins))
    m_in = [rl for b in shared_bins for rl in in_bins[b]]
    m_sh = [rl for b in shared_bins for rl in sh_bins[b]]

    matched = None
    if len(m_in) >= 4 and len(m_sh) >= 4:
        matched = _score_aurocs([r for r, _ in m_in],
                                [r for r, _ in m_sh], with_ci=True)
    matched_status = (
        "OK" if matched is not None else
        f"INSUFFICIENT matched stratum (in={len(m_in)} shift={len(m_sh)}, "
        f"need >=4 each); no overtake claim is made on too-small n")

    # SECOND VIEW only: 1:1 nearest-length matched subsample (greedy on
    # sorted lengths). Reported, NOT the verdict driver.
    nn_view = None
    if m_in and m_sh:
        sm_in = sorted(m_in, key=lambda t: t[1])
        sm_sh = sorted(m_sh, key=lambda t: t[1])
        pairs_in, pairs_sh = [], []
        used = [False] * len(sm_sh)
        for r, L in sm_in:
            best, bd = -1, None
            for j in range(len(sm_sh)):
                if used[j]:
                    continue
                d = abs(sm_sh[j][1] - L)
                if bd is None or d < bd:
                    bd, best = d, j
            if best >= 0:
                used[best] = True
                pairs_in.append(r)
                pairs_sh.append(sm_sh[best][0])
        if len(pairs_in) >= 4:
            nn_view = _score_aurocs(pairs_in, pairs_sh, with_ci=True)
        nn_status = (f"1:1 nearest-length subsample n_pairs="
                     f"{len(pairs_in)}")
    else:
        nn_status = "no nearest-length subsample (empty shared stratum)"

    # ---- pre-stated verdict (CIs considered; folded sep-strength) ----
    verdict = {
        "rule": (
            "PRE-STATED, written here BEFORE interpreting the numbers and "
            "NOT tuned: the prior NEGATIVE STANDS if, at matched length, "
            "c_seq AUROC and/or length AUROC separation-strength is >= "
            "beam_margin's. beam_margin is 'RESCUED' only if (a) its "
            "matched separation-strength is the HIGHEST of the three AND "
            "(b) its bootstrap-CI lower bound (folded) EXCEEDS the point "
            "separation-strengths of BOTH c_seq and length. Anything else "
            "= negative stands. Orientation is fixed (pos=shift); raw "
            "oriented AUROCs are also reported. This rule is honest and "
            "cannot be cherry-picked after the fact."),
        "negative_stands": None,
        "beam_margin_rescued": None,
        "basis": "matched_overlapping_length_stratum",
    }
    if matched is not None:
        def ss(name):
            d = matched.get(name, {})
            return d.get("sep_strength")

        bm, cs, ln = ss("beam_margin"), ss("c_seq"), ss("length")
        if None not in (bm, cs, ln):
            highest = bm > cs and bm > ln
            bm_d = matched["beam_margin"]
            ci = bm_d.get("ci95") or [None, None]
            bm_lo = None
            if ci[0] is not None and ci[1] is not None:
                a = bm_d.get("auroc")
                bm_lo = (_sep_strength(ci[0]) if a is not None and a >= 0.5
                         else _sep_strength(ci[1]))
            rescued = bool(highest and bm_lo is not None
                           and bm_lo > cs and bm_lo > ln)
            verdict["beam_margin_rescued"] = rescued
            verdict["negative_stands"] = (not rescued)
            verdict["matched_sep_strength"] = {
                "beam_margin": bm, "c_seq": cs, "length": ln,
                "beam_margin_ci_lo_folded": bm_lo}
        else:
            verdict["negative_stands"] = None
            verdict["note"] = ("a matched AUROC was not computable; no "
                               "rescue can be claimed (negative not "
                               "overturned).")
    else:
        verdict["note"] = ("matched stratum too small to compute; the "
                           "prior negative is NOT overturned by S5 (an "
                           "uncomputable test cannot rescue the thesis).")

    return {
        "pair": f"{in_label}(0)_vs_{shift_label}(1)",
        "label_convention": f"0={in_label} (in-dist), 1={shift_label} (shift)",
        "length_source": length_source,
        "length_source_record_counts": src_counts,
        "auroc_orientation": ("AUROC(pos=shift scores, neg=in-dist "
                              "scores); raw score, no abs/sign search; "
                              "<0.5 = in-dist scores higher (reported "
                              "verbatim)"),
        "binning_rule": ("pooled-length DECILES (9 equal-count quantile "
                         "cuts on the union of both corpora's lengths); "
                         "matched stratum = receipts whose length falls "
                         "in a decile bin populated by BOTH corpora. ONE "
                         "fixed rule, not searched."),
        "n_full": {in_label: len(in_l), shift_label: len(sh_l)},
        "full_auroc_unmatched": full,
        "pooled_decile_edges": edges,
        "matched_stratum_status": matched_status,
        "n_matched_stratum": {in_label: len(m_in), shift_label: len(m_sh)},
        "matched_stratum_auroc_with_ci": matched,
        "nearest_length_1to1_view": nn_view,
        "nearest_length_1to1_status": nn_status,
        "verdict": verdict,
    }


def s5_all_pairs(per_corpus_recs):
    """Every ORDERED pair among the corpora actually present in the
    cache. Honest scope statement is attached (single available shift
    pair on the box; NOT the paper's CORD->SROIE)."""
    labels = sorted(per_corpus_recs.keys())
    pairs = {}
    for a in labels:
        for b in labels:
            if a == b:
                continue
            key = f"{a}__vs__{b}"
            pairs[key] = s5_matched_length_auroc(
                a, per_corpus_recs[a], b, per_corpus_recs[b])
    return {
        "corpora_present": labels,
        "honest_scope": (
            "S5 runs on whatever corpora are decoded on THIS box - here "
            "cord_dev (OCR-derived CORD validation split, n~100) vs "
            "wildreceipt (n~472). This is NOT the paper's CORD->SROIE "
            "pair: SROIE has NO fetcher in these repos and cannot be "
            "replicated here. S5 answers the matched-length confound "
            "question for THIS single available shift pair only, at "
            "small matched-n; it is explicitly NOT a generalizable "
            "result and NOT a CORD->SROIE / paper-level result."),
        "pairs": pairs,
        "interpretation_rule": (
            "Per pair see verdict.rule. Negative stands unless "
            "beam_margin is RESCUED by that pre-stated rule. Given the "
            "prior length-confound-removed numbers (c_seq AUROC ~0.976 "
            "vs beam_margin ~0.809) and the negative S1-S4, a rescue is "
            "a priori unlikely; reported either way, NOT tuned."),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _self_test():
    """Dry self-test on a tiny SYNTHETIC in-memory record list: confirms
    S1-S4 functions return finite, sign-correct values. Writes NOTHING.
    NOT a substitute for a real run - it only proves the code path is
    sound when fed records (a real SEVERE.json requires the real cache)."""
    rng = random.Random(SEED)
    recs = []
    for i in range(120):
        cs = rng.random()
        # construct a KNOWN association: low c_seq -> more Axis-A errors,
        # so S1 should see a positive phi that we expect to (partly)
        # collapse within difficulty strata; S3 best_single finite.
        ap = (rng.random() < (0.35 + 0.5 * cs))
        bm = rng.gauss(cs, 0.1)
        correct = (ap and rng.random() < 0.85) or (rng.random() < 0.4)
        recs.append({
            "receipt_id": f"syn:{i}", "corpus": "syn",
            "c_seq": cs, "beam_margin": bm,
            "arith_pass": ap,
            "verdict": "pass" if ap else "fail",
            "correct": bool(correct)})
    s1 = s1_difficulty_controlled(recs)
    s2 = s2_placebo_axis(recs, n_placebo=50)
    s3 = s3_power_mde(recs)
    s4 = s4_split_stability({"syn": recs}, recs)
    import math as _m

    def finite(x):
        return isinstance(x, (int, float)) and _m.isfinite(x)

    assert "strata" in s1 and any("phi_mcc" in st for st in s1["strata"]), s1
    assert finite(s2["real_minus_placebo_precision_delta"]), s2
    assert finite(s3["observed_best_single"]), s3
    sp = s4["pooled"]["H3_margin_variance"].get(
        "spearman_marginvar_vs_difficulty")
    assert sp is None or finite(sp), s4

    # --- AUROC function: known-ordering => 1.0, reversed => 0.0,
    #     random ties => ~0.5 (stdlib, in-memory, writes NOTHING) ---
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0, "AUROC known-order != 1.0"
    assert auroc([0, 1, 2], [3, 4, 5]) == 0.0, "AUROC reversed != 0.0"
    assert auroc([1, 1, 1, 1], [1, 1, 1, 1]) == 0.5, "all-ties != 0.5"
    rng2 = random.Random(SEED)
    pos = [rng2.randint(0, 5) for _ in range(400)]
    neg = [rng2.randint(0, 5) for _ in range(400)]
    a_rand = auroc(pos, neg)
    assert 0.42 < a_rand < 0.58, f"random-ties AUROC {a_rand} not ~0.5"

    # --- matched-stratum logic on SYNTHETIC two-length-distribution
    #     data: in-dist short (lengths ~10), shift long (lengths ~40),
    #     overlapping band ~20-30. Full length AUROC must be near-perfect;
    #     in the matched overlap it must collapse toward 0.5. beam_margin
    #     here carries NO real shift signal, so it must NOT be 'rescued':
    #     verdict.negative_stands must be True (or None if uncomputable),
    #     never False. This validates the binning + verdict path. ---
    rng3 = random.Random(SEED)

    def mk(label, lo, hi, n):
        rs = []
        for i in range(n):
            L = rng3.uniform(lo, hi)
            rs.append({
                "receipt_id": f"{label}:{i}", "corpus": label,
                "c_seq": rng3.random(),
                "beam_margin": rng3.gauss(0.0, 1.0),
                "arith_pass": False, "verdict": "fail",
                "correct": False,
                "pred_len_proxy": int(round(L))})
        return rs
    in_recs = mk("indist", 5, 28, 90)
    sh_recs = mk("shift", 22, 55, 90)
    s5 = s5_all_pairs({"indist": in_recs, "shift": sh_recs})
    pr = s5["pairs"]["indist__vs__shift"]
    fa = pr["full_auroc_unmatched"]["length"]["sep_strength"]
    assert fa is not None and fa > 0.85, f"full length sep {fa} not strong"
    msa = pr["matched_stratum_auroc_with_ci"]
    if msa is not None:
        ml = msa["length"]["sep_strength"]
        assert ml is not None and ml < fa, (
            f"matched length sep {ml} did not collapse below full {fa}")
    vd = pr["verdict"]["negative_stands"]
    assert vd in (True, None), (
        f"synthetic no-margin-signal data must not 'rescue' margin; "
        f"negative_stands={vd}")
    # skip-on-no-length path: records with no length and no proxy.
    no_len = [{"receipt_id": f"x:{i}", "corpus": "x", "c_seq": 0.5,
               "beam_margin": 0.1, "arith_pass": False, "verdict": "f",
               "correct": False, "pred_len_proxy": None} for i in range(20)]
    s5b = s5_all_pairs({"x": no_len, "shift": sh_recs})
    assert "SKIPPED" in s5b["pairs"]["x__vs__shift"]["status"], \
        "no-length pair must be SKIPPED, not fabricated"
    print("SELF-TEST OK: S1-S4 return finite/sign-defined values; AUROC "
          "passes known-order(1.0)/reversed(0.0)/random-ties(~0.5); S5 "
          "matched-stratum collapses the length confound and does NOT "
          "rescue a no-signal beam_margin; no-length pair is SKIPPED. "
          "(no SEVERE.json written)")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Severe tests S1-S5 (cache-only, CPU, no GPU)")
    ap.add_argument("--checkpoint", required=False,
                    help="KIE checkpoint id/path (to locate the cache "
                         "file; same value run_parallel.sh passed to "
                         "Stage A). Required unless --self_test.")
    ap.add_argument("--task_prompt", default=TASK_PROMPT_DEFAULT)
    ap.add_argument("--corpora", nargs="+",
                    help="label=path ... (same as the other experiments)")
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "SEVERE.json"))
    ap.add_argument("--self_test", action="store_true",
                    help="run the in-memory synthetic self-test and exit "
                         "(writes nothing)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.checkpoint or not args.corpora:
        print("FATAL: --checkpoint and --corpora are required for a real "
              "run (or pass --self_test).", file=sys.stderr)
        sys.exit(2)

    per_corpus = {}
    missing = []
    for cp in args.corpora:
        label, path = split_corpus_arg(cp)
        recs = _records_from_cache(label, path, args.checkpoint,
                                   args.task_prompt)
        if recs is None:
            missing.append(label)
            continue
        per_corpus[label] = recs

    if missing:
        # REFUSE to run on an absent cache; do NOT invent records.
        print("FATAL: missing/incomplete decode-once cache for corpora: "
              + ", ".join(missing) + ". The severe tests are cache-only "
              "and will NOT decode or fabricate. Run Stage A first.",
              file=sys.stderr)
        sys.exit(3)
    if not per_corpus:
        print("FATAL: no corpora resolved.", file=sys.stderr)
        sys.exit(3)

    pooled = [r for recs in per_corpus.values() for r in recs]

    payload = {
        "experiment": "SEVERE",
        "purpose": ("robustness-check the NEGATIVE result (H1/H2/H3 "
                    "appeared to refute the thesis). These tests are "
                    "designed to let the NEGATIVE STAND if it is real; "
                    "no tuning/threshold-picking/stratification here can "
                    "manufacture a positive."),
        "scope": ",".join(f"{c}(n={len(r)})"
                          for c, r in sorted(per_corpus.items())),
        "per_corpus": {},
        "pooled": {},
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    for label, recs in sorted(per_corpus.items()):
        payload["per_corpus"][label] = {
            "n": len(recs),
            "S1_difficulty_controlled_decorrelation":
                s1_difficulty_controlled(recs),
            "S2_placebo_axis_negative_control": s2_placebo_axis(recs),
            "S3_power_minimum_detectable_effect": s3_power_mde(recs),
        }
    payload["pooled"] = {
        "n": len(pooled),
        "S1_difficulty_controlled_decorrelation":
            s1_difficulty_controlled(pooled),
        "S2_placebo_axis_negative_control": s2_placebo_axis(pooled),
        "S3_power_minimum_detectable_effect": s3_power_mde(pooled),
        "S4_split_stability": s4_split_stability(per_corpus, pooled),
    }
    payload["S5"] = s5_all_pairs(per_corpus)

    # write_result-equivalent guard: refuse without a real computed_on.
    if "computed_on" not in payload:
        raise ValueError("missing computed_on")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)),
                exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("per_corpus", "pooled")}, indent=2))
    print(f"[severe] wrote {args.out_json}")


if __name__ == "__main__":
    main()
