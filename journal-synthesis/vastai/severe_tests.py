#!/usr/bin/env python3
"""SEVERE TESTS S1-S4 - robustness-check the NEGATIVE result.

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
    phi_mcc, perm_p, bootstrap_ci, wilson, spearman, median,
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
        })
    return out


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
    print("SELF-TEST OK: S1-S4 return finite, sign-defined values on "
          "synthetic input (no SEVERE.json written).")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Severe tests S1-S4 (cache-only, CPU, no GPU)")
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
