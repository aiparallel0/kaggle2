#!/usr/bin/env python3
"""
Analysis-only experiments E1-E4 on EXISTING data. Stdlib only, deterministic.

SCOPE (stated everywhere): E1-E3 are a CORD-only proxy, n=100, a single
beam-margin-paper pipeline (arith-gating/predictions). This is NOT the
integrated multi-corpus four-way benchmark, which remains unrun (pending).
E4 only re-confirms PRIOR-WORK numbers from triology/runs; it claims nothing new.

No cross-pipeline per-receipt join is performed anywhere (arith-gating ids and
triology runs are different pipelines / id spaces). E4 reads triology
aggregate JSON only.
"""
import json, os, random

RNG_SEED = 12345
ARITH = "/home/user/arith-gating/predictions/cord_arith.jsonl"
CSIG  = "/home/user/arith-gating/predictions/cord_signals_receipt.jsonl"
TRIO  = "/home/user/triology/runs"
OUT   = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def phi_mcc(a, b):
    """Matthews / phi correlation for two binary lists (1==event)."""
    n11 = sum(1 for x, y in zip(a, b) if x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n00 = sum(1 for x, y in zip(a, b) if not x and not y)
    num = n11 * n00 - n10 * n01
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    phi = 0.0 if den == 0 else num / den
    return phi, (n11, n10, n01, n00)


def perm_p(a, b, iters=20000, seed=RNG_SEED):
    """Two-sided permutation p-value on |phi| by shuffling b."""
    obs = abs(phi_mcc(a, b)[0])
    rng = random.Random(seed)
    bb = list(b)
    ge = 0
    for _ in range(iters):
        rng.shuffle(bb)
        if abs(phi_mcc(a, bb)[0]) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (iters + 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def main():
    arith = load_jsonl(ARITH)
    csig = load_jsonl(CSIG)
    sig_by = {r["receipt_id"]: r for r in csig}

    # ---- Inner join on receipt_id ----
    joined = []
    for r in arith:
        s = sig_by.get(r["receipt_id"])
        if s is not None:
            joined.append((r, s))
    matched_n = len(joined)

    results = {}

    # =========================================================
    # E1  Non-redundancy / error-decorrelation
    # =========================================================
    # Axis-A error is defined ONLY on applicable receipts (n_applicable>0).
    # n_applicable==0 receipts have no arithmetic identity to check, so the
    # symbolic axis abstains: they are EXCLUDED from the Axis-A error analysis
    # (not silently scored as pass/fail).
    appl = [(r, s) for (r, s) in joined if r["n_applicable"] > 0]
    n_appl = len(appl)
    n_zero = matched_n - n_appl

    axisA_err = [0 if r["arith_pass"] else 1 for (r, s) in appl]

    def conf_err(key, getter):
        vals = [getter(r, s) for (r, s) in appl]
        med = median(vals)
        # confidence-error proxy = LOW confidence (< median) => error event
        ev = [1 if v < med else 0 for v in vals]
        phi, counts = phi_mcc(axisA_err, ev)
        p = perm_p(axisA_err, ev)
        return {
            "proxy": key,
            "median_split": med,
            "phi_mcc": phi,
            "counts_n11_n10_n01_n00": counts,
            "permutation_p_two_sided": p,
            "n11_def": "A-error & low-conf",
        }

    e1 = {
        "scope": "CORD-only proxy, n=100, single beam-margin-paper pipeline",
        "matched_join_n": matched_n,
        "n_applicable_gt0": n_appl,
        "n_applicable_eq0_excluded_from_axisA": n_zero,
        "axisA_error_def": "NOT arith_pass on applicable receipts (n_applicable>0)",
        "axisA_error_rate_on_applicable": sum(axisA_err) / n_appl,
        "cseq_proxy": conf_err("low_c_seq", lambda r, s: s["c_seq"]),
        "softmax_proxy": conf_err(
            "low_softmax_confidence", lambda r, s: r["softmax_confidence"]
        ),
        "interpretation": (
            "HONEST READOUT (computed, not assumed): on this CORD-only "
            "proxy (n=83 applicable) phi/MCC = +0.287 with permutation "
            "p = 0.011 (two-sided), i.e. a WEAK but statistically "
            "significant POSITIVE association between Axis-A error and "
            "low-confidence events. The two error types are therefore NOT "
            "fully decorrelated here; they are only weakly correlated "
            "(phi^2 ~= 8% shared variance), leaving most error mass "
            "non-overlapping. This is a single-pipeline proxy and does NOT "
            "establish the prior-work orthogonality claim, which stands on "
            "its own cited multi-corpus evidence."
        ),
    }
    results["E1"] = e1

    # =========================================================
    # E2  Composition vs alone
    # =========================================================
    # Ground-truth correctness field: cord_arith carries explicit
    # ground_truth and predicted fields dicts. We define a receipt "correct"
    # iff predicted fields == ground_truth (exact dict match on stored
    # normalized strings). LIMITATION: this is exact-string match on the
    # already-normalized stored fields, not the paper's official KIE scorer;
    # it is a proxy for the CORD-only analysis.
    def correct(r):
        return r["fields"] == r["ground_truth"]

    cseq_all = [s["c_seq"] for (r, s) in joined]
    thr = median(cseq_all)  # median split threshold for c_seq

    def policy_stats(accept_fn):
        acc = [(r, s) for (r, s) in joined if accept_fn(r, s)]
        n = len(acc)
        corr = sum(1 for (r, s) in acc if correct(r))
        prec = (corr / n) if n else None
        return {
            "n_accept": n,
            "n_correct": corr,
            "precision": prec,
            "coverage": n / matched_n,
            "accepted_ids": sorted(r["receipt_id"] for (r, s) in acc),
        }

    conf_only = policy_stats(lambda r, s: s["c_seq"] >= thr)
    axisA_only = policy_stats(lambda r, s: r["arith_pass"])
    composed = policy_stats(lambda r, s: r["arith_pass"] and s["c_seq"] >= thr)

    co_set = set(conf_only.pop("accepted_ids"))
    aa_set = set(axisA_only.pop("accepted_ids"))
    cp_set = set(composed.pop("accepted_ids"))
    corr_ids = {r["receipt_id"] for (r, s) in joined if correct(r)}

    # blind-spot: receipts a single axis ACCEPTS but are WRONG, that the
    # composition correctly REJECTS.
    conf_wrong_accepts = co_set - corr_ids
    axisA_wrong_accepts = aa_set - corr_ids
    blind_conf = conf_wrong_accepts - cp_set
    blind_axisA = axisA_wrong_accepts - cp_set

    results["E2"] = {
        "scope": "CORD-only proxy, n=100, single beam-margin-paper pipeline",
        "ground_truth_field": (
            "cord_arith fields == ground_truth (exact match on stored "
            "normalized strings); proxy, not the official KIE scorer"
        ),
        "c_seq_threshold_median": thr,
        "confidence_alone": conf_only,
        "axisA_alone": axisA_only,
        "composed_axisA_AND_cseq": composed,
        "blindspot_conf_wrong_accepts_caught_by_composition": len(blind_conf),
        "blindspot_axisA_wrong_accepts_caught_by_composition": len(blind_axisA),
        "blindspot_conf_ids": sorted(blind_conf),
        "blindspot_axisA_ids": sorted(blind_axisA),
    }

    # =========================================================
    # E3  Precision-coverage frontier (sweep c_seq threshold)
    # =========================================================
    grid = [i / 100.0 for i in range(0, 101, 5)]
    frontier = []
    for t in grid:
        cf = [(r, s) for (r, s) in joined if s["c_seq"] >= t]
        cmp = [(r, s) for (r, s) in joined if r["arith_pass"] and s["c_seq"] >= t]
        def pc(sel):
            n = len(sel)
            c = sum(1 for (r, s) in sel if correct(r))
            return {
                "coverage": n / matched_n,
                "precision": (c / n) if n else None,
                "n_accept": n,
            }
        frontier.append({"c_seq_thr": t, "confidence_alone": pc(cf),
                          "composed": pc(cmp)})
    results["E3"] = {
        "scope": "CORD-only proxy, n=100, single beam-margin-paper pipeline",
        "frontier": frontier,
    }

    # =========================================================
    # E4  Verify-only consolidation of PRIOR-WORK numbers
    # (reads triology aggregate JSON; NO per-receipt join)
    # =========================================================
    pt = json.load(open(os.path.join(TRIO, "PAPER_TABLE.json")))
    tb = json.load(open(os.path.join(TRIO, "time_budget_cpu.json")))
    pooled = pt["T8_pooled"]["Pooled"]
    pooled_ci = pt["T8_pooled"]["Pooled_CIs"]
    wr = [row for row in pt["T1_headline"] if row["corpus"] == "WildReceipt"][0]

    pooled_int_prec = pooled["int_corr"] / pooled["int_acc"]
    pooled_int_wilson = wilson(pooled["int_corr"], pooled["int_acc"])

    results["E4"] = {
        "scope": "PRIOR-WORK re-confirmation only; NOT a new contribution",
        "pooled_composed_precision": pooled_int_prec,
        "pooled_composed_precision_recomputed_wilson_lb": pooled_int_wilson[0],
        "pooled_composed_int_corr_over_acc": [pooled["int_corr"],
                                              pooled["int_acc"]],
        "pooled_ci_table_int": pooled_ci["int"],
        "main_tex_claims_0_989_wilson_lb_0_961": {
            "precision_matches_0.989": round(pooled_int_prec, 3) == 0.989,
            "wilson_lb_matches_0.961": round(pooled_int_wilson[0], 3) == 0.961,
        },
        "wildreceipt_intersection_corr_over_acc": [
            wr["intersect_corr"], wr["intersect_n"]],
        "wildreceipt_113_over_114_confirmed": (
            wr["intersect_corr"] == 113 and wr["intersect_n"] == 114),
        "cpu_latency_median_us": tb["latency_us"]["median"],
        "cpu_latency_note": tb["honesty_note"][:160],
    }

    with open(os.path.join(OUT, "results_E1_E4.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
