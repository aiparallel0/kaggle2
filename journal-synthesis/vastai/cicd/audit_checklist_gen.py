#!/usr/bin/env python3
"""audit_checklist_gen.py - map results to pre-registered hypotheses.

It READS results/*.json + results/SEVERE.json + journal-synthesis/
PREREGISTRATION.md and EMITS AUDIT_CHECKLIST.md that tags each
pre-registered hypothesis (H1/H2/H3/H4 + severe S1-S4) with a measured
outcome and one of PASS / FAIL / UNDERPOWERED / ARTIFACT.

IT INTERPRETS, IT DOES NOT DECIDE. The checklist is explicit that a
HUMAN must sign off versus PREREGISTRATION.md before any number is moved
into the paper. This script:
  * NEVER reads or writes journal-synthesis/main.tex,
  * NEVER edits a paper claim,
  * NEVER auto-merges anything,
  * always prepends the DO-NOT-MERGE integrity banner.

Stdlib only. No network. Deterministic.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VASTAI_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(VASTAI_DIR, "..", ".."))
DEFAULT_RESULTS = os.path.join(VASTAI_DIR, "results")
DEFAULT_PREREG = os.path.join(REPO_ROOT, "journal-synthesis",
                              "PREREGISTRATION.md")

BANNER = """\
> # DO NOT MERGE INTO PAPER WITHOUT HUMAN SIGN-OFF vs PREREGISTRATION.md
>
> **This file and the PR carrying it are INFRASTRUCTURE OUTPUT ONLY.**
> A machine collected these raw numbers and tagged them against the
> frozen pre-registration. **It did NOT decide the science.** No number
> here may enter `journal-synthesis/main.tex` until a human has
> independently checked it against `journal-synthesis/PREREGISTRATION.md`
> and signed off. This PR is labelled `needs-human-audit` and MUST NOT
> be auto-merged. Fabrication is forbidden; every tag below is derived
> mechanically from the attached JSON.
"""

TAGS = {
    "PASS": "hypothesis SUPPORTED by the measured outcome",
    "FAIL": "hypothesis NOT supported (negative result; publishable)",
    "UNDERPOWERED": "data cannot distinguish; not a rescue of the thesis",
    "ARTIFACT": "outcome looks driven by a confound/artifact, not signal",
    "MISSING": "required result file/field absent; cannot tag",
}


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _find(results_dir, *names):
    for n in names:
        p = os.path.join(results_dir, n)
        if os.path.exists(p):
            d = _load_json(p)
            if d is not None:
                return n, d
    # glob fallback (e.g. E1E3_fullscale.json)
    for n in names:
        for p in glob.glob(os.path.join(results_dir, n)):
            d = _load_json(p)
            if d is not None:
                return os.path.basename(p), d
    return None, None


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def assess(results_dir):
    """Return a list of (id, title, measured, tag, note) rows.

    Tagging rules are MECHANICAL and conservative. When a definitive
    machine call is not possible the row is left for the human with an
    explicit note (never silently PASS)."""
    rows = []

    # H1 - integrated four-way head-to-head (E5).
    fn, e5 = _find(results_dir, "E5_integrated_benchmark.json")
    if e5 is None:
        rows.append(("H1", "Composed two-axis policy dominates all "
                     "three baselines at matched cost", "—", "MISSING",
                     "E5_integrated_benchmark.json not found"))
    else:
        rows.append(("H1", "Composed two-axis policy dominates all "
                     "three baselines at matched cost",
                     f"see `{fn}` (composed-vs-baseline contrasts)",
                     "HUMAN-REVIEW",
                     "Machine does NOT call dominance: the decision rule "
                     "(CIs not overlapping vs ALL three baselines) is a "
                     "human pre-registration check."))

    # H2 - non-redundancy / error decorrelation (E1E3).
    fn, e13 = _find(results_dir, "E1E3_fullscale.json",
                    "E1E3*.json", "E1*.json")
    if e13 is None:
        rows.append(("H2", "The two axes' errors are uncorrelated "
                     "beyond threshold", "—", "MISSING",
                     "E1E3_fullscale.json not found"))
    else:
        rows.append(("H2", "The two axes' errors are uncorrelated "
                     "beyond threshold",
                     f"see `{fn}` (phi/MCC + permutation p)",
                     "HUMAN-REVIEW",
                     "Sign-stability of the association across runs is a "
                     "pre-registration judgement; see SEVERE S1/S4."))

    # H3 - Axis-B mechanism (E7).
    fn, e7 = _find(results_dir, "E7_mechanism_synthetic_shift.json")
    if e7 is None:
        rows.append(("H3", "Margin-variance compression tracks shift "
                     "distance, not difficulty", "—", "MISSING",
                     "E7_mechanism_synthetic_shift.json not found"))
    else:
        rows.append(("H3", "Margin-variance compression tracks shift "
                     "distance, not difficulty",
                     f"see `{fn}`", "HUMAN-REVIEW",
                     "Whether it tracks SHIFT vs DIFFICULTY is exactly "
                     "the pre-registered call; SEVERE S1 robustness-"
                     "checks the difficulty confound."))

    # H4 - Axis-A robustness / verifier bake-off (E9).
    fn, e9 = _find(results_dir, "E9_alt_verifier_bakeoff.json")
    if e9 is None:
        rows.append(("H4", "Subset-sum not dominated by an alternative "
                     "structural verifier", "—", "MISSING",
                     "E9_alt_verifier_bakeoff.json not found"))
    else:
        rows.append(("H4", "Subset-sum not dominated by an alternative "
                     "structural verifier",
                     f"see `{fn}`", "HUMAN-REVIEW",
                     "Dominance call is a pre-registration judgement."))

    # Severe tests S1-S4 (SEVERE.json). Here the JSON encodes its own
    # interpretation rule, so we can surface a MECHANICAL tag while
    # still demanding human sign-off.
    fn, sev = _find(results_dir, "SEVERE.json")
    if sev is None:
        for sid in ("S1", "S2", "S3", "S4"):
            rows.append((sid, "severe robustness check", "—", "MISSING",
                         "SEVERE.json not found"))
    else:
        pooled = sev.get("pooled", {}) if isinstance(sev, dict) else {}
        s1 = pooled.get("S1_difficulty_controlled_decorrelation", {})
        s3 = pooled.get("S3_power_minimum_detectable_effect", {})
        s4 = pooled.get("S4_split_stability", {})

        # S1: if the association VANISHES once difficulty is controlled,
        # the H2 positive was a difficulty ARTIFACT (the JSON says so).
        s1_stable = s1.get("sign_stable_within_strata")
        s1_tag = ("ARTIFACT" if s1_stable is False
                  else "HUMAN-REVIEW" if s1_stable is None
                  else "PASS")
        rows.append(("S1", "Difficulty-controlled decorrelation",
                     f"sign_stable_within_strata={_fmt(s1_stable)}",
                     s1_tag,
                     "ARTIFACT => the H2 association was the difficulty "
                     "confound; human confirms vs pre-registration."))

        # S2: placebo / negative control.
        s2 = pooled.get("S2_placebo_axis_negative_control", {})
        rows.append(("S2", "Placebo-axis negative control",
                     f"see `{fn}` -> S2", "HUMAN-REVIEW",
                     "A positive placebo delta would mean the apparatus "
                     "is illusory; human reads the fixed verdict."))

        # S3: conclusive vs underpowered (verdict is in the JSON).
        concl = s3.get("h1_negative_conclusive")
        s3_tag = ("FAIL" if concl is True            # H1 negative CONCLUSIVE
                  else "UNDERPOWERED" if concl is False
                  else "HUMAN-REVIEW")
        rows.append(("S3", "Power / minimum detectable effect for H1",
                     f"h1_negative_conclusive={_fmt(concl)}", s3_tag,
                     "FAIL here = the H1 NEGATIVE is statistically "
                     "CONCLUSIVE per the pre-stated rule. UNDERPOWERED "
                     "does NOT rescue the thesis. Human signs off."))

        # S4: split stability.
        rows.append(("S4", "Split-stability of the negative",
                     f"see `{fn}` -> S4", "HUMAN-REVIEW",
                     "Whether the negative is stable across the "
                     "corpus split is a human pre-registration check."))
        _ = (s2, s4)  # surfaced via file pointers; not auto-decided
    return rows


def render(rows, results_dir, prereg_path):
    now = datetime.datetime.utcnow().isoformat() + "Z"
    prereg_present = os.path.exists(prereg_path)
    files = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(results_dir, "*.json")))
    out = []
    out.append(BANNER)
    out.append("")
    out.append("# AUDIT CHECKLIST (machine-generated, human-gated)")
    out.append("")
    out.append(f"- Generated (UTC): `{now}`")
    out.append(f"- Results dir scanned: `{results_dir}`")
    out.append(f"- Pre-registration: "
               f"`{prereg_path}` "
               f"({'present' if prereg_present else 'MISSING'})")
    out.append(f"- Result JSON files: "
               f"{', '.join('`%s`' % f for f in files) or '(none)'}")
    out.append("")
    out.append("Tag legend: " + "; ".join(
        f"**{k}** = {v}" for k, v in TAGS.items()) +
        "; **HUMAN-REVIEW** = a pre-registration judgement the machine "
        "deliberately does NOT make.")
    out.append("")
    out.append("| ID | Pre-registered expectation | Measured (raw) "
               "| Machine tag | Note for the human auditor |")
    out.append("|----|----------------------------|----------------"
               "|-------------|-----------------------------|")
    for cid, title, measured, tag, note in rows:
        title = title.replace("|", "\\|")
        note = note.replace("|", "\\|")
        measured = str(measured).replace("|", "\\|")
        out.append(f"| {cid} | {title} | {measured} | "
                   f"`{tag}` | {note} |")
    out.append("")
    out.append("## Mandatory human sign-off")
    out.append("")
    out.append("- [ ] I have compared every row above against "
               "`journal-synthesis/PREREGISTRATION.md` myself.")
    out.append("- [ ] The decision rules (CI overlap, dominance, "
               "shift-vs-difficulty) were applied by ME, not the "
               "machine.")
    out.append("- [ ] No number was moved into `main.tex` by this "
               "automation (it cannot; it never writes the paper).")
    out.append("- [ ] Any negative result is reported AS a negative "
               "(pre-registration commits to this).")
    out.append("- [ ] I accept scientific responsibility for any number "
               "I subsequently transcribe into the paper.")
    out.append("")
    out.append("_This automation interprets; it does not decide. "
               "The scientific decision stays human._")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate AUDIT_CHECKLIST.md (interprets results "
                    "vs pre-registration; never edits the paper).")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS)
    p.add_argument("--prereg", default=DEFAULT_PREREG)
    p.add_argument("--out", default=None,
                   help="output .md path (default: "
                        "<results-dir>/AUDIT_CHECKLIST.md)")
    args = p.parse_args(argv)

    out_path = args.out or os.path.join(args.results_dir,
                                        "AUDIT_CHECKLIST.md")
    # Hard guard: never allow this tool to target main.tex.
    if os.path.basename(out_path) == "main.tex" or \
       out_path.endswith("main.tex"):
        print("REFUSING: this tool never writes main.tex.",
              file=sys.stderr)
        return 2

    rows = assess(args.results_dir)
    md = render(rows, os.path.abspath(args.results_dir),
                os.path.abspath(args.prereg))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)),
                exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"[audit] wrote {out_path}")
    print(f"[audit] {len(rows)} hypotheses tagged; HUMAN sign-off "
          f"REQUIRED before any number enters the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
