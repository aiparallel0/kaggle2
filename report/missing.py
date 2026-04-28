"""Typed ``\\MissingCell{key}`` markers for unresolved-but-required values.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: replace silent ``---`` em-dash backstops with a typed marker the
    LaTeX compile, build pipeline, and reviewer can all see.  A
    ``\\MissingCell{<key>}`` renders red in the PDF and is counted by
    :mod:`report.check_artefacts` as a hard build failure — unless the
    key is on an explicit allow-list (``MISSING_OK_KEYS``) of metrics
    we deliberately do not measure on a given build profile.

Two failure-recovery modes coexist:

  * Permissive (default, ``KAGGLE2_STRICT=0``): unresolved keys render
    as ``\\MissingCell{key}`` and are logged at WARNING; the PDF still
    compiles so reviewers see exactly which producers did not run.
  * Strict (``KAGGLE2_STRICT=1`` or ``config.strict_paper=true``):
    paper-stage raises ``EvalError`` listing every missing required
    key so a CI gate can fail the run before a half-empty PDF ships.

The allow-list is the single source of truth for "this cell is
intentionally not measured on this profile" — if you want to skip
producing latency on a CPU-only run, add ``donut_latency_*`` to
``MISSING_OK_KEYS`` and document why in the docstring.  Em-dash leaks
are then categorically prevented by ``check_artefacts``.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable

_log = logging.getLogger("kaggle2")

# Keys that may legitimately be absent on certain build profiles.
# Each prefix entry matches ``key.startswith(prefix)``; full matches
# require an exact string.  Keep this list short and documented —
# extension is fine, but every entry should explain itself.
MISSING_OK_PREFIXES: tuple[str, ...] = (
    # Foundation-model arm is opt-in; default is the conservative fixture
    # which exposes only a handful of foundation_* keys.  When the
    # flagship API call did NOT run we accept that the per-field
    # foundation_f1_* keys may be partial.
    "foundation_em_",
    "foundation_ned_",
    # RAG arm is opt-in; the headline IS the rag-off arm so rag_on_*
    # cells legitimately go missing on the default headline build.
    "rag_on_",
    # GAT arm is opt-in.
    "gat_",
    # Per-bug CI when single-seed (deltas resolve from bug_timeline.json
    # but CI bounds collapse — the heal path leaves them populated, but
    # the CI fields themselves are advisory).
    # Latency stage scoped-out (TRACKING.md): requires a dedicated
    # --stage latency that measures per-inference timing; absent on
    # standard train+eval runs.  Latency cells render \textit{n/a} so
    # the table compiles without a build blocker. Applies to both
    # template_basic.tex and template_advanced.tex.
    "donut_latency_",
    "pipeline_latency_",
    "donut_throughput_",
    "pipeline_throughput_",
    # Canonical-SROIE mode (``config.canonical_sroie_enabled=True``) runs
    # on the 347-image official ICDAR-2019 Task-3 test set, which ships
    # *without* GT bounding boxes.  ``stages/eval.py`` therefore skips
    # the GT-OCR rule-based baseline and oracle-patch diagnostic — see
    # ``_GTOCR_STRIP_PREFIXES`` there — and the matching ``\VAR{}`` keys
    # legitimately go missing on those builds.  Mirror the strip-prefix
    # tuple here so the paper compiles with ``\MissingCell{}`` markers
    # instead of failing the unresolved-VAR audit.
    "gtocr_rulebased_",
    "rulebased_",
    "oracle_patch_",
)

MISSING_OK_KEYS: frozenset[str] = frozenset({
    # Single-seed runs do not produce mean/std fan-out — the inject
    # layer renders the bare value in that case (see
    # ``inject._has_multi_seed``).
    "donut_f1_std", "pipeline_f1_std",
    "donut_f1_mean", "pipeline_f1_mean",
    # CostDiagnostics schema keys (wall_seconds + energy_wh) are
    # defined in core/schemas.py but the producer (--stage latency or
    # the new CostDiagnostics writer) is scoped-out.  Until that stage
    # lands, the training table renders these cells as \textit{n/a}.
    # Applies to both template_basic.tex and template_advanced.tex.
    "donut_wall_clock_s", "pipeline_wall_clock_s",
    "donut_energy_wh", "pipeline_energy_wh",
    # Per-image inference cost — emitted by the latency stage alongside
    # throughput measurements; scoped-out on standard train+eval runs.
    "donut_usd_per_img", "pipeline_usd_per_img",
    # McNemar p-value: only computed when both per-image correctness
    # vectors are present and equal-length.  Legitimately absent on
    # n=1 canonical-347 runs where the condition ``d_vec and p_vec and
    # len(d_vec) == len(p_vec)`` may not hold (e.g. skip_donut=True).
    # When absent, ``\\VAR{mcnemar_p:p}`` directives (handled by
    # ``report.inject_format.apply_formatters``) leave the placeholder
    # intact and the inject backstop converts it to ``\\MissingCell``.
    "mcnemar_p",
    # all_off row removed from bug-ablation table (Item 10): the floor
    # condition (all bugs reintroduced) is not a coherent single-run
    # measurement — each bug is a separate failure mode, not additive.
    # The table now shows the ceiling baseline (all bugs fixed) only.
    "all_off_delta", "all_off_ci_low", "all_off_ci_high",
    # Assigner sub-stage cost telemetry (A3): the assigner trains in
    # under a minute on commodity GPUs and we don't run the cost
    # collector for it (the collector amortises infrastructure cost
    # over the dominant DONUT/TrOCR stages).  These cells render
    # \textit{n/a} on every standard build until a dedicated
    # --stage cost_assigner producer lands.  Applies to template_basic
    # and template_advanced (Tables X & XI assigner sub-stage rows).
    "assigner_train_minutes", "assigner_peak_vram_gb",
    "assigner_cost_usd", "assigner_energy_kwh", "assigner_co2_kg",
    # Item 5: param ratio is computed from donut_params_m / pipeline_params_m.
    # Missing when either is absent (e.g., lightweight eval-only runs).
    "param_ratio_phrase", "param_ratio_numeric",
    # Item 15 (paper-corrections): assigner param count in millions —
    # derived in :func:`report.combine.merge_pipeline_diagnostics` from
    # ``assigner_params_k``; absent when the assigner did not run.
    "assigner_params_m",
    # Item 16 (paper-corrections): per-bug pre-fix F1 surfaced for the
    # bugs_code prose; absent if bug_timeline.json is unavailable.
    "bug_1_f1_before", "bug_2_f1_before", "bug_3_f1_before",
    "bug_4_f1_before", "bug_5_f1_before", "bug_6_f1_before",
    "bug_7_f1_before", "bug_8_f1_before", "bug_9_f1_before",
    "bug_10_f1_before", "bug_11_f1_before", "bug_12_f1_before",
    "bug_13_f1_before",
})


def is_missing_ok(key: str) -> bool:
    """Return True iff ``key`` is on the intentional-missing allow-list."""
    if key in MISSING_OK_KEYS:
        return True
    return any(key.startswith(p) for p in MISSING_OK_PREFIXES)


def render_missing_cell(key: str) -> str:
    r"""Return the LaTeX ``\MissingCell{<key>}`` literal for ``key``.

    ``\MissingCell`` is defined in ``report/template.tex`` as a small
    red typewriter macro: ``{\color{BrickRed}\textbf{?}\textsubscript{\tiny key}}``.
    Reviewers see immediately which producer did not write to
    ``runs/<run_id>/metrics/`` and the build can be configured (via
    ``KAGGLE2_STRICT``) to fail on its presence.

    Underscores are escaped as ``\_`` because ``_`` is the math-mode
    subscript operator in LaTeX/tectonic and causes ``Missing $ inserted``
    when it appears in text mode inside the macro argument.
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "_", key)[:48]
    safe_tex = safe.replace("_", r"\_")  # _ invalid in text mode; escape for tectonic
    return f"\\MissingCell{{{safe_tex}}}"


def is_strict() -> bool:
    """``True`` iff strict mode is active (env var or config flag)."""
    val = os.environ.get("KAGGLE2_STRICT", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def filter_blockers(unresolved: Iterable[str]) -> list[str]:
    """Return only the unresolved keys that should fail the build.

    The allow-list (``MISSING_OK_PREFIXES`` / ``MISSING_OK_KEYS``) is
    consulted; every other key is treated as a blocker so reviewers
    cannot accidentally ship a PDF with silent placeholders.
    """
    return sorted(k for k in unresolved if not is_missing_ok(k))


_RULEBASED_PREFIXES = ("gtocr_rulebased_", "rulebased_")
# Literals that belong to basic (500/63/63) split, forbidden in advanced sections.
_BASIC_SPLIT_PATTERNS = ("500/63/63", "500\\,/\\,63\\,/\\,63", "63-image test", "n=63")


def assert_no_rulebased_in_advanced(
    metrics: dict[str, object], variant: str,
) -> None:
    """Raise if advanced variant receives rulebased/gtocr_rulebased keys.

    The advanced variant purges all GT-OCR / rule-based baseline rows
    and must never inject metrics for those arms — doing so would
    indicate a producer mismatch or a merged config error.
    """
    if variant != "advanced":
        return
    bad = [k for k in metrics if any(k.startswith(p) for p in _RULEBASED_PREFIXES)]
    if bad:
        raise ValueError(
            f"advanced variant must not contain rulebased keys; found: {bad[:5]}"
        )


def assert_no_basic_split_in_advanced_sections(section_dir: str) -> None:
    """Raise if any advanced-specific .tex file mentions basic split literals.

    Advanced-specific sections (filenames containing 'advanced') must use
    test_set_size/test_set_kind VAR keys rather than hard-coded 500/63/63 refs.
    """
    import os
    import re
    for fname in os.listdir(section_dir):
        if "advanced" not in fname or not fname.endswith(".tex"):
            continue
        path = os.path.join(section_dir, fname)
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if line.lstrip().startswith("%"):
                    continue
                for pat in _BASIC_SPLIT_PATTERNS:
                    if re.search(re.escape(pat), line):
                        raise ValueError(
                            f"{fname}:{lineno}: basic split literal '{pat}' "
                            "forbidden in advanced sections"
                        )


_CI_FIELDS = ("company", "date", "address", "total")
_CI_SYSTEMS = ("donut", "pipeline")
_STALE_CI_GAP = 0.02


def assert_ci_bounds_valid(metrics: dict[str, object]) -> None:
    """Raise if any per-field CI bound violates lo ≤ point ≤ hi.

    Inspects two estimators per field and applies whichever check the
    sample shape supports:

      * ``<sys>_f1_<field>_mean`` — multi-seed mean-of-seed-means.
        Always bracketed when present (n>=2 runs).
      * ``<sys>_f1_<field>`` — bare point estimate written by
        :func:`report.combine.build_combined`, the per-image arithmetic
        mean of token-F1 (matched-statistic with the bootstrap in
        :func:`core.metrics_extended._bootstrap_field`).  On n=1 the
        bare point IS the bundle mean and must be bracketed; on n>=2
        the bare point is the last-seed value and is NOT bootstrap-
        comparable to a cross-seed CI, so the check is skipped (this
        preserves the multi-seed regression in
        :func:`tests.test_ci_bounds.test_point_outside_ci_but_mean_inside_does_not_raise`
        while still enforcing single-seed estimator-CI alignment).

    Both checks tolerate 1e-6 rounding.  Missing keys are silently
    skipped (single-seed runs may emit ``_ci_lo``/``_ci_hi`` as null).
    This is the regression check for the B1 producer-side bootstrap fix.

    PR #110 follow-up: when ``metrics["n_trials"] == 1`` a stale gap
    (>2 %) is now a hard failure rather than a WARNING.  The
    warn-only behaviour was a workaround for the asymmetric
    (normalised pred, raw gold) bundle that produced extreme stale
    gaps in single-seed runs; once :mod:`stages.eval_producers` plumbs
    each arm's normalised gold through, that asymmetry can no longer
    recur and the workaround should fail loudly.  Multi-seed runs
    (``n_trials >= 2``) keep the WARNING path because the stale-CI
    semantics there genuinely indicate an out-of-date sidecar rather
    than a producer-side bug.
    """
    tol = 1e-6
    n_trials_raw = metrics.get("n_trials")
    n_trials = int(n_trials_raw) if isinstance(n_trials_raw, int | float) else 1
    errs: list[str] = []
    stale_warns: list[str] = []
    for sys in _CI_SYSTEMS:
        for field in _CI_FIELDS:
            lo = metrics.get(f"{sys}_f1_{field}_ci_lo")
            hi = metrics.get(f"{sys}_f1_{field}_ci_hi")
            if not all(isinstance(x, int | float) for x in (lo, hi)):
                continue
            lof, hif = float(lo), float(hi)  # type: ignore[arg-type]
            mean = metrics.get(f"{sys}_f1_{field}_mean")
            point = metrics.get(f"{sys}_f1_{field}")
            checks: list[tuple[str, float]] = []
            if isinstance(mean, int | float):
                checks.append(("mean", float(mean)))
            # Only check the bare point when it matches the cross-seed
            # mean (single-seed run) or no mean is present — see docstring.
            single_seed = (
                isinstance(point, int | float)
                and (
                    not isinstance(mean, int | float)
                    or abs(float(point) - float(mean)) <= tol
                )
            )
            if single_seed and isinstance(point, int | float):
                checks.append(("point", float(point)))
            for label, vf in checks:
                if lof > vf + tol:
                    msg = f"{sys}_f1_{field}: ci_lo={lof:.4f} > {label}={vf:.4f}"
                    if lof - vf > _STALE_CI_GAP and n_trials > 1:
                        stale_warns.append(msg)
                    else:
                        errs.append(msg)
                if hif < vf - tol:
                    msg = f"{sys}_f1_{field}: ci_hi={hif:.4f} < {label}={vf:.4f}"
                    if vf - hif > _STALE_CI_GAP and n_trials > 1:
                        stale_warns.append(msg)
                    else:
                        errs.append(msg)
    if stale_warns:
        _log.warning(
            "CI bounds stale (extended_metrics.json is from a previous eval"
            " run — re-run --stage eval to refresh): %s",
            stale_warns[:5],
        )
    if errs:
        raise ValueError(f"CI bounds invalid: {errs[:5]}")
