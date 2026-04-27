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

import os
import re
from collections.abc import Iterable

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
    # Item 5: param ratio is computed from donut_params_m / pipeline_params_m.
    # Missing when either is absent (e.g., lightweight eval-only runs).
    "param_ratio_phrase", "param_ratio_numeric",
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


def assert_ci_bounds_valid(metrics: dict[str, object]) -> None:
    """Raise if any per-field CI bound violates lo ≤ mean ≤ hi.

    Inspects keys like donut_f1_company_mean, donut_f1_company_ci_lo,
    donut_f1_company_ci_hi.  Missing keys are silently skipped (single-seed
    runs emit neither _mean nor CI keys); present keys must satisfy
    ci_lo ≤ mean ≤ ci_hi (within 1e-6 tolerance for float rounding).
    """
    tol = 1e-6
    errs: list[str] = []
    for sys in _CI_SYSTEMS:
        for field in _CI_FIELDS:
            mean = metrics.get(f"{sys}_f1_{field}_mean")
            lo = metrics.get(f"{sys}_f1_{field}_ci_lo")
            hi = metrics.get(f"{sys}_f1_{field}_ci_hi")
            if not all(isinstance(x, int | float) for x in (mean, lo, hi)):
                continue
            mf, lof, hif = float(mean), float(lo), float(hi)  # type: ignore[arg-type]
            if lof > mf + tol:
                errs.append(f"{sys}_f1_{field}: ci_lo={lof:.4f} > mean={mf:.4f}")
            if hif < mf - tol:
                errs.append(f"{sys}_f1_{field}: ci_hi={hif:.4f} < mean={mf:.4f}")
    if errs:
        raise ValueError(f"CI bounds invalid: {errs[:5]}")
