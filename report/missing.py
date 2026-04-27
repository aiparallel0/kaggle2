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
