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
)

MISSING_OK_KEYS: frozenset[str] = frozenset({
    # Single-seed runs do not produce mean/std fan-out — the inject
    # layer renders the bare value in that case (see
    # ``inject._has_multi_seed``).
    "donut_f1_std", "pipeline_f1_std",
    "donut_f1_mean", "pipeline_f1_mean",
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
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "_", key)[:48]
    return f"\\MissingCell{{{safe}}}"


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
