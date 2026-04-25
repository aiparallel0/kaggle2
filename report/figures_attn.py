"""Per-receipt cross-attention heatmap figure (ICDAR interpretability story).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: consumes the ``results/attention_samples.npz`` written by
    :class:`models.pipeline_attn.AttentionSampler` and produces
    ``fig_attention_heatmap.pdf`` — a vertical stack of one
    ``4 × N`` heatmap per sampled test receipt.  The figure
    operationalises the interpretability claim in Section III:
    the attention assigner's field→line soft assignment is a direct
    visual explanation of every per-field prediction, whereas DONUT
    exposes no equivalent introspection.

    Kept separate from :mod:`report.figures_extra` so no single
    figure module crosses the 166-LOC per-file cap, and so the
    heatmap — which has the heaviest data-loading logic of the
    paper's figures — can be unit-tested in isolation.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

log = logging.getLogger("kaggle2")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_FIELDS = ("company", "date", "address", "total")


def render_attention_heatmap(results_dir: str, out_dir: str) -> str | None:
    """Render one ``4 × N`` cross-attention heatmap per sampled receipt.

    Returns the written path, or ``None`` when matplotlib or the
    source ``attention_samples`` artefact is missing.  Never raises:
    the paper stage must continue even if the pipeline-eval stage
    produced no attention samples (e.g.\\ CPU-only run, all receipts
    errored out on a stale checkpoint).
    """
    if not _HAS_MPL:
        warnings.warn(
            "matplotlib unavailable — skipping attention heatmap", stacklevel=2,
        )
        return None
    data = load_attention_samples(Path(results_dir))
    if not data:
        log.info(
            "attention_samples.npz missing in %s — skipping heatmap (run: eval)",
            results_dir,
        )
        return None
    paths, attns = data["image_paths"], data["attn"]
    n = len(paths)
    # ``constrained_layout`` resolves the suptitle ↔ subplot-title ↔
    # per-axes colorbar packing problem that ``tight_layout`` cannot:
    # without it, each subplot title overlapped the x-axis labels of
    # the panel above (visible as collisions in the source figure).
    fig, axes = plt.subplots(
        n, 1, figsize=(7.2, 2.0 * n + 0.6), constrained_layout=True,
    )
    if n == 1:
        axes = [axes]  # make iterable shape uniform across K=1 / K>1
    for ax, path, attn in zip(axes, paths, attns, strict=True):
        im = ax.imshow(attn, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_yticks(range(len(_FIELDS)))
        ax.set_yticklabels(_FIELDS, fontsize=9)
        ax.set_xlabel("Detected line index", fontsize=9)
        ax.set_title(Path(path).name, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        # Light row separators — peaks straddling the field boundary
        # become unambiguously row-attributed.  ``set_xlim`` keeps the
        # lines flush with the heatmap edges (imshow extends 0.5 beyond).
        n_lines = len(attn[0]) if attn and attn[0] else 0
        for k in range(1, len(_FIELDS)):
            ax.axhline(k - 0.5, color="white", linewidth=0.4, alpha=0.5)
        # Minor x-ticks at every detected line let the reader locate a
        # peak's exact line index without counting major-tick offsets.
        if n_lines:
            ax.set_xticks(range(0, n_lines), minor=True)
            ax.grid(which="minor", axis="x", color="white",
                    linewidth=0.15, alpha=0.25)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(
        "Attention-assigner cross-attention (field query → line)",
        fontsize=10,
    )
    out = str(Path(out_dir) / "fig_attention_heatmap.pdf")
    # ``bbox_inches="tight"`` is incompatible with ``constrained_layout``
    # and would re-introduce the spacing collapse we just fixed.
    fig.savefig(out)
    plt.close(fig)
    return out


def load_attention_samples(results_dir: Path) -> dict[str, Any] | None:
    """Load attention samples from ``.npz`` (preferred) or ``.json`` fallback.

    Returns a dict with keys ``image_paths`` and ``attn`` (a list of
    nested float lists of shape ``(F, N)``), or ``None`` when no
    source artefact is present or readable.  NumPy is an optional
    dependency: on import failure we transparently fall back to the
    JSON sidecar emitted by :class:`~models.pipeline_attn.AttentionSampler`.
    """
    npz_path = results_dir / "attention_samples.npz"
    if npz_path.exists():
        try:
            import numpy as np  # noqa: PLC0415 — optional dep
            arr = np.load(npz_path, allow_pickle=True)
            return {
                "image_paths": list(arr["image_paths"]),
                "attn": [list(a) for a in arr["attn"]],
            }
        except (ImportError, KeyError, OSError):
            return None
    json_path = results_dir / "attention_samples.json"
    if json_path.exists():
        try:
            return dict(json.loads(json_path.read_text()))
        except (json.JSONDecodeError, OSError):
            return None
    return None
