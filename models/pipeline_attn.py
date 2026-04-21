"""Emit per-receipt attention samples for the paper's heatmap figure.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: lightweight side-writer invoked from :mod:`models.pipeline_eval`.
    For the first ``K=3`` successfully processed test receipts it
    captures the cross-attention tensor that
    :class:`~models.attention_model.AttentionAssigner` already produces
    on the forward pass (no re-run, no extra gradients), together with
    the YOLO bounding boxes and the image path, and persists the triple
    as ``results/attention_samples.npz``.  The
    :func:`report.figures_extra.render_attention_heatmap` emitter
    consumes this file to produce the per-receipt 4-query × N-line
    heatmap used in the paper (Fig.~\\ref{fig:attn_heatmap}).

    Kept separate from :mod:`models.pipeline_eval` so the eval module
    stays under the 166-LOC per-file cap: each side channel (metrics
    JSON, attention samples, future telemetry overlays) is added as a
    peer module rather than a growing pile of ``if sample_k_left > 0:``
    branches in the main receipt loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_K = 3
"""Number of receipts whose attention tensor the paper ships with.

Chosen to keep the ``.npz`` small enough to commit alongside other
``results/*.json`` artefacts (< 50 KB for typical SROIE receipts)
while still showing one failure case and two successes in the
heatmap grid.  Override at call time if a reviewer wants the full
63-receipt test split.
"""


class AttentionSampler:
    """Accumulate up to ``K`` ``(image, bboxes, attn)`` triples.

    The sampler is intentionally dumb: it records the first ``K``
    successfully processed receipts in test-set order, with no
    stratification or random selection.  Selection stability matters
    more than representativeness for the heatmap figure, which is
    used qualitatively rather than aggregated.
    """

    def __init__(self, k: int = DEFAULT_SAMPLE_K) -> None:
        self.k = max(0, int(k))
        self._image_paths: list[str] = []
        self._bboxes: list[list[list[float]]] = []
        self._attn: list[list[list[float]]] = []

    @property
    def full(self) -> bool:
        """True once ``K`` samples have been captured."""
        return len(self._image_paths) >= self.k

    def capture(
        self,
        image_path: str,
        bboxes: list[list[float]],
        attn_weights: Any,
    ) -> None:
        """Record one ``(image, bboxes, attn[F, N])`` triple if not full.

        ``attn_weights`` may be any array-like with shape ``(F, N)``
        (tensor or nested list).  We immediately convert to a plain
        ``list[list[float]]`` so the sampler has no hard dependency
        on NumPy or PyTorch — important because this module sits on
        the hot path of :func:`eval_pipeline` and must import cleanly
        in the lightweight CI sandbox.
        """
        if self.full or not bboxes:
            return
        try:
            attn_list = _as_nested_list(attn_weights)
        except (TypeError, ValueError):
            return
        self._image_paths.append(str(image_path))
        self._bboxes.append([list(b) for b in bboxes])
        self._attn.append(attn_list)

    def write(self, out_dir: Path) -> Path | None:
        """Persist the accumulated triples to ``attention_samples.npz``.

        Returns the written path, or ``None`` when no samples were
        captured (silent skip — the downstream figure emitter already
        warns loudly when its source file is absent).  NumPy is an
        optional dependency here; if it is not installed we fall back
        to a JSON sidecar so the data is not lost.
        """
        if not self._image_paths:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np  # noqa: PLC0415 — optional dep
        except ImportError:
            return self._write_json_fallback(out_dir)
        out = out_dir / "attention_samples.npz"
        np.savez(
            out,
            image_paths=np.array(self._image_paths, dtype=object),
            bboxes=np.array(self._bboxes, dtype=object),
            attn=np.array(self._attn, dtype=object),
        )
        return out

    def _write_json_fallback(self, out_dir: Path) -> Path:
        """JSON fallback when NumPy is unavailable (kept for CI sandboxes)."""
        import json  # noqa: PLC0415 — used only in fallback path
        out = out_dir / "attention_samples.json"
        out.write_text(json.dumps({
            "image_paths": self._image_paths,
            "bboxes": self._bboxes,
            "attn": self._attn,
        }, indent=2))
        return out


def _as_nested_list(attn_weights: Any) -> list[list[float]]:
    """Convert a tensor / ndarray / nested sequence to ``list[list[float]]``.

    Accepts any object exposing ``.tolist()`` (PyTorch tensor,
    NumPy ndarray) or an already-nested Python sequence.  Raises on
    anything else so the caller can ``except`` and skip cleanly.
    """
    raw = attn_weights.tolist() if hasattr(attn_weights, "tolist") else attn_weights
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
        raise TypeError("attn_weights must be convertible to list[list[float]]")
    return [[float(x) for x in row] for row in raw]
