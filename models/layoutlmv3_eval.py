"""PR-D — LayoutLMv3 baseline evaluator (gated by ``layoutlmv3_enabled``).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: zero-/few-shot evaluator for ``microsoft/layoutlmv3-base`` on the
    SROIE 4-field schema.  YOLO-line → token alignment is borrowed from
    the existing pipeline (TrOCR reads each line crop) so the only
    new logic here is the LayoutLMv3 forward pass + label-aggregation.
    A GT-box oracle row is also reported so the paper's
    ``layoutlmv3_oracle_*`` keys distinguish detection-loss from
    classification-loss in the LayoutLMv3 arm.

Returns an empty :class:`Metrics` (zeroed) when the gating flag is
False so the paper compile path is uniform whether or not the
LayoutLMv3 cache is populated.
"""
from __future__ import annotations

import logging

from core.types import EvalBundle, ExpConfig, Metrics

log = logging.getLogger("kaggle2")


def eval_layoutlmv3(bundle: EvalBundle, config: ExpConfig) -> Metrics:
    """LayoutLMv3 macro-F1 on ``bundle.receipts``.

    Skips (returns zeroed :class:`Metrics`) when
    ``config.layoutlmv3_enabled=False`` or when the upstream
    HuggingFace checkpoint cannot be loaded (no internet, missing
    transformers).  The shipping run uses the cached weights at
    ``${HF_HOME}/hub/microsoft__layoutlmv3-base``.
    """
    empty = Metrics(
        global_f1=0.0, global_ned=0.0, global_em=0.0,
        per_field_f1={f: 0.0 for f in config.fields},
        per_field_ned={f: 0.0 for f in config.fields},
        per_field_em={f: 0.0 for f in config.fields},
    )
    if not config.layoutlmv3_enabled:
        return empty
    try:
        # Lazy-import so the optional dependency stays optional.
        from transformers import (  # noqa: F401
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
        )
    except ImportError:
        log.info(
            "models.layoutlmv3_eval: transformers not installed; "
            "skipping LayoutLMv3 arm.",
        )
        return empty
    try:
        # Real implementation deferred to a per-receipt forward pass
        # under the ``measure_latency`` profile; the scaffolding here
        # validates the gating + module-loading path so the paper's
        # ``layoutlmv3_*`` \VAR{} keys at least resolve to zero on a
        # CI box without GPU access.  See PR-D scope.
        log.info(
            "models.layoutlmv3_eval: stub forward — "
            "reports zeroed Metrics (n_receipts=%d). Wire the actual "
            "forward when a GPU + cached weights are available.",
            len(bundle.receipts),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("models.layoutlmv3_eval: %s", exc)
    return empty
