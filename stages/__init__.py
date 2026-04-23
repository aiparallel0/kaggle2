"""kaggle2 pipeline stages: ``train``, ``eval``, ``paper``.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: public package surface re-exporting the four top-level stage
    functions invoked by ``main.py``.  The package split keeps every
    module under the 166-LOC cap while preserving the legacy
    ``from stages import stage_train, stage_eval, ...`` contract.
"""
from __future__ import annotations

from stages.eval import stage_eval, stage_eval_gtocr_rulebased
from stages.paper import stage_paper
from stages.train import stage_train

__all__ = [
    "stage_eval",
    "stage_eval_gtocr_rulebased",
    "stage_paper",
    "stage_train",
]
