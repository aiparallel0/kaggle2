"""Custom exceptions encoding the paper's failure taxonomy.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: three exception classes map to distinct failure modes in the Bugs
    section — DataError (SROIE download/parse), TrainError (divergence or
    silent F1-destroying bug guardrail trip), EvalError (metric sanity).
"""


class DataError(RuntimeError):
    """SROIE download or parsing failure (corrupted archive, missing files)."""


class TrainError(RuntimeError):
    """Training diverged or a Bug-class guardrail tripped (Bug 4/6/7/8)."""


class ConfigError(TrainError):
    """Architecture-flag invariant tripped at config load time (Bug 18).

    Subclasses ``TrainError`` so existing ``except TrainError`` handlers
    in the train/eval pipeline still surface the failure pre-GPU; the
    distinct class lets callers (and tests) pin the exact failure mode.
    """


class EvalError(RuntimeError):
    """Post-eval F1 fell below the architecture-specific bug floor."""
