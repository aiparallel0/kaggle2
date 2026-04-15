"""Custom exceptions — DataError, TrainError, EvalError."""


class DataError(RuntimeError):
    """Raised when SROIE download or parsing fails."""


class TrainError(RuntimeError):
    """Raised when training diverges or a guardrail trips."""


class EvalError(RuntimeError):
    """Raised when evaluation metrics fail sanity checks."""
