"""Unified per-receipt record schema + writers.

THE reason E5 was blocked: arith-gating prediction ids and triology run
ids are different pipelines / id spaces, so no provably valid per-receipt
join existed. The fix, implemented by e5_integrated_benchmark.py, is to
re-run BOTH axes through ONE pipeline on a SHARED receipt set so the
`receipt_id` aligns BY CONSTRUCTION. Every experiment in this package
emits records in exactly this schema, which is the union of the schema in
experiments/results_E1_E4.json and the arith-gating prediction jsonl.

This module does NOT compute anything. It only validates + persists
records an experiment actually produced from real model output. There is
no default / placeholder numeric anywhere; a missing required field is a
hard error so a half-run can never masquerade as a result.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Union of: experiments/results_E1_E4.json fields used in E1-E3
# (receipt_id, c_seq, softmax_confidence, arith_pass, n_applicable,
# fields/ground_truth correctness) and the arith-gating prediction jsonl
# plus the Axis-B beam-margin signal from arith-gating BM_beam_margin.py.
REQUIRED_FIELDS = (
    "receipt_id",      # shared id; aligned by construction in E5
    "corpus",          # e.g. "cord", "sroie", "wildreceipt", "synthetic:<spec>"
    "backbone",        # e.g. "donut-cord", "layoutlmv3" (NO checkpoint hash)
    "gold_total",      # ground-truth total (cents int) or None if n/a
    "pred_total",      # decoded total (cents int) or None
    "softmax_confidence",  # arith-gating field
    "c_seq",           # sequence confidence (cord_signals_receipt schema)
    "arith_pass",      # Axis-A subset-sum verdict (bool) on applicable
    "subset_sum_verdict",  # explicit Axis-A verdict: "pass"|"fail"|"abstain"
    "beam_margin",     # Axis-B: avg_logp(top1)-avg_logp(top2) (float|None)
)


@dataclass
class UnifiedRecord:
    receipt_id: str
    corpus: str
    backbone: str
    gold_total: Optional[int]
    pred_total: Optional[int]
    softmax_confidence: Optional[float]
    c_seq: Optional[float]
    arith_pass: Optional[bool]
    subset_sum_verdict: str  # "pass" | "fail" | "abstain"
    beam_margin: Optional[float]
    # optional, experiment-specific extras (never required, never defaulted
    # to a fake value): correctness, n_applicable, shift label, etc.
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.subset_sum_verdict not in ("pass", "fail", "abstain"):
            raise ValueError(
                f"subset_sum_verdict must be pass|fail|abstain, "
                f"got {self.subset_sum_verdict!r} for {self.receipt_id}"
            )
        d = asdict(self)
        for k in REQUIRED_FIELDS:
            if k not in d:
                raise ValueError(f"record missing required field {k!r}")

    def to_json(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


def write_records(path: str, records: List[UnifiedRecord]) -> None:
    """Write the unified per-receipt jsonl. Fails loudly if a record is
    incomplete (so a crashed run cannot produce a deceptively partial,
    'looks finished' file)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")


def write_result(path: str, payload: Dict[str, Any]) -> None:
    """Write an experiment's computed result JSON.

    `payload` MUST be the output of real computation. This function
    refuses to write if the experiment did not stamp `computed_on` (the
    timestamp + host the script sets only after it finished real work) -
    a structural guard against committing fabricated results.
    """
    if "computed_on" not in payload:
        raise ValueError(
            "result payload missing 'computed_on'; refusing to write a "
            "result that was not produced by an actual run"
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
