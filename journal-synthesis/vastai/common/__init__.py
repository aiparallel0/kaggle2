"""Shared helpers for the vast.ai journal experiment package.

NOT a stub. These are the real metric implementations ported verbatim
from journal-synthesis/experiments/run_analysis.py (Wilson, phi/MCC,
two-sided permutation) plus bootstrap and Spearman, the deterministic
seeding helper, and the UNIFIED per-receipt record schema/writer that
every experiment must use so receipt_ids align by construction.

Nothing here fabricates a result. The record writer only persists
records that an experiment actually computed on real model output.
"""
from .schema import (
    UnifiedRecord,
    write_records,
    write_result,
    REQUIRED_FIELDS,
)
from .seeding import seed_everything
from .metrics import (
    median,
    phi_mcc,
    perm_p,
    wilson,
    bootstrap_ci,
    spearman,
    variance,
    variance_ratio_log2,
)
from .pipeline import (
    subset_sum_verdict,
    i3_accepts,
    reachable_targets,
    to_cents,
    load_donut,
    decode_fields,
    beam_margin_batch,
)
from .records import (
    decode_or_load,
    cache_path,
    checkpoint_sha,
    split_corpus_arg,
)
from .totals import (
    parse_money,
    parse_items,
    flatten_donut,
    gold_fields,
    gold_total_cents,
    gold_items_cents,
    gold_tax_cents,
    pred_total_cents,
    pred_items_cents,
    pred_tax_cents,
    is_correct,
    subset_sum_exists,
    subset_sum_verdict_prior,
    EPS_CENTS,
    ADMISSIBLE_RATES_CORD,
)

__all__ = [
    "UnifiedRecord",
    "write_records",
    "write_result",
    "REQUIRED_FIELDS",
    "seed_everything",
    "median",
    "phi_mcc",
    "perm_p",
    "wilson",
    "bootstrap_ci",
    "spearman",
    "variance",
    "variance_ratio_log2",
    "subset_sum_verdict",
    "i3_accepts",
    "reachable_targets",
    "to_cents",
    "load_donut",
    "decode_fields",
    "beam_margin_batch",
    "decode_or_load",
    "cache_path",
    "checkpoint_sha",
    "split_corpus_arg",
    "parse_money",
    "parse_items",
    "flatten_donut",
    "gold_fields",
    "gold_total_cents",
    "gold_items_cents",
    "gold_tax_cents",
    "pred_total_cents",
    "pred_items_cents",
    "pred_tax_cents",
    "is_correct",
    "subset_sum_exists",
    "subset_sum_verdict_prior",
    "EPS_CENTS",
    "ADMISSIBLE_RATES_CORD",
]
