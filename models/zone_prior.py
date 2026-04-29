"""3-state monotone receipt-zone HMM (header → items → totals).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: produce a per-line ``(p_header, p_items, p_total)`` posterior so
    FOCUS-C (company) and FOCUS-T (total) share a single relational
    prior over the receipt's vertical structure.  The HMM is
    fixed-topology (transitions hard-coded for monotonicity), uses ~30
    parameters, and runs on CPU at inference.  EM fitting against the
    500-receipt train split lives in :mod:`data.zone_prior_fit`.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from core.types import ZONE_HEADER, ZONE_ITEMS, ZONE_TOTAL, ZoneConfig, ZonePosterior

# Feature-derivation regexes — duplicate the spirit of the
# ``total_post`` / ``rule_fields`` patterns to keep this module
# importable without dragging in torch.
_MONEY_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*\.\d{2}|-?\d+\.\d{2}")
_ITEM_QTY_RE = re.compile(r"\b\d+\s*(?:x|@|pcs|pc|qty)\b", re.IGNORECASE)
_COMPANY_ANCHOR_RE = re.compile(
    r"\b(SDN\.?\s*BHD|BERHAD|ENTERPRISE|RESTAURANT|TRADING|CORPORATION|"
    r"HOLDINGS|MARKETING|GROUP|CAFE|RESOURCES|MART|S/B)\b",
    re.IGNORECASE,
)
_TOTALS_KW_RE = re.compile(
    r"\b(?:total|subtotal|sub[\s\-]?total|cash|tunai|change|kembalian|"
    r"jumlah|amount|tendered|payable|due|gst|sst|tax|cukai|service|"
    r"discount|round)\b",
    re.IGNORECASE,
)
# Default emission weights per state (header / items / total) over the
# 6-d feature vector ``[y_norm, money, item_qty, anchor, totals_kw,
# boilerplate]`` plus per-state bias.  Hand-tuned so SROIE-Malaysia
# single-column receipts decode correctly without EM fit; replaceable
# via :class:`ZoneConfig.params_path`.
_W: list[list[float]] = [
    [-3.0, -0.5, -1.0, 2.0, -2.0, 1.5],   # header
    [0.0, 0.5, 2.0, -0.5, -1.5, -1.0],    # items
    [3.0, 1.0, -0.5, -1.5, 2.5, -0.5],    # total
]
_B: list[float] = [1.0, 0.0, -0.5]


def _features(text: str, y_norm: float, boil: float) -> list[float]:
    """Build the 6-dim feature vector for one OCR line."""
    s = text or ""
    return [
        float(y_norm),
        1.0 if _MONEY_RE.search(s) else 0.0,
        1.0 if _ITEM_QTY_RE.search(s) else 0.0,
        1.0 if _COMPANY_ANCHOR_RE.search(s) else 0.0,
        1.0 if _TOTALS_KW_RE.search(s) else 0.0,
        float(boil),
    ]


def _emit(
    feat: list[float], weights: list[list[float]], biases: list[float],
) -> list[float]:
    """Per-state un-normalised log-emission for one line."""
    return [
        sum(wi * fi for wi, fi in zip(weights[s], feat, strict=True)) + biases[s]
        for s in range(3)
    ]


def _logsumexp(xs: list[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(x - m) for x in finite))


def _forward_backward(emits: list[list[float]]) -> ZonePosterior:
    """Decode the monotone H→I→T HMM via forward–backward.

    Transitions: from state ``s`` the next state is ``s`` or ``s+1``,
    so back-transitions are impossible.  Receipt must start in H and
    end in T.  This enforces the relational invariant
    ``argmax_y(company) < argmax_y(total)`` at the posterior level.
    """
    n = len(emits)
    if n == 0:
        return []
    a: list[list[float]] = [[-math.inf] * 3 for _ in range(n)]
    bt: list[list[float]] = [[-math.inf] * 3 for _ in range(n)]
    a[0] = [emits[0][0], -math.inf, -math.inf]
    for t in range(1, n):
        for s in range(3):
            opts = [a[t - 1][s]] + ([a[t - 1][s - 1]] if s > 0 else [])
            a[t][s] = emits[t][s] + _logsumexp(opts)
    bt[n - 1] = [-math.inf, -math.inf, 0.0]
    if n == 1:
        bt[0] = [0.0, 0.0, 0.0]
    for t in range(n - 2, -1, -1):
        for s in range(3):
            opts = [emits[t + 1][s] + bt[t + 1][s]]
            if s < 2:
                opts.append(emits[t + 1][s + 1] + bt[t + 1][s + 1])
            bt[t][s] = _logsumexp(opts)
    out: ZonePosterior = []
    for t in range(n):
        joint = [a[t][s] + bt[t][s] for s in range(3)]
        z = _logsumexp(joint)
        if not math.isfinite(z):
            out.append((1.0 / 3, 1.0 / 3, 1.0 / 3))
            continue
        p = [math.exp(joint[s] - z) for s in range(3)]
        out.append((p[ZONE_HEADER], p[ZONE_ITEMS], p[ZONE_TOTAL]))
    return out


def _load_params(path: str) -> tuple[list[list[float]], list[float]]:
    if not path:
        return _W, _B
    p = Path(path)
    if not p.is_file():
        return _W, _B
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return _W, _B
    w = raw.get("weights")
    b = raw.get("biases")
    if (
        isinstance(w, list) and len(w) == 3
        and all(isinstance(r, list) and len(r) == 6 for r in w)
        and isinstance(b, list) and len(b) == 3
    ):
        return [[float(x) for x in r] for r in w], [float(x) for x in b]
    return _W, _B


def decode_zone_posterior(
    lines: list[tuple[str, float, float]], cfg: ZoneConfig,
) -> ZonePosterior:
    """Forward–backward decode of the 3-state receipt-zone HMM.

    ``lines`` is a list of ``(text, y_norm, is_boilerplate)`` triples in
    reading order; ``y_norm`` is ``priors_v4[i, V4_Y_NORM_IDX]`` and
    ``is_boilerplate`` is ``priors_v4[i, V4_IS_COMPANY_BOILERPLATE_IDX]``.
    Returns a ``ZonePosterior`` aligned with ``lines``; empty input
    returns an empty posterior.  When ``cfg.enabled`` is False the
    posterior is uniform so callers see no zone signal.
    """
    if not lines:
        return []
    if not cfg.enabled:
        return [(1.0 / 3, 1.0 / 3, 1.0 / 3) for _ in lines]
    weights, biases = _load_params(cfg.params_path)
    emits = [_emit(_features(t, y, b), weights, biases) for t, y, b in lines]
    return _forward_backward(emits)


__all__ = ["decode_zone_posterior"]

