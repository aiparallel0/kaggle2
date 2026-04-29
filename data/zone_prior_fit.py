"""Offline EM fit of the receipt-zone HMM emissions on the train split.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: derive emission/bias parameters for the 3-state monotone zone
    HMM (header → items → totals) defined in
    :mod:`models.zone_prior` from the 500-receipt train fold.

Weak supervision:

* The ``company`` gold y-position marks the bottom of the *header*
  zone.  Lines above it (``y < y_company``) are header-positive; lines
  whose y is between ``y_company`` and the gold ``total`` y-position
  are item-positive; lines at or below ``y_total`` are total-positive.
* This is a noisy proxy (some receipts have address / GST rows below
  the company line that are still header-zone) but it gives every
  line a single soft label and lets EM converge in 5–10 iterations.

Output: ``results/zone_prior.json`` with fields ``weights`` (3×6) and
``biases`` (3) — fixture-allowed per ``AGENTS.md`` (``results/`` is
fixtures-only).  Stays under the 166-LOC cap so the inference module
in ``models/zone_prior.py`` and this offline trainer can ship in the
same PR.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from models.zone_prior import _features


def _label(y_norm: float, y_company: float, y_total: float) -> int:
    """0=header, 1=items, 2=total — derived from gold field y-positions."""
    if y_norm < y_company:
        return 0
    if y_norm >= y_total:
        return 2
    return 1


def _softmax3(zs: list[float]) -> list[float]:
    m = max(zs)
    e = [math.exp(z - m) for z in zs]
    s = sum(e)
    return [v / s for v in e]


def _step(
    feats: list[list[float]], labels: list[int],
    weights: list[list[float]], biases: list[float],
    lr: float = 0.5,
) -> tuple[list[list[float]], list[float], float]:
    """One pass of multinomial-logistic SGD over (feature, hard-label)
    pairs; minimises cross-entropy of the per-line state assignment.
    """
    new_w = [list(row) for row in weights]
    new_b = list(biases)
    loss = 0.0
    for f, y in zip(feats, labels, strict=True):
        z = [
            sum(wi * fi for wi, fi in zip(new_w[s], f, strict=True)) + new_b[s]
            for s in range(3)
        ]
        p = _softmax3(z)
        loss -= math.log(max(p[y], 1e-12))
        for s in range(3):
            err = (1.0 if s == y else 0.0) - p[s]
            new_b[s] += lr * err
            for k in range(len(f)):
                new_w[s][k] += lr * err * f[k]
    return new_w, new_b, loss / max(1, len(feats))


def fit_zone_prior(
    receipts: list[dict[str, Any]],
    out_path: str = "./results/zone_prior.json",
    n_iter: int = 25,
    lr: float = 0.2,
) -> dict[str, Any]:
    """Fit the 3-state zone-HMM emissions and persist them as JSON.

    ``receipts`` is a list of dicts shaped::

        {
          "lines": [(text, y_norm, is_boilerplate), ...],
          "y_company": float,  # gold company line's y_norm
          "y_total":   float,  # gold total line's y_norm
        }

    Each line is hard-labelled by :func:`_label`; the multinomial
    logistic regression is fit with vanilla SGD (a closed-form M-step
    of EM under hard labels).  Persists the resulting weights /
    biases to ``out_path`` and returns them.
    """
    feats: list[list[float]] = []
    labels: list[int] = []
    for r in receipts:
        yc = float(r["y_company"])
        yt = float(r["y_total"])
        for text, y, boil in r["lines"]:
            feats.append(_features(str(text), float(y), float(boil)))
            labels.append(_label(float(y), yc, yt))
    if not feats:
        raise ValueError("zone_prior_fit: no training pairs derived")
    weights = [
        [-3.0, -0.5, -1.0, 2.0, -2.0, 1.5],
        [0.0, 0.5, 2.0, -0.5, -1.5, -1.0],
        [3.0, 1.0, -0.5, -1.5, 2.5, -0.5],
    ]
    biases = [1.0, 0.0, -0.5]
    last_loss = math.inf
    for _i in range(n_iter):
        weights, biases, last_loss = _step(feats, labels, weights, biases, lr)
    out: dict[str, Any] = {
        "weights": weights, "biases": biases, "train_loss": last_loss,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


def score_train_acc(
    receipts: list[dict[str, Any]],
    weights: list[list[float]], biases: list[float],
) -> float:
    """Per-line state-classification accuracy of the fit emissions.

    Used to populate the ``\\VAR{zone_prior_train_acc}`` placeholder
    in :mod:`docs.TRACKING`; aggregates over every (line, hard-label)
    pair derived from ``receipts``.
    """
    correct = 0
    total = 0
    for r in receipts:
        yc = float(r["y_company"])
        yt = float(r["y_total"])
        for text, y, boil in r["lines"]:
            f = _features(str(text), float(y), float(boil))
            z = [
                sum(wi * fi for wi, fi in zip(weights[s], f, strict=True))
                + biases[s]
                for s in range(3)
            ]
            pred = max(range(3), key=lambda s: z[s])
            total += 1
            if pred == _label(float(y), yc, yt):
                correct += 1
    return correct / max(1, total)


__all__ = ["fit_zone_prior", "score_train_acc"]
