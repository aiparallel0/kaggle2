"""YOLO detection-time diagnostics — mAP, per-class AP, IoU histogram.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: reduce a list of ``YoloPred`` objects (boxes + confidences) +
    matched gold boxes to the scalars reviewers want in the YOLO
    detection table — mAP@0.5, mAP@[0.5:0.95], per-class AP,
    precision/recall at several confidence thresholds, plus an IoU
    histogram the figure emitter turns into a distribution plot.
    Every function is 2-in/1-out.  Missing inputs return neutral zeros
    rather than raising — telemetry code should never abort a run.
"""
from __future__ import annotations

from collections.abc import Sequence

from core.schemas import SCHEMA_VERSIONS, YoloDiagnostics

# A box is (x1, y1, x2, y2, confidence, class_id).  A gold box is
# (x1, y1, x2, y2, class_id).  Using plain tuples keeps the module
# torch-free so it imports inside the CPU-only CI matrix.
PredBox = tuple[float, float, float, float, float, int]
GoldBox = tuple[float, float, float, float, int]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _ap_at(
    preds: Sequence[PredBox], golds: Sequence[GoldBox], iou_thresh: float,
) -> float:
    """11-point interpolated AP at a single IoU threshold (VOC style)."""
    if not preds or not golds:
        return 0.0
    preds_sorted = sorted(preds, key=lambda p: -p[4])
    matched: set[int] = set()
    tps: list[int] = []
    fps: list[int] = []
    for p in preds_sorted:
        best_iou = 0.0
        best_idx = -1
        for j, g in enumerate(golds):
            if j in matched or g[4] != p[5]:
                continue
            score = iou((p[0], p[1], p[2], p[3]), (g[0], g[1], g[2], g[3]))
            if score > best_iou:
                best_iou = score
                best_idx = j
        if best_iou >= iou_thresh and best_idx >= 0:
            tps.append(1)
            fps.append(0)
            matched.add(best_idx)
        else:
            tps.append(0)
            fps.append(1)
    cum_tp = 0
    cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    n_gold = len(golds)
    for t, f in zip(tps, fps, strict=True):
        cum_tp += t
        cum_fp += f
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / n_gold)
    # 11-point interpolation at recall levels 0.0, 0.1, ..., 1.0.
    ap = 0.0
    for r_target in [i / 10.0 for i in range(11)]:
        best_p = 0.0
        for p_val, r_val in zip(precisions, recalls, strict=True):
            if r_val >= r_target and p_val > best_p:
                best_p = p_val
        ap += best_p / 11.0
    return ap


def _precision_recall_at(
    preds: Sequence[PredBox], golds: Sequence[GoldBox],
    iou_thresh: float, conf: float,
) -> tuple[float, float]:
    """Single-point precision + recall at a given confidence / IoU cutoff."""
    filtered = [p for p in preds if p[4] >= conf]
    if not filtered:
        return (0.0, 0.0)
    matched: set[int] = set()
    tp = 0
    for p in filtered:
        for j, g in enumerate(golds):
            if j in matched or g[4] != p[5]:
                continue
            if iou((p[0], p[1], p[2], p[3]), (g[0], g[1], g[2], g[3])) >= iou_thresh:
                tp += 1
                matched.add(j)
                break
    prec = tp / len(filtered) if filtered else 0.0
    rec = tp / len(golds) if golds else 0.0
    return (prec, rec)


def compute_yolo_diagnostics(
    preds_per_img: Sequence[Sequence[PredBox]],
    golds_per_img: Sequence[Sequence[GoldBox]],
) -> YoloDiagnostics:
    """Aggregate mAP / per-class AP / IoU hist across a test set."""
    if len(preds_per_img) != len(golds_per_img):
        return YoloDiagnostics(schema_version=SCHEMA_VERSIONS["YoloDiagnostics"])
    all_preds: list[PredBox] = [p for img in preds_per_img for p in img]
    all_golds: list[GoldBox] = [g for img in golds_per_img for g in img]
    classes = sorted({int(g[4]) for g in all_golds})
    per_class: dict[str, float] = {}
    for cls in classes:
        cls_preds = [p for p in all_preds if p[5] == cls]
        cls_golds = [g for g in all_golds if g[4] == cls]
        per_class[str(cls)] = _ap_at(cls_preds, cls_golds, 0.5)
    map50 = sum(per_class.values()) / len(per_class) if per_class else 0.0
    map5095_acc = 0.0
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    for t in thresholds:
        per_t = [_ap_at(
            [p for p in all_preds if p[5] == c],
            [g for g in all_golds if g[4] == c],
            t,
        ) for c in classes] or [0.0]
        map5095_acc += sum(per_t) / len(per_t)
    map5095 = map5095_acc / len(thresholds)
    # IoU of best match per gold box (zero if no overlap).
    ious: list[float] = []
    for img_preds, img_golds in zip(preds_per_img, golds_per_img, strict=True):
        for g in img_golds:
            best = 0.0
            for p in img_preds:
                if p[5] == g[4]:
                    best = max(best, iou(
                        (p[0], p[1], p[2], p[3]), (g[0], g[1], g[2], g[3]),
                    ))
            ious.append(best)
    ious_sorted = sorted(ious)
    iou_median = ious_sorted[len(ious_sorted) // 2] if ious_sorted else 0.0
    box_counts = [len(img) for img in preds_per_img]
    p25, r25 = _precision_recall_at(all_preds, all_golds, 0.5, 0.25)
    return YoloDiagnostics(
        schema_version=SCHEMA_VERSIONS["YoloDiagnostics"],
        map50=map50, map5095=map5095, per_class_ap=per_class,
        iou_median=iou_median,
        iou_mean=(sum(ious) / len(ious)) if ious else 0.0,
        boxes_per_receipt_mean=(sum(box_counts) / len(box_counts)) if box_counts else 0.0,
        boxes_per_receipt_median=(
            sorted(box_counts)[len(box_counts) // 2] if box_counts else 0
        ),
        p_at_0_25=p25, r_at_0_25=r25,
    )
