"""
Average Precision (AP) computation with N-point interpolation.

AP is the area under the Precision-Recall curve.  We use the standard
N-point interpolation method (default N=11, matching the PASCAL VOC metric):

  AP = (1/N) * Σ max(precision at recall >= r_k)
        for r_k in {0, 1/(N-1), 2/(N-1), ..., 1}

A detection is a True Positive if:
  - Its predicted class matches a ground-truth box's class, AND
  - IoU(predicted_bbox, gt_bbox) >= iou_threshold, AND
  - The ground-truth box has not already been matched by a higher-confidence detection.

All other detections are False Positives.
Ground-truth boxes not matched by any detection are False Negatives.
"""
from __future__ import annotations

from typing import Any

from app.core.iou import compute_iou


def match_detections(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[bool], int]:
    """
    Match predictions to ground-truth boxes for a single class.

    Parameters
    ----------
    predictions   : sorted descending by confidence; each has bbox, confidence.
    ground_truths : list of GT boxes for this class; each has bbox.
    iou_threshold : minimum IoU for a positive match.

    Returns
    -------
    tp_flags : list[bool] — True if the i-th prediction is a TP.
    n_gt     : number of ground-truth boxes (= total positives).
    """
    matched_gt: set[int] = set()
    tp_flags: list[bool] = []

    for pred in predictions:
        best_iou  = 0.0
        best_gt_i = -1
        for gt_i, gt in enumerate(ground_truths):
            if gt_i in matched_gt:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou  = iou
                best_gt_i = gt_i

        if best_iou >= iou_threshold and best_gt_i >= 0:
            tp_flags.append(True)
            matched_gt.add(best_gt_i)
        else:
            tp_flags.append(False)

    return tp_flags, len(ground_truths)


def precision_recall_curve(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[float], list[float]]:
    """
    Build the raw precision-recall curve for one class.

    Returns two parallel lists (recalls, precisions) at each confidence step,
    sorted in ascending recall order.
    """
    # Sort predictions by confidence descending
    preds_sorted = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    tp_flags, n_gt = match_detections(preds_sorted, ground_truths, iou_threshold)

    if n_gt == 0:
        return [], []

    recalls:    list[float] = []
    precisions: list[float] = []
    cum_tp = 0
    cum_fp = 0

    for is_tp in tp_flags:
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        recalls.append(cum_tp / n_gt)
        precisions.append(cum_tp / (cum_tp + cum_fp))

    return recalls, precisions


def n_point_interpolation(
    recalls: list[float],
    precisions: list[float],
    n_points: int = 11,
) -> float:
    """
    Compute AP using N-point interpolation over the PR curve.

    At each of the N uniformly spaced recall levels, we take the maximum
    precision for recall >= that level (the envelope).
    """
    if not recalls:
        return 0.0

    recall_levels = [i / (n_points - 1) for i in range(n_points)]
    ap = 0.0
    for r_level in recall_levels:
        # Max precision at recalls >= r_level
        max_prec = max(
            (p for r, p in zip(recalls, precisions) if r >= r_level),
            default=0.0,
        )
        ap += max_prec
    return ap / n_points


def compute_ap(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    n_points: int = 11,
) -> dict[str, Any]:
    """
    Compute AP for a single class.

    Parameters
    ----------
    predictions   : detection boxes for this class (all with same class_id).
    ground_truths : GT boxes for this class.
    iou_threshold : IoU threshold for a TP match.
    n_points      : interpolation points (default 11 = PASCAL VOC).

    Returns
    -------
    Dict with keys: ap, recalls, precisions (raw curve before interpolation).
    """
    recalls, precisions = precision_recall_curve(predictions, ground_truths, iou_threshold)
    ap = n_point_interpolation(recalls, precisions, n_points)
    return {"ap": round(ap, 6), "recalls": recalls, "precisions": precisions}
