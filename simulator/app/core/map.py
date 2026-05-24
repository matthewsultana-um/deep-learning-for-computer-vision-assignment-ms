"""
mAP (mean Average Precision) computation.

Supports:
  - Single-IoU mode  : one IoU threshold, one AP per class, mAP = mean(AP).
  - COCO-range mode  : IoU in {0.50, 0.55, ..., 0.95}, AP averaged over
                       thresholds then over classes.

The public entry point is compute_map().
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.core.ap import compute_ap


def compute_map(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_thresholds: list[float],
    n_points: int = 11,
) -> dict[str, Any]:
    """
    Compute per-class AP and overall mAP.

    Parameters
    ----------
    predictions   : list of kept detection boxes; each has bbox, class_id,
                    class_name, confidence.
    ground_truths : list of GT boxes; each has bbox, class_id, class_name.
    iou_thresholds : one or more IoU thresholds.  When len > 1 the AP for each
                     class is averaged over thresholds (COCO style).
    n_points      : interpolation points for AP (default 11).

    Returns
    -------
    {
      per_class_ap : [{class_name, class_id, ap}],
      map          : float,
      pr_curves    : {class_name: {recalls, precisions}},   # at first IoU threshold
      interim_table: [{conf_threshold, recall, precision}], # for selected class
    }
    """
    # Group by class_name so YOLO 0-based IDs and COCO 1-based IDs match correctly.
    preds_by_class: dict[str, list[dict]] = defaultdict(list)
    gt_by_class:    dict[str, list[dict]] = defaultdict(list)

    for p in predictions:
        preds_by_class[p["class_name"]].append(p)
    for g in ground_truths:
        gt_by_class[g["class_name"]].append(g)

    all_class_names = sorted(set(preds_by_class) | set(gt_by_class))

    per_class_ap: list[dict[str, Any]] = []
    pr_curves:    dict[str, Any]       = {}

    for cls_name in all_class_names:
        cls_preds = preds_by_class.get(cls_name, [])
        cls_gt    = gt_by_class.get(cls_name, [])
        # Use the class_id from predictions if available, else from ground truth
        cls_id = (cls_preds[0]["class_id"] if cls_preds
                  else cls_gt[0]["class_id"] if cls_gt
                  else -1)

        # Average AP over all IoU thresholds
        ap_values:      list[float] = []
        raw_recalls:    list[float] = []
        raw_precisions: list[float] = []

        for iou_t in iou_thresholds:
            result = compute_ap(cls_preds, cls_gt, iou_threshold=iou_t, n_points=n_points)
            ap_values.append(result["ap"])
            if not raw_recalls:
                raw_recalls    = result["recalls"]
                raw_precisions = result["precisions"]

        ap_mean = sum(ap_values) / len(ap_values) if ap_values else 0.0

        # Build N-point interpolated curve so all classes share the same recall
        # axis — required for a well-defined macro-average curve in the frontend.
        recall_levels = [i / (n_points - 1) for i in range(n_points)]
        interp_prec   = [
            max(
                (p for r, p in zip(raw_recalls, raw_precisions) if r >= rl),
                default=0.0,
            )
            for rl in recall_levels
        ]

        per_class_ap.append({
            "class_id":   cls_id,
            "class_name": cls_name,
            "ap":         round(ap_mean, 6),
        })
        pr_curves[cls_name] = {
            "recalls":        recall_levels,
            "precisions":     [round(p, 6) for p in interp_prec],
            "raw_recalls":    [round(r, 6) for r in raw_recalls],
            "raw_precisions": [round(p, 6) for p in raw_precisions],
        }

    # mAP: average only over classes that have at least one GT instance.
    # Predicted classes with no GT (pure false positives) are shown in the
    # per-class breakdown but not counted in the mean — this matches the
    # standard PASCAL VOC / COCO evaluation convention.
    gt_class_names = set(gt_by_class.keys())
    map_values = [c["ap"] for c in per_class_ap if c["class_name"] in gt_class_names]
    overall_map = sum(map_values) / len(map_values) if map_values else 0.0

    # Interim table: raw PR points for the first class (or empty)
    interim_table: list[dict[str, Any]] = []
    if per_class_ap:
        first_cls = per_class_ap[0]["class_name"]
        curve = pr_curves.get(first_cls, {})
        recalls    = curve.get("recalls", [])
        precisions = curve.get("precisions", [])
        n = len(recalls)
        for i, (r, p) in enumerate(zip(recalls, precisions)):
            # Approximate conf threshold as descending from 1 to 0
            cf = 1.0 - i / max(n - 1, 1) if n > 1 else 1.0
            interim_table.append({
                "conf_threshold": round(cf, 4),
                "recall":         round(r, 4),
                "precision":      round(p, 4),
            })

    return {
        "per_class_ap":  per_class_ap,
        "map":           round(overall_map, 6),
        "pr_curves":     pr_curves,
        "interim_table": interim_table,
    }
