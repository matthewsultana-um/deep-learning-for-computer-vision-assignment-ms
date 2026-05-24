"""
IoU (Intersection over Union) helpers.

All functions operate on COCO-format bounding boxes: [x_min, y_min, w, h].
Inputs are never mutated; new values are returned.
"""
from typing import Any


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.

    Returns a value in [0.0, 1.0].  Returns 0.0 if either box has zero area.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to [x1, y1, x2, y2]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    area_a     = aw * ah
    area_b     = bw * bh
    union_area  = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def compute_suppression_score(box_a: list[float], box_b: list[float]) -> float:
    """
    Return max(IoU, containment_ratio) where containment_ratio is
    intersection / min(area_a, area_b).

    Standard IoU fails to suppress a small box that is fully inside a much
    larger box because the union is dominated by the large box.  Taking the
    max with the containment ratio catches that case: a fully-engulfed box
    always scores 1.0, regardless of the size difference.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    area_a     = aw * ah
    area_b     = bw * bh
    union_area  = area_a + area_b - inter_area

    iou         = inter_area / union_area if union_area > 0 else 0.0
    containment = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return max(iou, containment)


def batch_iou(pivot: list[float], others: list[list[float]]) -> list[float]:
    """
    Compute IoU between *pivot* and each box in *others*.

    Returns a list of IoU scores in the same order as *others*.
    """
    return [compute_iou(pivot, other) for other in others]
