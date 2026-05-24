"""
Soft-NMS with a per-step trace.

Soft-NMS (Bodla et al., 2017) replaces hard suppression with a confidence
decay.  Instead of zeroing the score of overlapping boxes, it reduces them
proportional to their IoU with the pivot:

  Linear  :  conf = conf * (1 - IoU)    when IoU > iou_threshold
  Gaussian:  conf = conf * exp(-IoU² / sigma)   always

After all pivots are selected, boxes with confidence below a minimum score
threshold (hard-coded here to 0.001) are considered suppressed.

Each loop iteration produces one Step record (same contract as NMS).
"""
import math
from typing import Any

from app.core.iou import compute_iou, compute_suppression_score

_MIN_SCORE = 0.001  # boxes below this after decay are treated as suppressed


def run_soft_nms(
    boxes: list[dict[str, Any]],
    iou_threshold: float,
    method: str = "linear",
    sigma: float = 0.5,
    min_score: float = 0.001,
) -> list[dict[str, Any]]:
    """
    Run Soft-NMS on *boxes* and return a list of Step dicts.

    Parameters
    ----------
    boxes         : pre-confidence-filtered detection boxes.
    iou_threshold : hard IoU threshold (Linear) or decay inflection (Gaussian).
    method        : 'linear' | 'gaussian'.
    sigma         : Gaussian decay width (ignored for linear).
    min_score     : boxes whose confidence falls below this after decay are
                    treated as suppressed (replaces the old hard-coded 0.001).

    Returns
    -------
    List of Step dicts matching the NMS step trace contract.
    """
    if not boxes:
        return []

    method = method.lower()

    # Working copies with a mutable confidence field
    candidates: list[dict[str, Any]] = [dict(b) for b in boxes]

    kept_ids:       list[int] = []
    suppressed_ids: list[int] = []
    steps:          list[dict[str, Any]] = []

    steps.append(_make_step(
        step_index=0,
        pivot_box_id=None,
        compared_box_ids=[],
        suppressed_box_ids=[],
        kept_box_ids_so_far=[],
        comment=(
            f"Sorted {len(candidates)} candidate(s) by confidence. "
            f"Soft-NMS ({method}) will decay scores instead of hard suppression. "
            f"Boxes decayed below {min_score:.3f} will be suppressed."
        ),
        referenced_box_ids=[b["id"] for b in candidates],
        iou_threshold=iou_threshold,
    ))

    step_idx = 1
    remaining = list(candidates)

    while remaining:
        # Pick highest-confidence remaining box
        remaining.sort(key=lambda b: b["confidence"], reverse=True)
        pivot = remaining[0]
        pivot_id = pivot["id"]
        kept_ids.append(pivot_id)

        others = remaining[1:]
        compared_ids       = [b["id"] for b in others]
        new_suppressed:  list[int]           = []
        weight_updates:  list[dict[str, Any]] = []
        next_remaining: list[dict[str, Any]] = []

        for other in others:
            iou    = compute_suppression_score(pivot["bbox"], other["bbox"])
            old_cf = other["confidence"]

            if other["class_id"] != pivot["class_id"]:
                # Different class — skip decay entirely, no iou_score emitted
                next_remaining.append(other)
                continue

            if method == "gaussian":
                decay = math.exp(-(iou ** 2) / sigma) if iou > 0 else 1.0
                new_cf = old_cf * decay
            else:
                # Linear decay only when IoU > iou_threshold
                if iou > iou_threshold:
                    new_cf = old_cf * (1.0 - iou)
                else:
                    new_cf = old_cf
                decay = new_cf / old_cf if old_cf else 1.0

            weight_updates.append({
                "box_id":    other["id"],
                "iou":       round(iou, 4),
                "old_conf":  round(old_cf, 4),
                "new_conf":  round(new_cf, 4),
                "decay":     round(decay, 4),
            })
            other["confidence"] = new_cf

            if new_cf < min_score:
                new_suppressed.append(other["id"])
                suppressed_ids.append(other["id"])
            else:
                next_remaining.append(other)

        # Build iou_scores from weight_updates for consistency with NMS contract
        iou_scores = {str(u["box_id"]): u["iou"] for u in weight_updates}

        steps.append(_make_step(
            step_index=step_idx,
            pivot_box_id=pivot_id,
            compared_box_ids=compared_ids,
            suppressed_box_ids=new_suppressed,
            kept_box_ids_so_far=list(kept_ids),
            comment=(
                f"Box #{pivot_id} ({pivot['class_name']}, conf {pivot['confidence']:.2f}) "
                f"selected as pivot. Decayed {len(weight_updates)} box(es): "
                f"{len(new_suppressed)} suppressed."
            ),
            referenced_box_ids=[pivot_id] + compared_ids,
            iou_scores=iou_scores,
            iou_threshold=iou_threshold,
            weight_updates=weight_updates,
        ))

        remaining = next_remaining
        step_idx += 1

    return steps


def _make_step(
    *,
    step_index: int,
    pivot_box_id: int | None,
    compared_box_ids: list[int],
    suppressed_box_ids: list[int],
    kept_box_ids_so_far: list[int],
    comment: str,
    referenced_box_ids: list[int],
    iou_scores: dict[str, float] | None = None,
    iou_threshold: float | None = None,
    weight_updates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "step_index":          step_index,
        "pivot_box_id":        pivot_box_id,
        "compared_box_ids":    compared_box_ids,
        "suppressed_box_ids":  suppressed_box_ids,
        "kept_box_ids_so_far": kept_box_ids_so_far,
        "comment":             comment,
        "referenced_box_ids":  referenced_box_ids,
        "iou_scores":          iou_scores or {},
        "iou_threshold":       iou_threshold,
        "weight_updates":      weight_updates,
    }
