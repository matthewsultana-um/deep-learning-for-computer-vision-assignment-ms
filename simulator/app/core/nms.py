"""
Standard Greedy NMS with a per-step trace.

Algorithm
---------
1.  Sort remaining candidates by confidence (descending).
2.  Pick the top-confidence box as the pivot; add it to the keep list.
3.  Compute IoU between the pivot and every other remaining candidate.
4.  Suppress any candidate whose IoU >= iou_threshold.
5.  Repeat until no candidates remain.

Each loop iteration produces one Step record.  Step 0 is the initial state
(all candidates visible, no pivot selected yet).
"""
from typing import Any

from app.core.iou import compute_suppression_score


def run_nms(
    boxes: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """
    Run greedy NMS on *boxes* and return a list of Step dicts.

    Parameters
    ----------
    boxes : list of dicts with keys id, bbox, class_name, class_id, confidence.
            Must already be pre-filtered by confidence threshold.
    iou_threshold : float in [0, 1].

    Returns
    -------
    List of Step dicts matching the NMS step trace contract in CLAUDE.md.
    """
    if not boxes:
        return []

    # Work on a copy; sort descending by confidence
    candidates: list[dict[str, Any]] = sorted(
        [dict(b) for b in boxes],
        key=lambda b: b["confidence"],
        reverse=True,
    )

    kept_ids:       list[int] = []
    suppressed_ids: list[int] = []
    steps:          list[dict[str, Any]] = []

    # Step 0 — initial state, no pivot selected yet
    steps.append(_make_step(
        step_index=0,
        pivot_box_id=None,
        compared_box_ids=[],
        suppressed_box_ids=[],
        kept_box_ids_so_far=[],
        comment=f"Sorted {len(candidates)} candidate(s) by confidence. Ready to run NMS.",
        referenced_box_ids=[b["id"] for b in candidates],
        iou_threshold=iou_threshold,
    ))

    step_idx = 1
    remaining = list(candidates)  # ordered list of still-active boxes

    while remaining:
        # Pick the highest-confidence remaining box as pivot
        pivot = remaining[0]
        pivot_id = pivot["id"]
        kept_ids.append(pivot_id)

        # Evaluate pivot against all other remaining candidates
        others = remaining[1:]
        compared_ids  = [b["id"] for b in others]
        new_suppressed: list[int] = []
        still_alive: list[dict[str, Any]] = []
        iou_scores: dict[str, float] = {}  # box_id → IoU with pivot

        for other in others:
            if other["class_id"] == pivot["class_id"]:
                score = compute_suppression_score(pivot["bbox"], other["bbox"])
                iou_scores[str(other["id"])] = round(score, 4)
                if score >= iou_threshold:
                    new_suppressed.append(other["id"])
                    suppressed_ids.append(other["id"])
                else:
                    still_alive.append(other)
            else:
                still_alive.append(other)

        referenced = [pivot_id] + compared_ids

        steps.append(_make_step(
            step_index=step_idx,
            pivot_box_id=pivot_id,
            compared_box_ids=compared_ids,
            suppressed_box_ids=new_suppressed,
            kept_box_ids_so_far=list(kept_ids),
            iou_scores=iou_scores,
            iou_threshold=iou_threshold,
            comment=(
                f"Box #{pivot_id} ({pivot['class_name']}, conf {pivot['confidence']:.2f}) "
                f"selected as pivot. Checked {len(others)} candidate(s): "
                f"{len(new_suppressed)} suppressed, {len(still_alive)} remain."
            ),
            referenced_box_ids=referenced,
        ))

        remaining = still_alive
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
    }
