"""
AP / mAP computation endpoint.
Accepts final kept boxes + image id (to look up ground-truth labels)
and returns per-class AP, PR curves, mAP, interim threshold table,
and per-GT-box match information (IoU vs best detection, matched/unmatched).
"""
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.ap import compute_ap
from app.core.iou import compute_iou
from app.core.map import compute_map

ROOT        = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = ROOT / "app" / "data" / "samples"
UPLOADS_DIR = ROOT / "app" / "data" / "uploads"

router = APIRouter(tags=["metrics"])


class KeptBox(BaseModel):
    id: int
    bbox: list[float]
    class_name: str
    class_id: int
    confidence: float


class APRequest(BaseModel):
    image_id:             str
    kept_boxes:           list[KeptBox]
    interpolation_points: int   = Field(11, ge=1, le=101)
    iou_mode:             str   = Field("single", pattern="^(single|range)$")
    iou_value:            float = Field(0.50, ge=0.05, le=0.95)


def _load_ground_truth(image_id: str) -> dict[str, Any]:
    for directory in [SAMPLES_DIR, UPLOADS_DIR]:
        path = directory / f"{image_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No ground-truth labels found for image '{image_id}'.")


def _compute_matching(
    gt_boxes: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Match GT boxes to predictions.

    Returns:
        gt_results           — one entry per GT box, matched-first then by IoU desc.
        false_positives      — predictions not matched to any GT box (FP).
        predictions_matching — all predictions with is_tp flag and iou, conf-desc order.
        false_negatives      — GT boxes not matched by any prediction (FN).
    """
    matched_pred_ids: set[int] = set()

    gt_results: list[dict[str, Any]] = []
    for gt in gt_boxes:
        best_iou  = 0.0
        best_pred = None
        for pred in predictions:
            if pred["class_name"] == gt["class_name"]:
                iou = compute_iou(gt["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou  = iou
                    best_pred = pred

        matched = best_iou >= iou_threshold
        if matched and best_pred:
            matched_pred_ids.add(best_pred["id"])

        gt_results.append({
            "gt_id":      gt["id"],
            "class_id":   gt["class_id"],
            "class_name": gt["class_name"],
            "bbox":       gt["bbox"],
            "best_iou":   round(best_iou, 4),
            "matched":    matched,
            "pred_id":    best_pred["id"] if matched and best_pred else None,
        })

    gt_results.sort(key=lambda r: (0 if r["matched"] else 1, -r["best_iou"]))

    # Build lookup: pred_id → GT row (for TP predictions)
    pred_to_gt = {r["pred_id"]: r for r in gt_results if r["pred_id"] is not None}

    # Pre-compute best IoU between every prediction and any same-class GT box.
    # This lets us show FP predictions that overlap a GT box which was already
    # claimed by a higher-confidence detection ("GT taken").
    pred_best: dict[int, tuple[float, int | None]] = {}  # pred_id → (best_iou, gt_id)
    for p in predictions:
        best_iou, best_gt_id = 0.0, None
        for gt in gt_boxes:
            if gt["class_name"] == p["class_name"]:
                iou = compute_iou(p["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_gt_id = iou, gt["id"]
        pred_best[p["id"]] = (round(best_iou, 4), best_gt_id)

    # All predictions with TP/FP label, sorted by confidence descending
    predictions_matching: list[dict[str, Any]] = []
    for p in sorted(predictions, key=lambda x: -x["confidence"]):
        is_tp = p["id"] in pred_to_gt
        best_iou, best_gt_id = pred_best[p["id"]]
        predictions_matching.append({
            "pred_id":    p["id"],
            "class_name": p["class_name"],
            "confidence": round(p["confidence"], 4),
            "is_tp":      is_tp,
            "iou":        pred_to_gt[p["id"]]["best_iou"] if is_tp else best_iou,
            "gt_taken":   not is_tp and best_iou >= iou_threshold,
        })

    # FP predictions — not matched to any GT
    false_positives: list[dict[str, Any]] = [
        {
            "pred_id":    p["id"],
            "class_name": p["class_name"],
            "confidence": round(p["confidence"], 4),
            "bbox":       p["bbox"],
        }
        for p in predictions
        if p["id"] not in matched_pred_ids
    ]
    false_positives.sort(key=lambda p: -p["confidence"])

    # FN ground truths — not matched by any prediction
    false_negatives: list[dict[str, Any]] = [
        {
            "gt_id":      r["gt_id"],
            "class_name": r["class_name"],
            "best_iou":   r["best_iou"],
        }
        for r in gt_results if not r["matched"]
    ]

    return gt_results, false_positives, predictions_matching, false_negatives


@router.post("/metrics/ap")
async def compute_metrics(req: APRequest) -> dict[str, Any]:
    """Compute per-class AP, mAP, PR curves, interim table, and GT match info."""
    try:
        gt = _load_ground_truth(req.image_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    predictions = [b.model_dump() for b in req.kept_boxes]
    cat_map = {c["id"]: c["name"] for c in gt.get("categories", [])}

    # Full GT box list (with IDs for matching display)
    gt_boxes_full: list[dict[str, Any]] = [
        {
            "id":         ann["id"],
            "bbox":       ann["bbox"],
            "class_id":   ann["category_id"],
            "class_name": cat_map.get(ann["category_id"], str(ann["category_id"])),
        }
        for ann in gt.get("annotations", [])
    ]

    # Stripped version for AP computation (no id field needed)
    ground_truths = [
        {"bbox": b["bbox"], "class_id": b["class_id"], "class_name": b["class_name"]}
        for b in gt_boxes_full
    ]

    iou_thresholds = (
        [round(0.5 + i * 0.05, 2) for i in range(10)]
        if req.iou_mode == "range"
        else [req.iou_value]
    )

    try:
        result = compute_map(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_thresholds=iou_thresholds,
            n_points=req.interpolation_points,
        )
    except Exception as exc:
        raise HTTPException(500, f"Metrics computation failed: {exc}") from exc

    gt_matching, false_positives, predictions_matching, false_negatives = _compute_matching(
        gt_boxes_full, predictions, iou_thresholds[0]
    )
    result["gt_matching"]          = gt_matching
    result["false_positives"]      = false_positives
    result["predictions_matching"] = predictions_matching
    result["false_negatives"]      = false_negatives
    return result
