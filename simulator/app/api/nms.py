"""
NMS / Soft-NMS step-by-step endpoint.
Accepts raw detection boxes + parameters; returns the full step trace.
"""
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.nms import run_nms
from app.core.soft_nms import run_soft_nms

router = APIRouter(tags=["nms"])


class RawBox(BaseModel):
    id: int
    bbox: list[float]          # [x, y, w, h] in pixels
    class_name: str
    class_id: int
    confidence: float


class NMSRequest(BaseModel):
    boxes:          list[RawBox]
    algorithm:      str = Field("nms", pattern="^(nms|soft_nms)$")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold:  float = Field(0.50, ge=0.0, le=1.0)
    method:         str   = Field("linear", pattern="^(linear|gaussian)$")
    sigma:          float = Field(0.50, ge=0.0, le=1.0)
    min_score:      float = Field(0.001, ge=0.0, le=1.0)


@router.post("/nms/run")
async def run_nms_endpoint(req: NMSRequest) -> dict[str, Any]:
    """
    Run the requested NMS variant and return all steps plus a warning string
    if a degenerate condition (IoU=0, IoU=1, no surviving boxes) was detected.
    """
    boxes_dicts = [b.model_dump() for b in req.boxes]

    warning: str | None = None

    # Degenerate threshold warnings
    if req.iou_threshold == 0.0:
        warning = "IoU threshold = 0: every overlapping box will be suppressed."
    elif req.iou_threshold == 1.0:
        warning = "IoU threshold = 1: no box will be suppressed by IoU alone."

    # Pre-filter by confidence
    filtered = [b for b in boxes_dicts if b["confidence"] >= req.conf_threshold]
    if not filtered:
        no_box_msg = "No boxes survive the confidence pre-filter."
        warning = f"{warning} {no_box_msg}" if warning else no_box_msg
        return {"steps": [], "final_kept_ids": [], "warning": warning}

    try:
        if req.algorithm == "nms":
            steps = run_nms(filtered, req.iou_threshold)
        else:
            steps = run_soft_nms(filtered, req.iou_threshold, req.method, req.sigma, req.min_score)
    except Exception as exc:
        raise HTTPException(500, f"NMS failed: {exc}") from exc

    final_kept = steps[-1]["kept_box_ids_so_far"] if steps else []
    return {"steps": steps, "final_kept_ids": final_kept, "warning": warning}
