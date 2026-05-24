"""
YOLO inference endpoint.

Using a plain `def` (not `async def`) so FastAPI runs it automatically in
Starlette's default thread-pool executor.  This keeps the async event loop
free without needing a manual ThreadPoolExecutor, and avoids the
asyncio/PyTorch threading conflict that caused segfaults with run_in_executor.
"""
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.detection.yolo import run_inference

router = APIRouter(tags=["detection"])


class DetectRequest(BaseModel):
    image_id: str


@router.post("/detect")
def detect(req: DetectRequest) -> dict[str, Any]:
    """Run YOLOv8n on the image identified by image_id and return raw boxes."""
    try:
        boxes = run_inference(req.image_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Detection failed: {exc}") from exc
    return {"image_id": req.image_id, "boxes": boxes}
