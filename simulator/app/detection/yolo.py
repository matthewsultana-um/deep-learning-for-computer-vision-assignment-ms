"""
YOLO inference wrapper.

Loads YOLOv8n once at startup and caches results per image (keyed by SHA-256
of the raw image bytes).  Slider adjustments only re-run NMS, not the detector.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT        = Path(__file__).resolve().parent.parent.parent
WEIGHTS     = ROOT / "app" / "weights" / "yolov8n.pt"
SAMPLES_DIR = ROOT / "app" / "data" / "samples"
UPLOADS_DIR = ROOT / "app" / "data" / "uploads"

# Module-level cache: sha256 → list of box dicts
_inference_cache: dict[str, list[dict[str, Any]]] = {}


@lru_cache(maxsize=1)
def _get_model():
    """Load and return the YOLO model (called once, result cached by lru_cache)."""
    from ultralytics import YOLO  # import here so the module loads without GPU

    if not WEIGHTS.exists():
        raise RuntimeError(
            f"Model weights not found at {WEIGHTS}. "
            "Run scripts/download_weights.py first."
        )
    model = YOLO(str(WEIGHTS))
    return model


def _find_image(image_id: str) -> Path:
    """Locate an image file by its stem (image_id)."""
    for directory in [SAMPLES_DIR, UPLOADS_DIR]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = directory / f"{image_id}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Image '{image_id}' not found in samples or uploads.")


def run_inference(image_id: str) -> list[dict[str, Any]]:
    """
    Run YOLOv8n on the image and return raw detection boxes.

    Returns a list of dicts with keys:
        id, bbox ([x, y, w, h] in pixels), class_name, class_id, confidence

    Results are cached by SHA-256 of the image bytes so repeated calls with
    the same image are free.  Confidence pre-filtering is done later in NMS.
    """
    img_path = _find_image(image_id)
    img_bytes = img_path.read_bytes()
    cache_key = hashlib.sha256(img_bytes).hexdigest()

    if cache_key in _inference_cache:
        return _inference_cache[cache_key]

    model = _get_model()
    # conf=0.01 so we keep almost everything; NMS pre-filter is done client-side
    results = model.predict(str(img_path), conf=0.01, iou=1.0, device="cpu", verbose=False)

    boxes: list[dict[str, Any]] = []
    if results and len(results) > 0:
        r = results[0]
        names = r.names  # {class_id: class_name}
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            cls_id = int(box.cls[0].item())
            boxes.append({
                "id":         i,
                "bbox":       [round(v, 2) for v in [x, y, w, h]],
                "class_name": names[cls_id],
                "class_id":   cls_id,
                "confidence": round(float(box.conf[0].item()), 4),
            })

    _inference_cache[cache_key] = boxes
    return boxes
