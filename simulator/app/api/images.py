"""
Endpoints for listing sample images, curated examples, and handling uploads.
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

ROOT        = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = ROOT / "app" / "data" / "samples"
UPLOADS_DIR = ROOT / "app" / "data" / "uploads"
EXAMPLES_TXT = ROOT / "app" / "data" / "examples.txt"

router = APIRouter(tags=["images"])


def _image_url(filename: str, source: str) -> str:
    return f"/api/image/{source}/{filename}"


@router.get("/samples")
async def list_samples() -> list[dict[str, Any]]:
    """Return all available sample images (bundled + uploaded)."""
    tiles: list[dict[str, Any]] = []
    for src, directory in [("samples", SAMPLES_DIR), ("uploads", UPLOADS_DIR)]:
        for img in sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png")):
            tiles.append({
                "id":         img.stem,
                "filename":   img.name,
                "url":        _image_url(img.name, src),
                "has_labels": (directory / f"{img.stem}.json").exists(),
            })
    return tiles


@router.get("/examples")
async def list_examples() -> list[dict[str, Any]]:
    """Return curated example images with descriptions from examples.txt."""
    if not EXAMPLES_TXT.exists():
        return []
    tiles: list[dict[str, Any]] = []
    for line in EXAMPLES_TXT.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 1)
        filename = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""
        img_path = SAMPLES_DIR / filename
        if img_path.exists():
            tiles.append({
                "id":          img_path.stem,
                "filename":    filename,
                "url":         _image_url(filename, "samples"),
                "description": description,
                "has_labels":  (SAMPLES_DIR / f"{img_path.stem}.json").exists(),
            })
    return tiles


@router.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    label: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Accept an image + COCO-shaped JSON label file.
    Both are required; the JSON must contain images/annotations/categories keys.
    """
    # Validate content types
    if not image.filename or not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Image must be a .jpg or .png file.")
    if not label.filename or not label.filename.lower().endswith(".json"):
        raise HTTPException(400, "Label must be a .json file.")

    # Read and validate label JSON
    label_bytes = await label.read()
    try:
        coco = json.loads(label_bytes)
    except json.JSONDecodeError:
        raise HTTPException(400, "Label file is not valid JSON.")
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise HTTPException(400, f"Label JSON is missing required key '{key}'.")

    # Read image bytes
    image_bytes = await image.read()

    # Use SHA-256 stem so re-uploads of the same file are idempotent
    stem = hashlib.sha256(image_bytes).hexdigest()[:16]
    suffix = Path(image.filename).suffix.lower()
    img_filename = f"{stem}{suffix}"
    lbl_filename = f"{stem}.json"

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / img_filename).write_bytes(image_bytes)
    (UPLOADS_DIR / lbl_filename).write_bytes(label_bytes)

    return {
        "id":       stem,
        "filename": img_filename,
        "url":      _image_url(img_filename, "uploads"),
    }


# ── Ground-truth labels ────────────────────────────────────────────────────

@router.get("/groundtruth/{image_id}")
async def get_ground_truth(image_id: str) -> dict[str, Any]:
    """Return the COCO ground-truth boxes for an image."""
    for directory in [SAMPLES_DIR, UPLOADS_DIR]:
        path = directory / f"{image_id}.json"
        if path.exists():
            coco = json.loads(path.read_text(encoding="utf-8"))
            cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
            boxes = [
                {
                    "id":         ann["id"],
                    "bbox":       ann["bbox"],          # [x, y, w, h]
                    "class_id":   ann["category_id"],
                    "class_name": cat_map.get(ann["category_id"], str(ann["category_id"])),
                    "area":       ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd":    ann.get("iscrowd", 0),
                }
                for ann in coco.get("annotations", [])
            ]
            return {"image_id": image_id, "boxes": boxes}
    raise HTTPException(404, f"No labels found for image '{image_id}'.")


# ── Static image serving ───────────────────────────────────────────────────
from fastapi.responses import FileResponse  # noqa: E402


@router.get("/image/{source}/{filename}")
async def serve_image(source: str, filename: str) -> FileResponse:
    """Serve a sample or uploaded image file."""
    if source == "samples":
        path = SAMPLES_DIR / filename
    elif source == "uploads":
        path = UPLOADS_DIR / filename
    else:
        raise HTTPException(404, "Unknown source.")
    if not path.exists():
        raise HTTPException(404, f"Image {filename!r} not found.")
    return FileResponse(path)
