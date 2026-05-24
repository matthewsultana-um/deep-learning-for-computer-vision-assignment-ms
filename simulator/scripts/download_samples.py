#!/usr/bin/env python3
"""
One-time bootstrap: populate app/data/samples/ with a COCO val2017 subset
and app/data/examples.txt with curated descriptions.

Strategy
--------
1.  Download the ultralytics COCO128 zip (~7 MB), which contains 128
    real COCO train2017 images plus YOLO-format (.txt) labels.
2.  Pick the 10 images that have the most annotations (best for NMS demo).
3.  Convert each YOLO label to a per-image COCO-shaped JSON
    ({images, annotations, categories}) that the app expects.
4.  Write app/data/examples.txt with curated descriptions for 6 of them.

Run from repo root:
    python scripts/download_samples.py
"""
import io
import json
import zipfile
import urllib.request
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "app" / "data" / "samples"
EXAMPLES_TXT = REPO_ROOT / "app" / "data" / "examples.txt"

COCO128_URL = "https://ultralytics.com/assets/coco128.zip"
N_PICK = 10  # images to keep

# 80 standard COCO category names in class-index order (0-based)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# COCO category list (id is 1-based to match COCO convention)
CATEGORIES = [
    {"id": i + 1, "name": name, "supercategory": "object"}
    for i, name in enumerate(COCO_CLASSES)
]


def _report(block_num: int, block_size: int, total: int) -> None:
    done = block_num * block_size
    if total > 0:
        pct = min(done * 100 // total, 100)
        print(f"\r  {pct:3d}%  ({done / 1e6:.1f} / {total / 1e6:.1f} MB)", end="", flush=True)


def yolo_to_coco_bbox(
    cx_n: float, cy_n: float, w_n: float, h_n: float,
    img_w: int, img_h: int,
) -> list[float]:
    """Convert normalised YOLO box to COCO [x_min, y_min, w, h] in pixels."""
    w_px = w_n * img_w
    h_px = h_n * img_h
    x_min = (cx_n - w_n / 2) * img_w
    y_min = (cy_n - h_n / 2) * img_h
    return [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)]


def parse_yolo_label(txt: str) -> list[dict]:
    rows = []
    for line in txt.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append({
            "class_id": int(parts[0]),
            "cx": float(parts[1]),
            "cy": float(parts[2]),
            "w": float(parts[3]),
            "h": float(parts[4]),
        })
    return rows


def build_coco_json(
    image_id: int,
    filename: str,
    img_w: int,
    img_h: int,
    yolo_rows: list[dict],
) -> dict:
    annotations = []
    seen_cat_ids: set[int] = set()
    for ann_id, row in enumerate(yolo_rows, start=1):
        cat_id = row["class_id"] + 1  # 1-based
        bbox = yolo_to_coco_bbox(row["cx"], row["cy"], row["w"], row["h"], img_w, img_h)
        area = round(bbox[2] * bbox[3], 2)
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
        seen_cat_ids.add(cat_id)

    categories = [c for c in CATEGORIES if c["id"] in seen_cat_ids]

    return {
        "images": [
            {"id": image_id, "file_name": filename, "width": img_w, "height": img_h}
        ],
        "annotations": annotations,
        "categories": categories,
    }


# Curated descriptions for examples.txt (filename|description)
EXAMPLES_DESCRIPTIONS = {
    "000000000009.jpg": "Scattered dinnerware — many overlapping plates and bowls stress-test IoU thresholds",
    "000000000025.jpg": "Outdoor scene with people and vehicles — shows cross-class NMS behaviour",
    "000000000030.jpg": "Sports scene — fast IoU decay across similar bounding boxes",
    "000000000034.jpg": "Kitchen clutter — dense small objects, useful for high confidence-threshold demo",
    "000000000036.jpg": "Street scene with pedestrians — classic multi-person NMS use case",
    "000000000042.jpg": "Traffic intersection — overlapping cars demonstrate aggressive NMS suppression",
    "000000000049.jpg": "Group portrait — high-overlap person boxes, ideal for Soft-NMS comparison",
    "000000000061.jpg": "Bathroom objects — sparse scene useful for IoU=1 degenerate-case warning",
    "000000000064.jpg": "Indoor furniture — moderate overlap, good baseline for AP/mAP exploration",
    "000000000072.jpg": "Animals in a field — multi-class detections with varying confidence spreads",
}


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    already = list(SAMPLES_DIR.glob("*.jpg"))
    if len(already) >= N_PICK:
        print(f"Samples already present ({len(already)} images). Delete {SAMPLES_DIR} to re-download.")
        return

    print(f"Downloading COCO128 from {COCO128_URL} …")
    buf = io.BytesIO()
    urllib.request.urlretrieve(COCO128_URL, filename=None, reporthook=_report)  # type: ignore[arg-type]
    # urlretrieve with filename=None returns a temp file; use urlopen instead
    with urllib.request.urlopen(COCO128_URL) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 64 * 1024
        downloaded = 0
        while True:
            data = resp.read(chunk)
            if not data:
                break
            buf.write(data)
            downloaded += len(data)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct:3d}%  ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)", end="", flush=True)
    print()

    print("Extracting images and labels …")
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
        img_entries = [n for n in names if n.startswith("coco128/images/") and n.endswith(".jpg")]
        lbl_entries = {
            Path(n).stem: n
            for n in names
            if n.startswith("coco128/labels/") and n.endswith(".txt")
        }

        # Score each image by annotation count; pick top N_PICK
        scored: list[tuple[int, str]] = []
        for img_path in img_entries:
            stem = Path(img_path).stem
            lbl_path = lbl_entries.get(stem)
            if lbl_path is None:
                continue
            txt = zf.read(lbl_path).decode()
            count = len([l for l in txt.strip().splitlines() if l.strip()])
            scored.append((count, img_path))

        scored.sort(reverse=True)
        selected = scored[:N_PICK]
        print(f"Selected {N_PICK} images (annotation counts: {[s for s, _ in selected]})")

        ann_global_id = 1
        for image_id, (ann_count, img_zip_path) in enumerate(selected, start=1):
            stem = Path(img_zip_path).stem
            filename = Path(img_zip_path).name

            # Write image
            img_bytes = zf.read(img_zip_path)
            img_dest = SAMPLES_DIR / filename
            img_dest.write_bytes(img_bytes)

            # Parse image dimensions
            with Image.open(io.BytesIO(img_bytes)) as im:
                img_w, img_h = im.size

            # Parse YOLO labels
            lbl_path = lbl_entries.get(stem)
            yolo_rows: list[dict] = []
            if lbl_path:
                txt = zf.read(lbl_path).decode()
                yolo_rows = parse_yolo_label(txt)

            # Re-number annotation IDs globally
            coco = build_coco_json(image_id, filename, img_w, img_h, yolo_rows)
            for ann in coco["annotations"]:
                ann["id"] = ann_global_id
                ann_global_id += 1

            label_dest = SAMPLES_DIR / f"{stem}.json"
            with open(label_dest, "w") as f:
                json.dump(coco, f, indent=2)

            print(f"  {filename}  ({img_w}×{img_h}, {len(yolo_rows)} annotations)")

    # Write examples.txt  — use whichever selected filenames match descriptions,
    # falling back to a generic description for any that don't have one.
    lines: list[str] = []
    for _, img_zip_path in selected:
        filename = Path(img_zip_path).name
        desc = EXAMPLES_DESCRIPTIONS.get(
            filename,
            "COCO scene — multiple overlapping detections ideal for NMS visualisation",
        )
        lines.append(f"{filename}|{desc}")

    EXAMPLES_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {EXAMPLES_TXT}")
    print("Done — samples are in app/data/samples/")


if __name__ == "__main__":
    main()
