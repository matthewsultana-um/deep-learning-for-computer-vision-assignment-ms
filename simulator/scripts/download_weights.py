#!/usr/bin/env python3
"""
One-time bootstrap: copy YOLOv8n weights into app/weights/yolov8n.pt.

Ultralytics downloads the file to its internal cache on first load;
this script triggers that download then copies the cached file to the
repo-local path so the app never fetches it at runtime.

Run from repo root:
    python scripts/download_weights.py
"""
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "app" / "weights" / "yolov8n.pt"


def main() -> None:
    if DEST.exists():
        print(f"Weights already present at {DEST}  ({DEST.stat().st_size / 1e6:.1f} MB)")
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading yolov8n.pt via ultralytics (~6 MB) …")

    # Let ultralytics handle the download, then find where it landed.
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO("yolov8n.pt")

    # Try locations in priority order
    candidates = [
        Path("yolov8n.pt"),                                         # cwd (older ultralytics)
        Path(getattr(model, "ckpt_path", "") or ""),                 # model attribute
        Path.home() / ".cache" / "ultralytics" / "yolov8n.pt",
        Path.home() / "AppData" / "Local" / "Ultralytics" / "yolov8n.pt",
    ]
    src = next((p for p in candidates if p.exists()), None)

    if src is None:
        # Walk user home for a fallback (slow but reliable)
        import glob
        hits = glob.glob(str(Path.home() / "**" / "yolov8n.pt"), recursive=True)
        if hits:
            src = Path(hits[0])
        else:
            raise FileNotFoundError(
                "ultralytics downloaded yolov8n.pt but the path is unknown. "
                "Locate it manually and copy to app/weights/yolov8n.pt."
            )

    shutil.copy2(src, DEST)
    print(f"Saved {DEST.stat().st_size / 1e6:.1f} MB → {DEST}")


if __name__ == "__main__":
    main()
