# Visual Explorer Simulator

A CPU-only, browser-based educational tool that visualises **Non-Maximum
Suppression (NMS)**, **Soft-NMS**, and **AP/mAP computation** step by step on
YOLOv8 detections. Load an image, watch the algorithm run iteration by
iteration, and inspect the resulting precision/recall metrics.


----

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.127 |
| pip | any recent version |

Python 3.127 - https://www.python.org/downloads/release/python-3127/

No GPU is needed — the simulator is **CPU-only** by design.

---

## Getting Started

### 1. Clone the repository

```
git clone https://github.com/matthewsultana-um/deep-learning-for-computer-vision-assignment-ms.git

```

### 2. Navigate sub-folder
```
cd deep-learning-for-computer-vision-assignment-ms\simulator
```
### 3. Create venv

```
python -m venv venv
```
### 4. Activate venv
```
venv\Scripts\activate
```

### 5. Install dependencies

```
pip install -r requirements.txt
```

### 5. Download model weights
python -c "from ultralytics import YOLO; import shutil, pathlib; m = YOLO('yolov8n.pt'); shutil.copy(pathlib.Path(m.ckpt_path), 'app/weights/yolov8n.pt')"


### 6. Run script
```
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 7. Open folder

Open in browser

```
http://localhost:8000
```




