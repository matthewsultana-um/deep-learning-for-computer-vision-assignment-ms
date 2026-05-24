FROM python:3.10-slim

WORKDIR /app

# System libs needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU build first (large — separate layer so Docker can cache it)
RUN pip install --no-cache-dir \
    torch==2.9.1+cpu \
    torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else (skip torch lines — already installed above)
COPY requirements.txt .
RUN grep -vE "^torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Copy application code
COPY . .

# Download model weights at build time using ultralytics (handles URL automatically)
RUN mkdir -p app/weights && \
    python -c "from ultralytics import YOLO; import shutil, pathlib; m = YOLO('yolov8n.pt'); shutil.copy(pathlib.Path(m.ckpt_path), 'app/weights/yolov8n.pt')"

# Download sample images at build time
RUN python scripts/download_samples.py

# Ensure uploads directory exists
RUN mkdir -p app/data/uploads

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
