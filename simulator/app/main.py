"""
FastAPI application entry point for the Visual Explorer Simulator.

Mounts static files, registers Jinja2 templates, and wires up all API routers.
Run with:  python -m app
"""
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent


def _warmup_and_precache() -> None:
    """
    Load YOLO, JIT-compile the inference path, then pre-run every bundled
    sample image so that all /api/detect calls for samples are instant cache
    hits — no YOLO inference happens at request time for sample images.
    """
    from app.detection.yolo import run_inference, SAMPLES_DIR

    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not samples:
        # No sample images — fall back to numpy warmup
        import numpy as np
        from app.detection.yolo import _get_model
        model = _get_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, conf=0.01, iou=1.0, device="cpu", verbose=False)
        return

    total = len(samples)
    for i, img_path in enumerate(samples, 1):
        print(f"  Pre-caching {img_path.name} ({i}/{total})…", flush=True)
        run_inference(img_path.stem)
    print(f"All {total} sample images cached.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load + JIT-warm the YOLO model so the first /detect request is fast."""
    loop = asyncio.get_event_loop()
    print("Pre-loading YOLO + caching all sample images (one-time, ~60 s)…", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(pool, _warmup_and_precache)
        print("YOLO model ready — server accepting requests.", flush=True)
    except Exception as exc:
        print(f"Pre-cache skipped ({exc}); first click may be slow.", flush=True)
    yield  # server runs here


app = FastAPI(title="Visual Explorer Simulator", version="1.0.0", lifespan=lifespan)

# Static assets (CSS, JS, icons)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=ROOT / "templates")

# ── API routers ───────────────────────────────────────────────────────────────
from app.api import images, detection, nms, metrics  # noqa: E402

app.include_router(images.router,    prefix="/api")
app.include_router(detection.router, prefix="/api")
app.include_router(nms.router,       prefix="/api")
app.include_router(metrics.router,   prefix="/api")

# ── Page route ────────────────────────────────────────────────────────────────
from fastapi import Request  # noqa: E402


@app.get("/", include_in_schema=False)
async def index(request: Request):
    """Serve the single-page simulator."""
    return templates.TemplateResponse(request, "index.html")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
