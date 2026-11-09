import io
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

import caption
import predict

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models once at startup so the first request is not slow.
    # Set PRELOAD_MODELS=0 to skip (used by the tests).
    if os.getenv("PRELOAD_MODELS", "1") == "1":
        logger.info("Loading models...")
        predict.get_classifier()
        caption.get_captioner()
        logger.info("Models ready")
    yield


app = FastAPI(title="Multimodal Image Understanding Pipeline", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


# A plain `def` (not `async def`): inference is CPU-bound and blocking, so FastAPI
# runs it in a worker thread instead of freezing the event loop.
@app.post("/predict")
def predict_endpoint(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10),
    threshold: float = Query(0.05, ge=0.0, le=1.0),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type. Use one of: {sorted(ALLOWED_TYPES)}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decoding now so a corrupt file fails here, not later
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the file as an image")

    t0 = time.perf_counter()
    labels = predict.predict_image(image, top_k=top_k, threshold=threshold)
    t1 = time.perf_counter()
    text = caption.generate_caption(image)
    t2 = time.perf_counter()

    return {
        "labels": labels,
        "caption": text,
        "latency_ms": {
            "classification": round((t1 - t0) * 1000, 1),
            "captioning": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        },
    }
