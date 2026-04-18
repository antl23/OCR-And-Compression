"""
server.py
=========
FastAPI REST endpoint for the book-scan OCR model.

Endpoints
---------
POST /ocr/image
    Upload a single image file → returns recognised text.

POST /ocr/batch
    Upload multiple image files → returns a JSON map of filename → text.

GET  /health
    Liveness / readiness probe.

Setup
-----
pip install fastapi uvicorn python-multipart torch opencv-python-headless

Run
---
# Place best.pt and vocab.json in the same directory as this file, then:
uvicorn server:app --host 0.0.0.0 --port 8000

# Or point to a different checkpoint directory:
CHECKPOINT_DIR=runs/run_001 uvicorn server:app --host 0.0.0.0 --port 8000

# With auto-reload during development:
uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from model import CRNN, MultiLineCRNN

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "."))
CHECKPOINT     = CHECKPOINT_DIR / "best.pt"
VOCAB_JSON     = CHECKPOINT_DIR / "vocab.json"

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MAX_IMAGE_BYTES  = 50 * 1024 * 1024   # 50 MB hard limit per image
MAX_BATCH_FILES  = 64                  # upper bound for /ocr/batch

# ── Model state (loaded once at startup) ─────────────────────────────────────
_state: dict = {}


def _load_model() -> tuple[MultiLineCRNN, torch.device]:
    if not CHECKPOINT.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT}")
    if not VOCAB_JSON.exists():
        raise RuntimeError(f"Vocabulary file not found: {VOCAB_JSON}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading model from %s  [device=%s]", CHECKPOINT, device)

    ck = torch.load(CHECKPOINT, map_location=device)

    with open(VOCAB_JSON) as f:
        v = json.load(f)
    idx2char   = {int(k): val for k, val in v["idx2char"].items()}
    vocab_size = len(v["char2idx"])

    crnn = CRNN(vocab_size, lstm_hidden=256, lstm_layers=2).to(device)
    crnn.load_state_dict(ck["model"])
    crnn.eval()

    model = MultiLineCRNN(crnn, idx2char)
    log.info("Model ready. Vocab size: %d", vocab_size)
    return model, device


# ── Lifespan (FastAPI ≥ 0.93) ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    model, device = _load_model()
    _state["model"]  = model
    _state["device"] = device
    yield
    _state.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Book-Scan OCR API",
    description="CRNN-based OCR for book-scan images. Supports single and batch inference.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_file(upload: UploadFile) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )


def _decode_image(raw: bytes, filename: str) -> np.ndarray:
    """Decode raw bytes → grayscale numpy array."""
    arr  = np.frombuffer(raw, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {filename}",
        )
    return img


def _run_ocr(gray: np.ndarray) -> str:
    model: MultiLineCRNN = _state["model"]
    device: torch.device = _state["device"]
    return model.forward_image(gray, device)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
def health():
    """Liveness probe — returns 200 when the model is loaded."""
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    device = str(_state["device"])
    return {"status": "ok", "device": device}


@app.post("/ocr/image", tags=["OCR"])
async def ocr_single(
    file: Annotated[UploadFile, File(description="A single book-scan image (PNG/JPEG/TIFF …)")]
):
    """
    Recognise text in a **single** uploaded image.

    Returns
    -------
    ```json
    {
      "filename": "page_001.png",
      "text": "The quick brown fox …",
      "elapsed_ms": 142
    }
    ```
    """
    _validate_file(file)

    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB limit.",
        )

    gray = _decode_image(raw, file.filename or "upload")

    t0   = time.perf_counter()
    text = _run_ocr(gray)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    log.info("OCR single | file=%s | chars=%d | %dms", file.filename, len(text), elapsed_ms)
    return JSONResponse({
        "filename":   file.filename,
        "text":       text,
        "elapsed_ms": elapsed_ms,
    })


@app.post("/ocr/batch", tags=["OCR"])
async def ocr_batch(
    files: Annotated[
        list[UploadFile],
        File(description="One or more book-scan images"),
    ]
):
    """
    Recognise text in **multiple** uploaded images in one request.

    Returns
    -------
    ```json
    {
      "results": {
        "page_001.png": "The quick brown fox …",
        "page_002.png": "… jumps over the lazy dog."
      },
      "errors": {
        "bad_file.xyz": "Unsupported file type"
      },
      "total_elapsed_ms": 380
    }
    ```
    """
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum batch size is {MAX_BATCH_FILES}.",
        )

    results: dict[str, str] = {}
    errors:  dict[str, str] = {}
    t0_total = time.perf_counter()

    for upload in files:
        fname = upload.filename or f"file_{len(results)+len(errors)}"
        suffix = Path(fname).suffix.lower()

        # Validate type
        if suffix not in ALLOWED_SUFFIXES:
            errors[fname] = f"Unsupported file type '{suffix}'"
            continue

        raw = await upload.read()
        if len(raw) > MAX_IMAGE_BYTES:
            errors[fname] = f"File exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB limit"
            continue

        try:
            gray = _decode_image(raw, fname)
            text = _run_ocr(gray)
            results[fname] = text
        except HTTPException as exc:
            errors[fname] = exc.detail
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error on file %s", fname)
            errors[fname] = f"Internal error: {exc}"

    total_elapsed_ms = round((time.perf_counter() - t0_total) * 1000)
    log.info(
        "OCR batch | ok=%d err=%d | %dms",
        len(results), len(errors), total_elapsed_ms,
    )
    return JSONResponse({
        "results":          results,
        "errors":           errors,
        "total_elapsed_ms": total_elapsed_ms,
    })