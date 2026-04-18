from __future__ import annotations

import os
import time
from typing import Annotated

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse


STAGE1_URL = os.getenv("STAGE1_URL", "http://127.0.0.1:8000")
STAGE2_URL = os.getenv("STAGE2_URL", "http://127.0.0.1:8002")

OCR_SINGLE_ENDPOINT = f"{STAGE1_URL}/ocr/image"
OCR_BATCH_ENDPOINT = f"{STAGE1_URL}/ocr/batch"
COMPRESS_ENDPOINT = f"{STAGE2_URL}/compress"
DECOMPRESS_ENDPOINT = f"{STAGE2_URL}/decompress"

TIMEOUT_SECONDS = 120
MAX_BATCH_FILES = 64


app = FastAPI(
    title="2-Stage OCR + Compression Pipeline",
    description="Main pipeline service that connects Stage 1 OCR to Stage 2 Adaptive Huffman compression.",
    version="1.0.0",
)


def _check_service(url: str, name: str) -> dict:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return {"name": name, "ok": True, "response": r.json()}
    except Exception as e:
        return {"name": name, "ok": False, "error": str(e)}


@app.get("/health", tags=["Utility"])
def health():
    stage1 = _check_service(f"{STAGE1_URL}/health", "stage1")
    stage2 = _check_service(f"{STAGE2_URL}/health", "stage2")

    overall_ok = stage1["ok"] and stage2["ok"]
    status_code = 200 if overall_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if overall_ok else "degraded",
            "stage1": stage1,
            "stage2": stage2,
        },
    )


@app.post("/pipeline/image", tags=["Pipeline"])
async def pipeline_single(
    file: Annotated[UploadFile, File(description="A single scanned document image")]
):
    """
    Full end-to-end pipeline:
    image -> Stage 1 OCR -> Stage 2 compression -> Stage 2 decompression
    """
    filename = file.filename or "upload"
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    t0 = time.perf_counter()

    try:
        # Stage 1 OCR
        files = {
            "file": (filename, raw, file.content_type or "application/octet-stream")
        }
        r1 = requests.post(OCR_SINGLE_ENDPOINT, files=files, timeout=TIMEOUT_SECONDS)
        r1.raise_for_status()
        ocr_result = r1.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Stage 1 OCR request failed: {e}")

    extracted_text = ocr_result.get("text", "")
    if extracted_text is None:
        extracted_text = ""

    try:
        # Stage 2 compress
        r2 = requests.post(
            COMPRESS_ENDPOINT,
            json={"text": extracted_text},
            timeout=TIMEOUT_SECONDS,
        )
        r2.raise_for_status()
        compress_result = r2.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Stage 2 compress request failed: {e}")

    bitstring = compress_result.get("bitstring", "")
    if bitstring is None:
        bitstring = ""

    try:
        # Stage 2 decompress
        r3 = requests.post(
            DECOMPRESS_ENDPOINT,
            json={"bitstring": bitstring},
            timeout=TIMEOUT_SECONDS,
        )
        r3.raise_for_status()
        decompress_result = r3.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Stage 2 decompress request failed: {e}")

    decoded_text = decompress_result.get("decoded_text", "")
    roundtrip_ok = decoded_text == extracted_text
    total_elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return JSONResponse({
        "filename": filename,
        "ocr": ocr_result,
        "compression": compress_result,
        "decompression": decompress_result,
        "roundtrip_ok": roundtrip_ok,
        "total_elapsed_ms": total_elapsed_ms,
    })


@app.post("/pipeline/batch", tags=["Pipeline"])
async def pipeline_batch(
    files: Annotated[list[UploadFile], File(description="One or more scanned document images")]
):
    """
    Batch pipeline:
    images -> Stage 1 batch OCR -> Stage 2 compress/decompress for each OCR result
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size is {MAX_BATCH_FILES}.",
        )

    prepared_files = []
    for f in files:
        raw = await f.read()
        prepared_files.append(
            ("files", (f.filename or "upload", raw, f.content_type or "application/octet-stream"))
        )

    t0 = time.perf_counter()

    try:
        r1 = requests.post(OCR_BATCH_ENDPOINT, files=prepared_files, timeout=TIMEOUT_SECONDS)
        r1.raise_for_status()
        batch_ocr = r1.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Stage 1 batch OCR request failed: {e}")

    results = {}
    errors = dict(batch_ocr.get("errors", {}))

    for fname, text in batch_ocr.get("results", {}).items():
        try:
            r2 = requests.post(
                COMPRESS_ENDPOINT,
                json={"text": text},
                timeout=TIMEOUT_SECONDS,
            )
            r2.raise_for_status()
            compress_result = r2.json()

            bitstring = compress_result.get("bitstring", "")

            r3 = requests.post(
                DECOMPRESS_ENDPOINT,
                json={"bitstring": bitstring},
                timeout=TIMEOUT_SECONDS,
            )
            r3.raise_for_status()
            decompress_result = r3.json()

            decoded_text = decompress_result.get("decoded_text", "")
            results[fname] = {
                "ocr_text": text,
                "compression": compress_result,
                "decompression": decompress_result,
                "roundtrip_ok": decoded_text == text,
            }

        except requests.RequestException as e:
            errors[fname] = f"Stage 2 request failed: {e}"

    total_elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return JSONResponse({
        "results": results,
        "errors": errors,
        "total_elapsed_ms": total_elapsed_ms,
    })
