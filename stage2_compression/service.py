from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException

from adaptive_huffman import AdaptiveHuffman

sys.path.append(str(Path(__file__).resolve().parents[1]))
from shared.schemas import CompressionRequest, CompressionResponse

app = FastAPI(title="Stage 2 Compression Microservice", version="1.0.0")
codec = AdaptiveHuffman()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "algorithm": "FGK adaptive Huffman"}


@app.post("/compress", response_model=CompressionResponse)
def compress(req: CompressionRequest) -> CompressionResponse:
    text = req.text
    bitstring = codec.encode(text)
    blob = codec.pack_bits(bitstring)
    decoded = codec.decode(bitstring)
    if decoded != text:
        raise HTTPException(status_code=500, detail="Lossless recovery failed during self-check")
    return CompressionResponse(
        original_text=text,
        bitstring=bitstring,
        compressed_bytes_hex=blob.hex(),
        decoded_text=decoded,
        metrics=codec.metrics(text, bitstring),
        alphabet_size=len(set(text)),
        symbol_count=len(text),
    )


@app.post("/decompress")
def decompress(payload: dict) -> dict:
    if "bitstring" in payload:
        bitstring = payload["bitstring"]
    elif "compressed_bytes_hex" in payload:
        raw = bytes.fromhex(payload["compressed_bytes_hex"])
        bitstring = codec.unpack_bits(raw)
    else:
        raise HTTPException(status_code=400, detail="Provide bitstring or compressed_bytes_hex")

    text = codec.decode(bitstring)
    return {"text": text, "symbol_count": len(text)}
