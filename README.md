# OCR-And-Compression

# Stage 2: Adaptive Huffman Compression Microservice

This project implements **Stage 2** of a two-stage neural compression pipeline.  
The microservice accepts text input, compresses it using a **custom Adaptive Huffman Encoding** implementation, and supports **lossless decompression** of the original text.

## Overview

The goal of this stage is to take OCR text output from Stage 1 and compress it without using prebuilt compression libraries such as `zlib` or `gzip`.

This implementation:
- exposes a microservice API
- compresses text using a custom Adaptive Huffman algorithm
- decompresses the compressed representation back to the exact original text
- reports useful compression metrics

## Features

- Custom **Adaptive Huffman Encoding**
- Lossless decompression
- FastAPI microservice
- `/health` endpoint for service verification
- `/compress` endpoint for compression and round-trip validation
- Metrics reporting:
  - compression ratio
  - space saving
  - entropy
  - average code length
  - encoding efficiency

## Project Structure

```text
stage2_compression/
├── adaptive_huffman.py   # Core Adaptive Huffman encoder/decoder
├── service.py            # FastAPI microservice
└── __pycache__/          # Python cache files
```
## Requirements

- Python 3.10+
- pip
- Virtual environment (recommended)

---

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

Use virtual environment:
```bash
cd stage2_compression
source ../.venv/bin/activate
uvicorn service:app --host 0.0.0.0 --port 8002 --reload
```

## API Endpoints

Health Check
```bash
curl http://127.0.0.1:8002/health
```

Compress Text
```bash
curl -s -X POST "http://127.0.0.1:8002/compress" \
  -H "Content-Type: application/json" \
  -d '{"text":"this is a test this is a test this is a test"}' | python3 -m json.tool
```

## Output Fields
- **`original_text`**: input text given to the service  
- **`bitstring`**: compressed binary output  
- **`compressed_bytes_hex`**: packed compressed output in hex form  
- **`decoded_text`**: decompressed result  
- **`metrics`**: compression performance metrics  
- **`alphabet_size`**: number of unique symbols in the text  
- **`symbol_count`**: total number of symbols processed  

