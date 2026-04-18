# OCR-And-Compression

# Main Pipeline Service: OCR → Adaptive Huffman Compression

This service connects the two required stages of the project into one end-to-end pipeline:

1. **Stage 1 OCR microservice** receives a noisy scanned document image and extracts text using a CNN-based OCR model.
2. **Stage 2 compression microservice** compresses that OCR output using a custom Adaptive Huffman algorithm and can then decompress it back to the original OCR text.

This pipeline service acts as the coordinator between the two microservices.

---

## Overview

The main pipeline service does not run OCR or compression itself.  
Instead, it sends requests to the existing Stage 1 and Stage 2 services and combines their outputs into a single response.

### End-to-end flow

```text
Image
  ↓
Stage 1 OCR microservice
  ↓
Extracted text
  ↓
Stage 2 compression microservice
  ↓
Compressed bitstring
  ↓
Stage 2 decompression microservice
  ↓
Recovered text
```

## Stage 1 
```text
CHECKPOINT_DIR=./runs/run_005 uvicorn server:app --host 0.0.0.0 --port 8000
```

## Stage 2
```text
uvicorn service:app --host 0.0.0.0 --port 8002
```

## Pipeline
```text
uvicorn pipeline_server:app --host 0.0.0.0 --port 9000
```
