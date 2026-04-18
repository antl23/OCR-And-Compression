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
