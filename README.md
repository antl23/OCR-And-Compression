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
- fastapi==0.115.0
- uvicorn==0.30.6
- pydantic==2.9.2
- requests==2.32.3
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
Example output
```JSON
{
    "original_text": "this is a test this is a test this is a test",
    "bitstring": "001110100001101000100110100100011100111100010000011110100000001100001110111110100110010111111111100011101111111111011100001110111101101111111010000110011110111011110001111011101101111111",
    "compressed_bytes_hex": "063a1a2691cf107a030efa65ff8effdc3bdbfa19eef1eedfc0",
    "metrics": {
        "original_bits": 352.0,
        "compressed_bits": 186.0,
        "compression_ratio": 1.89247311827957,
        "space_saving": 0.47159090909090906,
        "entropy_bits_per_symbol": 2.621094451778451,
        "average_code_length": 4.2272727272727275,
        "encoding_efficiency": 0.6200438488078056
    },
    "alphabet_size": 7,
    "symbol_count": 44
}
```

Decompress Text
- Use the bitstring returned by /compress.
```bash
curl -s -X POST "http://127.0.0.1:8002/decompress" \
  -H "Content-Type: application/json" \
  -d '{"bitstring":"001110100001101000100110100100011100111100010000011110100000001100001110111110100110010111111111100011101111111111011100001110111101101111111010000110011110111011110001111011101101111111"}' | python3 -m json.tool
```
Example Ouput
```JSON
{
    "bitstring": "001110100001101000100110100100011100111100010000011110100000001100001110111110100110010111111111100011101111111111011100001110111101101111111010000110011110111011110001111011101101111111",
    "decoded_text": "this is a test this is a test this is a test"
}
```

## Output Fields
- **`original_text`**: input text given to the service  
- **`bitstring`**: compressed binary output  
- **`compressed_bytes_hex`**: packed compressed output in hex form  
- **`decoded_text`**: decompressed result  
- **`metrics`**: compression performance metrics  
- **`alphabet_size`**: number of unique symbols in the text  
- **`symbol_count`**: total number of symbols processed  

