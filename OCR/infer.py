"""
infer.py
========
Run the trained model on any book-scan crop and print the recognised text.

The model was trained on individual line crops.  At inference, this script
uses MultiLineCRNN.forward_image() to:
  1. Segment the full image into text-line bands (projection profile)
  2. Run the per-line CRNN on each band
  3. Join the decoded lines with newlines

Usage
-----
# Single image
python infer.py --image path/to/scan.png --checkpoint runs/run_001/best.pt

# Batch mode
python infer.py --image_dir some_folder/ --checkpoint runs/run_001/best.pt --out results.json
"""

import argparse
import json
from pathlib import Path

import cv2
import torch

from model import CRNN, MultiLineCRNN


def load_model(checkpoint: Path, device: torch.device):
    ck = torch.load(checkpoint, map_location=device)

    vocab_json = checkpoint.parent / "vocab.json"
    with open(vocab_json) as f:
        v = json.load(f)
    char2idx   = v["char2idx"]
    idx2char   = {int(k): val for k, val in v["idx2char"].items()}
    vocab_size = len(char2idx)

    crnn = CRNN(vocab_size, lstm_hidden=256, lstm_layers=2).to(device)
    crnn.load_state_dict(ck["model"])
    crnn.eval()

    model = MultiLineCRNN(crnn, idx2char)
    return model, idx2char


def predict_one(model, image_path: Path, device: torch.device) -> str:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Cannot read: {image_path}")
    return model.forward_image(gray, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      default=None,  help="Single image path")
    p.add_argument("--image_dir",  default=None,  help="Directory of PNG images")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--out",        default=None,  help="JSON output (batch mode)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.checkpoint), device)
    print(f"Model loaded  [{device}]")

    if args.image:
        text = predict_one(model, Path(args.image), device)
        print("\n─── Recognised Text ───────────────────────────")
        print(text)
        print("───────────────────────────────────────────────")

    elif args.image_dir:
        paths   = sorted(Path(args.image_dir).glob("*.png"))
        results = {}
        for path in paths:
            try:
                text = predict_one(model, path, device)
                results[path.name] = text
                preview = text.replace("\n", " ↵ ")
                print(f"{path.name}: {preview[:100]}{'...' if len(preview)>100 else ''}")
            except Exception as e:
                results[path.name] = f"ERROR: {e}"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.out}")
    else:
        p.error("Provide --image or --image_dir")


if __name__ == "__main__":
    main()