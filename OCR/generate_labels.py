"""
generate_labels.py
==================
Generates ground-truth text labels for every image in the dataset.

Strategy
--------
We have *four* renderings of every scene:
  1. clean_images_grayscale/              (clean, full-res)
  2. clean_images_grayscale_doubleresolution/  (clean, 2× res  ← best for OCR)
  3. clean_images_binaryscale_lowresolution/   (binary, lower-res  ← fast fallback)
  4. simulated_noisy_images_grayscale/    (noisy  ← do NOT use for labelling)

Approach:
  • Run Tesseract on the 2× clean image (best signal-to-noise + resolution).
  • If confidence is low, fall back to the binary image.
  • Output one  <stem>.txt  per image (stem = everything before _TR/_VA/_TE/_RE).
  • A single label covers all four noise variants that share the same stem prefix.

Usage:
  python generate_labels.py --data_root data/ --out_dir data/labels/ [--workers 8]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm


# ── Tesseract config ─────────────────────────────────────────────────────────
# PSM 6: assume a uniform block of text (best for multi-line crops)
TESS_CONFIG = r"--oem 3 --psm 6"
CONFIDENCE_THRESHOLD = 60          # mean word-confidence below this → use fallback


def _preprocess_for_tesseract(img: np.ndarray) -> np.ndarray:
    """Light preprocessing that helps Tesseract on book-scan crops."""
    # Ensure uint8 grayscale
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mild CLAHE to normalise contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Sharpen slightly
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)
    return img


def _ocr_image(path: Path, config: str = TESS_CONFIG) -> tuple[str, float]:
    """Return (text, mean_confidence) for a single image path."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "", 0.0

    img = _preprocess_for_tesseract(img)
    pil_img = Image.fromarray(img)

    try:
        data = pytesseract.image_to_data(
            pil_img, config=config, output_type=pytesseract.Output.DICT
        )
        confidences = [c for c in data["conf"] if c != -1]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        text = pytesseract.image_to_string(pil_img, config=config)
    except Exception:
        text, mean_conf = "", 0.0

    return text.strip(), mean_conf


def _stem_from_filename(fname: str) -> str:
    # OLD: 'Fontfre_Noisec_TR.png' → 'Fontfre_Noisec'  (drops split)
    # NEW: 'Fontfre_Noisec_TR.png' → 'Fontfre_Noisec_TR'  (keeps split)
    return Path(fname).stem

def generate_label(
    stem: str,          # now includes split, e.g. 'Fontfre_Noisec_TR'
    hr_dir: Path,
    bin_dir: Path,
    clean_dir: Path,
    out_dir: Path,
) -> tuple[str, bool]:
    out_path = out_dir / f"{stem}.txt"
    if out_path.exists():
        return stem, True

    # stem already includes split suffix — find the clean equivalent directly
    clean_stem = re.sub(r"_Noise.", "_Clean", stem)  # e.g. Fontfre_Clean_TR

    def find_file(directory: Path) -> Path | None:
        p = directory / f"{clean_stem}.png"
        return p if p.exists() else None

    hr_path    = find_file(hr_dir)
    clean_path = find_file(clean_dir)
    bin_path   = find_file(bin_dir)

    if hr_path is None and clean_path is None and bin_path is None:
        return stem, False

    text, conf = "", 0.0
    if hr_path:
        text, conf = _ocr_image(hr_path)
    if conf < CONFIDENCE_THRESHOLD and clean_path:
        t2, c2 = _ocr_image(clean_path)
        if c2 > conf:
            text, conf = t2, c2
    if conf < CONFIDENCE_THRESHOLD and bin_path:
        t3, c3 = _ocr_image(bin_path)
        if c3 > conf:
            text, conf = t3, c3

    if not text:
        return stem, False

    out_path.write_text(text, encoding="utf-8")
    return stem, True


def main():
    parser = argparse.ArgumentParser(description="Generate OCR labels from clean images")
    parser.add_argument("--data_root", default="data/", help="Root of the data/ folder")
    parser.add_argument("--out_dir",   default="data/labels/", help="Where to write .txt labels")
    parser.add_argument("--workers",   type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hr_dir    = data_root / "clean_images_grayscale_doubleresolution"
    clean_dir = data_root / "clean_images_grayscale"
    bin_dir   = data_root / "clean_images_binaryscale_lowresolution"
    noisy_dir = data_root / "simulated_noisy_images_grayscale"

    # Collect all unique stems from the noisy folder (that's our model input universe)
    all_files = list(noisy_dir.glob("*.png"))
    if not all_files:
        sys.exit(f"[ERROR] No PNG files found in {noisy_dir}")

    stems = sorted({_stem_from_filename(f.name) for f in all_files})
    print(f"Found {len(stems)} unique (font, noise) stems across {len(all_files)} images.")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_label, stem, hr_dir, bin_dir, clean_dir, out_dir): stem
            for stem in stems
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Labelling"):
            _, success = fut.result()
            if success:
                ok += 1
            else:
                fail += 1

    print(f"\n✓ Labels written : {ok}")
    print(f"✗ Failed / skipped: {fail}")
    print(f"Labels saved to  : {out_dir}")


if __name__ == "__main__":
    main()
