#!/usr/bin/env python3
"""
diagnose.py  —  run this to find why val_CER is stuck
Usage: py -3.12 diagnose.py --data_root data/ --label_dir data/labels/
"""
import re, sys, argparse
from pathlib import Path
import cv2, numpy as np
from model import segment_lines

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="data/")
    p.add_argument("--label_dir",  default="data/labels/")
    p.add_argument("--n_samples",  type=int, default=20)
    args = p.parse_args()

    noisy_dir = Path(args.data_root) / "simulated_noisy_images_grayscale"
    clean_dir = Path(args.data_root) / "clean_images_grayscale"
    label_dir = Path(args.label_dir)

    files = sorted(noisy_dir.glob("*_TR.png"))[:args.n_samples]
    if not files:
        sys.exit(f"No *_TR.png files found in {noisy_dir}")

    band_match  = 0   # segment count == label line count
    band_miss   = 0   # mismatch — wrong label assigned to wrong crop
    clean_found = 0
    clean_miss  = 0
    label_found = 0
    label_miss  = 0

    print(f"Checking {len(files)} samples...\n")
    print(f"{'File':<45} {'CleanOK':>7} {'LblLines':>8} {'Bands':>5} {'Match':>5}")
    print("-" * 75)

    for noisy_path in files:
        stem      = re.sub(r"_(TR|VA|TE|RE)$", "", noisy_path.stem)
        split_tag = re.search(r"_(TR|VA|TE|RE)$", noisy_path.stem).group(0)

        # Check clean path variants
        clean_name = re.sub(r"_Noise[a-z]", "_Clean", noisy_path.name)
        clean_path = clean_dir / clean_name
        c_ok = clean_path.exists()
        if c_ok: clean_found += 1
        else:    clean_miss  += 1

        # Check label
        lbl_path = label_dir / f"{stem}.txt"
        l_ok = lbl_path.exists()
        if l_ok: label_found += 1
        else:
            label_miss += 1
            print(f"{noisy_path.name:<45} {'?':>7} {'MISSING':>8}")
            continue

        label_text  = lbl_path.read_text(encoding="utf-8").rstrip("\n")
        label_lines = [l for l in label_text.split("\n") if l.strip()]
        n_lbl       = len(label_lines)

        # Segment using best available image
        ref_path = clean_path if c_ok else noisy_path
        img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        bands = segment_lines(img) if img is not None else []
        n_bands = len(bands)

        match = "OK" if n_bands == n_lbl else "MISMATCH"
        if n_bands == n_lbl: band_match += 1
        else:                band_miss  += 1

        print(f"{noisy_path.name:<45} {str(c_ok):>7} {n_lbl:>8} {n_bands:>5} {match:>5}")

    print()
    print("=" * 75)
    print(f"Clean image found   : {clean_found}/{len(files)}")
    print(f"Label found         : {label_found}/{len(files)}")
    print(f"Band count matches  : {band_match}/{label_found}  <- this is the key number")
    print(f"Band count MISMATCH : {band_miss}/{label_found}  <- these cause wrong label assignment")
    print()
    if band_miss > label_found * 0.1:
        print("!! DIAGNOSIS: >10% band mismatches — line segmentation is unreliable.")
        print("   Fix: use label-count-aware uniform fallback, or increase segmentation")
        print("   robustness (see fix_segmentation suggestion below).")
    elif clean_miss == len(files):
        print("!! DIAGNOSIS: clean images not found — check clean_dir path and filename pattern.")
        print("   The dataset uses a different naming convention than expected.")
        print("   Run: ls data/clean_images_grayscale/ | head -5")
    else:
        print("Segmentation looks OK. Check label content and encoding next.")

    # Also print a sample label to check content
    sample_lbl = next(iter(label_dir.glob("*.txt")), None)
    if sample_lbl:
        text = sample_lbl.read_text(encoding="utf-8")
        lines = text.split("\n")
        print(f"\nSample label ({sample_lbl.name}):")
        print(f"  Total lines   : {len(lines)}")
        print(f"  Non-empty     : {sum(1 for l in lines if l.strip())}")
        print(f"  First 3 lines : {lines[:3]}")
        print(f"  Has \\r       : {chr(13) in text}")

if __name__ == "__main__":
    main()