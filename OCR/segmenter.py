"""
prepare_lines.py
================
Slices multi-line document images and their corresponding multi-line text labels
into single-line crops for CTC training.
"""

import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_line_crops(image_path: Path):
    """Uses Horizontal Projection to slice an image into horizontal line strips."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    # 1. Binarize and invert (text = white, background = black)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Horizontal Projection (sum pixels across the width)
    proj = np.sum(thresh, axis=1)

    # 3. Find rows where the projection exceeds a noise threshold
    threshold = np.max(proj) * 0.05
    is_text = proj > threshold

    # 4. Find the start and end of continuous text blocks
    diff = np.diff(is_text.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if is_text[0]:  starts = np.insert(starts, 0, 0)
    if is_text[-1]: ends = np.append(ends, len(is_text))

    # 5. Crop the original image using those coordinates
    crops = []
    H, W = img.shape
    for s, e in zip(starts, ends):
        s_pad = max(0, s - 5)  # Add 5px padding top
        e_pad = min(H, e + 5)  # Add 5px padding bottom
        
        # Filter out tiny noise blobs (less than 10px tall)
        if (e_pad - s_pad) > 10:
            crops.append(img[s_pad:e_pad, :])
            
    return crops

def main():
    noisy_dir = Path("data/simulated_noisy_images_grayscale")
    label_dir = Path("data/labels")
    
    # New output directories
    out_img_dir = Path("data/lines/images")
    out_lbl_dir = Path("data/lines/labels")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list(noisy_dir.glob("*.png"))
    success = 0
    missing_labels = 0
    mismatched_lines = 0

    for img_path in tqdm(images, desc="Segmenting Lines"):
        # Just use the noisy stem, since that is how generate_labels.py saved them
        stem = re.sub(r"_(TR|VA|TE|RE)$", "", img_path.stem)
        lbl_path = label_dir / f"{stem}.txt"

        if not lbl_path.exists():
            missing_labels += 1
            continue

        # Get lines of text, ignoring empty lines
        text_lines = [line.strip() for line in lbl_path.read_text(encoding="utf-8").split("\n") if line.strip()]
        
        # Get image strips
        img_lines = get_line_crops(img_path)

        # Critical Check: Ensure image lines found match text lines found
        if len(text_lines) != len(img_lines):
            mismatched_lines += 1
            continue

        # Save the pairs
        for i, (crop, text) in enumerate(zip(img_lines, text_lines)):
            line_name = f"{img_path.stem}_L{i:02d}"
            
            # Save Image
            cv2.imwrite(str(out_img_dir / f"{line_name}.png"), crop)
            
            # Save Label
            (out_lbl_dir / f"{line_name}.txt").write_text(text, encoding="utf-8")
            
            success += 1

    print(f"\n✓ Generated {success} single-line training pairs.")
    
    if missing_labels > 0:
        print(f"✗ Skipped {missing_labels} files because the .txt label was missing.")
    if mismatched_lines > 0:
        print(f"✗ Skipped {mismatched_lines} files because the OpenCV contour count didn't match the text line count.")

if __name__ == "__main__":
    main()