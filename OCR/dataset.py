"""
dataset.py
==========
PyTorch Dataset for the book-scan OCR pipeline.

MULTI-LINE DESIGN
-----------------
Full-image CTC fails for multi-line scans because the number of characters
(hundreds) far exceeds the available CTC time steps after CNN width
compression.  The fix: segment each image into individual text lines at
load time and train/evaluate on one line at a time.

Each sample:
  image      : Tensor [1, LINE_H, W]   (normalised grayscale, LINE_H=32, W varies)
  text       : str                     (ground-truth text for THIS LINE only)
  line_index : int                     (which line within the source image)

The full-image label (multi-line text) is split on newlines; each line crop
is paired with its corresponding label line.

Label ↔ crop alignment
  The label file is a plain-text file where lines correspond 1-to-1 with the
  visual text lines in the image (produced by generate_labels.py using
  Tesseract PSM 6, which preserves line structure including blank lines).
  segment_lines() is called on the CLEAN grayscale image (not the noisy one)
  so that band detection is reliable; the same crop coordinates are then
  applied to the noisy image for the actual tensor.
"""

import re
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance

from model import LINE_H, segment_lines


# ── Vocabulary ────────────────────────────────────────────────────────────────
BLANK_IDX = 0


def build_vocab(label_dir: Path) -> tuple[dict, dict]:
    """Scan all .txt labels and return (char→idx, idx→char) with blank at 0."""
    chars = set()
    for txt in label_dir.glob("*.txt"):
        chars.update(txt.read_text(encoding="utf-8"))
    chars.discard("\x00")
    chars.discard("\r")
    sorted_chars = sorted(chars)
    char2idx = {c: i + 1 for i, c in enumerate(sorted_chars)}   # 1-indexed; 0=blank
    idx2char = {v: k for k, v in char2idx.items()}
    idx2char[BLANK_IDX] = ""
    return char2idx, idx2char



# ── Crop legibility thresholds ────────────────────────────────────────────────
# Adjust these if your data has unusually faint or dense print.
_MIN_CROP_HEIGHT = 8      # px  — slivers shorter than this can't hold a text baseline
_MIN_INK_RATIO   = 0.02   # fraction of pixels that must be "ink" (dark)
_MAX_INK_RATIO   = 0.90   # above this = solid smudge / ink bleed-through
_MIN_STD_DEV     = 5.0    # flat uniform-grey crop → no readable contrast


def _is_legible_crop(crop: np.ndarray, is_last_line: bool = False) -> bool:
    """
    Return True only when the crop plausibly contains human-readable text.

    Rejects four failure modes a human couldn't read:

      1. Segmentation sliver  — height too small to hold a text baseline.
      2. Washed-out / blank   — too few dark pixels; text cropped away.
      3. Solid smudge / bleed — too many dark pixels; ink bleed-through.
      4. Flat grey            — no contrast; blank margin or over-exposed scan.

    Additionally, when is_last_line=True, a fifth check catches the common
    bottom-of-page fragment problem: letters like T, i, -, n whose tops just
    peek into the bottom of the scan.  These pass checks 1-4 because they do
    contain ink and contrast — but nearly ALL of that ink sits in the top
    portion of the crop (the tips of the letters), with almost nothing below.
    A genuine full text line distributes ink across its entire height.
    """
    h, w = crop.shape

    # 1. Height gate
    if h < _MIN_CROP_HEIGHT:
        return False

    # 2. Contrast gate
    if float(crop.std()) < _MIN_STD_DEV:
        return False

    # 3. Ink-ratio gate (Otsu binarisation for scan-brightness robustness)
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(bw.sum()) / (h * w * 255)
    if ink_ratio < _MIN_INK_RATIO or ink_ratio > _MAX_INK_RATIO:
        return False

    # 4. Bottom-fragment gate — last line of page only.
    #    Letter-tip fragments concentrate their ink in the TOP of the crop
    #    (the tips of letters poking in from below the page boundary).
    #    A real readable line spreads ink across its full height.
    if is_last_line:
        # Stricter height requirement — needs room for a full character body
        if h < _MIN_CROP_HEIGHT * 2:
            return False
        top_band   = bw[: max(1, int(h * 0.40)), :]
        bottom_band = bw[int(h * 0.40):, :]
        total_ink  = float(bw.sum())
        if total_ink > 0:
            top_ratio = float(top_band.sum()) / total_ink
            # >75% of ink in the top 40% of the crop → just letter tips, drop it
            if top_ratio > 0.75:
                return False

    return True


# ── Augmentation ─────────────────────────────────────────────────────────────
class BookScanAugment:
    def __call__(self, img: Image.Image) -> Image.Image:
        # Brightness/contrast — wider range
        if random.random() < 0.8:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
        if random.random() < 0.8:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
        # Rotation — slightly more aggressive
        if random.random() < 0.5:
            angle = random.uniform(-2.0, 2.0)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        # Blur
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        # NEW: random horizontal stretch to simulate different scan widths
        if random.random() < 0.4:
            w, h = img.size
            new_w = int(w * random.uniform(0.85, 1.15))
            img = img.resize((new_w, h), Image.BILINEAR)
        # NEW: add gaussian noise
        if random.random() < 0.5:
            arr = np.array(img, dtype=np.float32)
            arr += np.random.normal(0, random.uniform(5, 20), arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img


# ── Dataset ───────────────────────────────────────────────────────────────────
class BookScanDataset(Dataset):
    """
    Parameters
    ----------
    noisy_dir  : path to simulated_noisy_images_grayscale/
    clean_dir  : path to clean_images_grayscale/  (used for line segmentation)
    label_dir  : path to generated labels/
    split      : 'TR', 'VA', 'TE', or None (all)
    char2idx   : vocabulary mapping
    max_w      : maximum line-crop width (wider crops are right-cropped)
    augment    : whether to apply augmentation (train only)
    """

    def __init__(
        self,
        noisy_dir: Path,
        clean_dir: Path,
        label_dir: Path,
        split: Optional[str],
        char2idx: dict,
        max_w: int = 768,
        augment: bool = False,
    ):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.label_dir = Path(label_dir)
        self.char2idx  = char2idx
        self.max_w     = max_w
        self.augment   = augment
        self.aug_fn    = BookScanAugment() if augment else None

        # ── Collect (noisy_path, clean_path, label_path) triples ────────────
        pattern = re.compile(
            rf"_({'|'.join([split]) if split else 'TR|VA|TE|RE'})\.png$"
        )

        # Each entry: (noisy_img_path, clean_img_path, label_lines, line_idx)
        self.samples: list[tuple[Path, Path, list[str], int]] = []
        skipped_illegible = 0

        for noisy_path in sorted(self.noisy_dir.glob("*.png")):
            if not pattern.search(noisy_path.name):
                continue

            # Derive clean and label paths from noisy filename
            stem = re.sub(r"_(TR|VA|TE|RE)$", "", noisy_path.stem)
            lbl_path = self.label_dir / f"{noisy_path.stem}.txt"
            
            # The noisy file is named Fontfre_Noisec_TR.png, but the clean 
            # file is named Fontfre_Clean_TR.png. We must translate the name:
            clean_name = re.sub(r"_Noise[a-z]", "_Clean", noisy_path.name)
            clean_path = self.clean_dir / clean_name

            if not lbl_path.exists():
                continue

            label_text = lbl_path.read_text(encoding="utf-8").rstrip("\n")
            label_lines = label_text.split("\n")
            # Remove trailing blank lines
            while label_lines and not label_lines[-1].strip():
                label_lines.pop()

            if not label_lines:
                continue

            # Pre-screen line crops for legibility.
            # Prefer the clean image for screening — better contrast for
            # Otsu binarisation.  Fall back to noisy if clean is absent.
            screen_img = None
            for candidate in (clean_path, noisy_path):
                if candidate.exists():
                    screen_img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                    if screen_img is not None:
                        break

            screen_step: float = 0.0
            if screen_img is not None:
                screen_step = screen_img.shape[0] / len(label_lines)

            for i, line_text in enumerate(label_lines):
                if not line_text.strip():
                    continue   # skip blank label lines

                # Visual legibility gate — uniform slice is cheap and good
                # enough at construction time; the accurate crop is checked
                # again inside __getitem__ after proper segment_lines().
                if screen_img is not None:
                    y0c = int(i * screen_step)
                    y1c = int((i + 1) * screen_step)
                    preview = screen_img[y0c:y1c, :]
                    if not _is_legible_crop(preview, is_last_line=(i == len(label_lines) - 1)):
                        skipped_illegible += 1
                        continue  # drop — cut-off / smudged / blank sliver

                self.samples.append((noisy_path, clean_path, label_lines, i))

        if skipped_illegible:
            print(
                f"[BookScanDataset] Dropped {skipped_illegible} illegible line crop(s) "
                f"during construction (split={split}). "
                "Tune _MIN_CROP_HEIGHT / _MIN_INK_RATIO / _MAX_INK_RATIO / _MIN_STD_DEV "
                "if too many good lines are being removed."
            )

        if not self.samples:
            raise FileNotFoundError(
                f"No line-level samples found in {noisy_dir} for split={split}.\n"
                "Check that:\n"
                "  1. generate_labels.py has been run\n"
                "  2. The split suffix matches files present\n"
                "  3. clean_dir contains matching clean images\n"
                "  4. Legibility thresholds are not too strict — see "
                     "_MIN_CROP_HEIGHT, _MIN_INK_RATIO, _MAX_INK_RATIO, _MIN_STD_DEV"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_gray(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read image: {path}")
        return img

    def _crop_line(
        self,
        full_img: np.ndarray,
        clean_img: Optional[np.ndarray],
        line_idx: int,
        n_label_lines: int,
    ) -> tuple[np.ndarray, bool]:
        """
        Segment lines using the clean image (better contrast for projection),
        then return (crop, alignment_ok).

        Passes expected_lines to segment_lines so it tries multiple thresholds
        until the band count matches.  Falls back to uniform slicing only if
        all attempts fail, and returns alignment_ok=False in that case.
        """
        ref = clean_img if (clean_img is not None and clean_img.shape == full_img.shape) else full_img
        bands = segment_lines(ref, expected_lines=n_label_lines)

        if len(bands) == n_label_lines:
            y0, y1 = bands[line_idx]
            aligned = True
        else:
            # Last-resort uniform fallback — labels may be misaligned
            h = full_img.shape[0]
            step = h / n_label_lines
            y0 = int(line_idx * step)
            y1 = int((line_idx + 1) * step)
            aligned = False

        crop = full_img[y0:y1, :]
        if crop.shape[0] < 2:
            crop = full_img
        return crop, aligned

    def _encode_label(self, text: str) -> list[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __getitem__(self, idx: int) -> dict:
        noisy_path, clean_path, label_lines, line_idx = self.samples[idx]
        n_lines = len(label_lines)
        line_text = label_lines[line_idx]

        # Load images
        noisy_img = self._load_gray(noisy_path)
        clean_img = self._load_gray(clean_path) if clean_path.exists() else None

        # Crop to the target line
        crop, aligned = self._crop_line(noisy_img, clean_img, line_idx, n_lines)

        # Second legibility gate — now using the accurate segment_lines() crop
        # rather than the cheap uniform slice used at construction time.
        # Crops that were borderline at build time and are still illegible
        # (cut-off tops/bottoms, smudges, slivers) are dropped here.
        # ctc_collate() filters out None entries so the DataLoader stays clean.
        if not _is_legible_crop(crop, is_last_line=(line_idx == n_lines - 1)):
            return None

        # Resize to LINE_H, preserve aspect ratio
        h, w = crop.shape
        new_w = min(max(1, int(w * LINE_H / h)), self.max_w)
        pil   = Image.fromarray(crop).resize((new_w, LINE_H), Image.LANCZOS)

        if self.aug_fn:
            pil = self.aug_fn(pil)

        arr    = np.array(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)        # [1, LINE_H, W]

        label  = self._encode_label(line_text)

        return {
            "image":      tensor,
            "label":      torch.tensor(label, dtype=torch.long),
            "label_len":  torch.tensor(len(label), dtype=torch.long),
            "text":       line_text,
            "image_path": str(noisy_path),
            "line_idx":   line_idx,
            "aligned":    aligned,   # False when uniform fallback was used
        }


# ── Collate ───────────────────────────────────────────────────────────────────
def ctc_collate(batch: list[dict], max_w: int = 768) -> dict:
    """
    Pad line images to the max width in the batch and stack for CTC.
    Images: [B, 1, LINE_H, W_max]

    None entries (illegible crops rejected by __getitem__) are silently
    dropped before collation.  If the entire batch is None, an empty dict
    is returned — callers should skip the training step in that case.
    """
    # Drop any samples that were flagged as illegible at runtime
    batch = [s for s in batch if s is not None]
    if not batch:
        return {}
    imgs   = [s["image"]    for s in batch]
    labels = [s["label"]    for s in batch]
    l_lens = [s["label_len"] for s in batch]
    texts  = [s["text"]     for s in batch]
    paths  = [s["image_path"] for s in batch]

    max_w_batch = min(max(img.shape[2] for img in imgs), max_w)

    padded = []
    for img in imgs:
        w = img.shape[2]
        if w < max_w_batch:
            pad = torch.zeros(1, img.shape[1], max_w_batch - w)
            img = torch.cat([img, pad], dim=2)
        else:
            img = img[:, :, :max_w_batch]
        padded.append(img)

    images_t   = torch.stack(padded, dim=0)    # [B, 1, LINE_H, W_max]
    labels_cat = torch.cat(labels)             # [sum(label_lens)]
    label_lens = torch.stack(l_lens)           # [B]

    return {
        "images":     images_t,
        "labels":     labels_cat,
        "label_lens": label_lens,
        "texts":      texts,
        "paths":      paths,
    }