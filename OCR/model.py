"""
model.py
========
CRNN (CNN + BiLSTM) for multi-line OCR with CTC loss.

WHY THE ORIGINAL DESIGN FAILS FOR MULTI-LINE
---------------------------------------------
The naive approach — resize the full image to a fixed height (e.g. H=64)
and collapse height to 1 via pooling — catastrophically compresses WIDTH:

  Original image   520 × 240 px  (landscape, 8 lines of text)
  Resized to H=64: 138 × 64  px
  After CNN (two (2,2) MaxPool): W' = 138 // 4 = 34 time steps

  But 8 lines × ~60 chars/line = ~480 characters.
  CTC requires T ≥ L (strictly).  34 << 480 → training is impossible.

For larger images the situation is worse (480 tall → T=17 steps).

THE CORRECT ARCHITECTURE: Line Segmentation + Per-Line CRNN
------------------------------------------------------------
Step 1  segment_lines()
        Use a horizontal projection profile on the binarised image to find
        the row-ranges that contain text.  Each range → one line crop.

Step 2  Per-line CRNN (this class)
        Each cropped line is resized to a fixed HEIGHT (LINE_H = 32 px).
        Width is kept proportional (~520 px in practice).
        The CNN uses HEIGHT-ONLY pooling so width is never compressed:

          Input  [B, 1, 32, W]
          After five (2,1) MaxPool steps: H: 32→1, W unchanged
          Output [B, 256, 1, W]   → W time steps for CTC

        With W ≈ 520:  T = 520 >> L ≈ 60 chars/line  ✓

Step 3  MultiLineCRNN.forward_image()
        Segments the full image, runs the line CRNN on every line crop,
        concatenates the decoded text with newlines.

This approach:
  • Always satisfies the CTC T ≥ L constraint
  • Requires NO extra annotations (line boxes not needed — projection works)
  • Handles variable numbers of lines naturally
  • Trains on individual line crops → very efficient batching
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────
LINE_H = 32          # every line crop is resized to this height
MIN_LINE_H_PX = 8    # discard projection valleys shallower than this


# ── Line segmentation ─────────────────────────────────────────────────────────
def segment_lines(
    gray: np.ndarray,           # H×W uint8 grayscale
    min_line_height: int = MIN_LINE_H_PX,
    padding: int = 3,
    expected_lines: int = 0,    # if >0, merge/split until count matches
) -> list[tuple[int, int]]:
    """
    Return a list of (y_start, y_end) row ranges, one per text line.

    Algorithm
    ---------
    1. Denoise then binarise with Otsu (invert so text = white).
    2. Morphological closing to merge broken characters within a line.
    3. Horizontal projection profile → smooth → threshold.
    4. If expected_lines > 0 and count mismatches, try a range of
       thresholds until the count matches (or return best effort).
    5. Pad bands and clamp to image boundaries.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Mild denoise before binarisation
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarise: THRESH_BINARY_INV so text pixels = 255
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing along the horizontal axis ONLY (width=20, height=1):
    # merges broken chars within a line WITHOUT bridging across inter-line gaps.
    # The old (30,3) kernel was too tall and merged adjacent lines into one band.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    def _find_bands(bw_img: np.ndarray, threshold_frac: float,
                    smooth: int = 3) -> list[tuple[int, int]]:
        proj = bw_img.sum(axis=1).astype(np.float32)
        if smooth > 1:
            k = np.ones(smooth, dtype=np.float32) / smooth
            proj = np.convolve(proj, k, mode="same")
        thr = proj.max() * threshold_frac
        in_band, bands, y_start = False, [], 0
        for y, val in enumerate(proj):
            if not in_band and val > thr:
                in_band, y_start = True, y
            elif in_band and val <= thr:
                in_band = False
                if (y - y_start) >= min_line_height:
                    bands.append((
                        max(0, y_start - padding),
                        min(bw_img.shape[0], y + padding),
                    ))
        if in_band and (len(proj) - y_start) >= min_line_height:
            bands.append((max(0, y_start - padding), bw_img.shape[0]))
        return bands

    def _split_band_at_valley(
        bw_img: np.ndarray,
        y0: int, y1: int,
        n_splits: int = 1,
    ) -> list[tuple[int, int]]:
        """
        Split a band (y0..y1) into n_splits+1 pieces by finding the
        n_splits deepest local minima in the horizontal projection profile.
        Used when a band is suspected to contain multiple merged lines.
        """
        proj = bw_img[y0:y1, :].sum(axis=1).astype(np.float32)
        if len(proj) < 2 * min_line_height:
            return [(y0, y1)]

        # Score each interior row by how much it dips below its neighbours
        margin = min_line_height
        interior = proj[margin:-margin]
        if len(interior) == 0:
            return [(y0, y1)]

        scores = np.zeros(len(interior), dtype=np.float32)
        for i in range(1, len(interior) - 1):
            scores[i] = (interior[i - 1] + interior[i + 1]) / 2.0 - interior[i]

        # Pick the n_splits deepest valleys (non-adjacent within min_line_height)
        split_rows_local: list[int] = []
        tmp = scores.copy()
        for _ in range(n_splits):
            best = int(np.argmax(tmp))
            split_rows_local.append(best + margin)
            lo = max(0, best - min_line_height)
            hi = min(len(tmp), best + min_line_height)
            tmp[lo:hi] = -1e9

        split_rows_local.sort()
        split_rows = [y0 + r for r in split_rows_local]

        boundaries = [y0] + split_rows + [y1]
        result = []
        for i in range(len(boundaries) - 1):
            a = max(0, boundaries[i] - padding)
            b = min(bw_img.shape[0], boundaries[i + 1] + padding)
            if (b - a) >= min_line_height:
                result.append((a, b))
        return result if result else [(y0, y1)]

    # ── Search over threshold fractions and smoothing levels ─────────────────
    # Fine-grained fractions (including very small ones) are needed when
    # inter-line gaps have low but non-zero ink density (ascenders/descenders).
    # Smaller smoothing (3 vs old 7) preserves narrow inter-line valleys.
    best_bands: list[tuple[int, int]] = []
    best_delta = 10 ** 9

    fracs   = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]
    smooths = [3, 5]

    for smooth in smooths:
        for frac in fracs:
            for src in [bw_closed, bw]:
                bands = _find_bands(src, frac, smooth)
                if not bands:
                    continue
                if expected_lines > 0:
                    delta = abs(len(bands) - expected_lines)
                    if delta < best_delta:
                        best_delta, best_bands = delta, bands
                    if delta == 0:
                        return bands   # perfect match — stop early
                else:
                    return bands       # no target — first valid result is fine

    # ── Active band-splitting recovery ───────────────────────────────────────
    # If we're still short (typically by 1), split the tallest bands at their
    # weakest internal row — these are almost certainly two merged lines.
    if expected_lines > 0 and best_bands and best_delta > 0:
        deficit = expected_lines - len(best_bands)   # positive → need more bands
        if 0 < deficit <= 3:
            sorted_by_height = sorted(
                range(len(best_bands)),
                key=lambda i: best_bands[i][1] - best_bands[i][0],
                reverse=True,
            )
            candidate_bands = list(best_bands)
            splits_done = 0
            for orig_idx in sorted_by_height:
                if splits_done >= deficit:
                    break
                y0, y1 = candidate_bands[orig_idx]
                avg_h = sum(b[1] - b[0] for b in candidate_bands) / len(candidate_bands)
                if (y1 - y0) < avg_h * 1.3:
                    continue
                new_sub = _split_band_at_valley(bw, y0, y1, n_splits=1)
                if len(new_sub) > 1:
                    candidate_bands[orig_idx : orig_idx + 1] = new_sub
                    splits_done += 1

            if abs(len(candidate_bands) - expected_lines) < best_delta:
                best_bands = candidate_bands
            if len(candidate_bands) == expected_lines:
                return candidate_bands

    bands = best_bands if best_bands else [(0, gray.shape[0])]
    return bands


# ── CNN backbone (HEIGHT-ONLY pooling) ───────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + res, inplace=True)


class CNNFE(nn.Module):
    """
    Feature extractor for a SINGLE TEXT LINE.

    Input  : [B, 1, LINE_H, W]   (LINE_H = 32, W varies ~520)
    Output : [B, 256, 1, W]      (height fully collapsed, width preserved)

    Pooling schedule (height × width):
        MaxPool (2,1): 32→16
        MaxPool (2,1): 16→8
        MaxPool (2,1):  8→4
        MaxPool (2,1):  4→2
        Conv    (2,1):  2→1   ← final height collapse
    Width is NEVER pooled, so T = W ≈ 520 for CTC.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d((2, 1)),           # H: 32→16

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.MaxPool2d((2, 1)),           # H: 16→8

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            nn.MaxPool2d((2, 1)),           # H: 8→4

            # Block 4
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),           # H: 4→2

            # Block 5 — final height collapse
            nn.Conv2d(256, 256, (2, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # After: [B, 256, 1, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        assert out.shape[2] == 1, (
            f"CNN height should be 1 after pooling, got {out.shape[2]}. "
            f"Input height was {x.shape[2]} — must be exactly {LINE_H}."
        )
        return out


# ── Per-line CRNN ─────────────────────────────────────────────────────────────
class CRNN(nn.Module):
    """
    Single-line CRNN for CTC training.

    Input  : [B, 1, LINE_H, W]   (line crops, H=32)
    Output : [T, B, vocab+1]     log-probabilities for CTC  (T = W)

    Parameters
    ----------
    vocab_size  : number of character classes (excluding blank)
    lstm_hidden : hidden size per LSTM direction
    lstm_layers : number of stacked BiLSTM layers
    dropout     : applied between LSTM layers and before the head
    """

    def __init__(
        self,
        vocab_size: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.cnn     = CNNFE()
        self.lstm    = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head    = nn.Linear(lstm_hidden * 2, vocab_size + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, LINE_H, W]
        Returns log_probs : [T, B, vocab+1]   T = W (time-major for CTC)
        """
        feat = self.cnn(x)                          # [B, 256, 1, W]
        seq  = feat.squeeze(2).permute(0, 2, 1)     # [B, W, 256]
        seq  = self.dropout(seq)
        out, _ = self.lstm(seq)                     # [B, W, 2*hidden]
        logits = self.head(out)                     # [B, W, vocab+1]
        lp     = F.log_softmax(logits, dim=2)
        return lp.permute(1, 0, 2)                  # [T, B, vocab+1]


# ── Multi-line wrapper (inference only) ───────────────────────────────────────
class MultiLineCRNN(nn.Module):
    """
    Wraps CRNN with line segmentation for full-image inference.

    During TRAINING, use the bare CRNN on pre-cropped line images
    (see dataset.py — BookScanDataset crops lines and stores them as samples).

    During INFERENCE on a raw full-image scan, call forward_image().
    """

    def __init__(self, crnn: "CRNN", idx2char: dict):
        super().__init__()
        self.crnn    = crnn
        self.idx2char = idx2char

    @torch.no_grad()
    def forward_image(
        self,
        gray: np.ndarray,       # H×W uint8 grayscale
        device: torch.device,
        max_line_w: int = 2048,
    ) -> str:
        """
        Segment lines, run CRNN on each, return joined text.
        """
        bands = segment_lines(gray)
        line_texts = []

        for y0, y1 in bands:
            crop = gray[y0:y1, :]                   # H_line × W
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                continue

            # Resize crop to LINE_H, preserve width
            w_orig, h_orig = crop.shape[1], crop.shape[0]
            new_w = min(int(w_orig * LINE_H / h_orig), max_line_w)
            pil   = Image.fromarray(crop).resize((new_w, LINE_H), Image.LANCZOS)
            arr   = np.array(pil, dtype=np.float32) / 255.0
            t     = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            # t: [1, 1, LINE_H, new_w]

            lp    = self.crnn(t)                    # [T, 1, V+1]
            text  = ctc_greedy_decode(lp, self.idx2char)[0]
            line_texts.append(text)

        return "\n".join(line_texts)


# ── CTC greedy decoder ────────────────────────────────────────────────────────
def ctc_greedy_decode(
    log_probs: torch.Tensor,        # [T, B, V+1]
    idx2char: dict,
    blank_idx: int = 0,
) -> list[str]:
    """Return list of decoded strings for the batch."""
    preds = log_probs.argmax(dim=2).permute(1, 0)   # [B, T]
    results = []
    for seq in preds.cpu().tolist():
        chars = []
        prev  = blank_idx
        for idx in seq:
            if idx != blank_idx and idx != prev:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results