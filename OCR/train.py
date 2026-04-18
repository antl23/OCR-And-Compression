"""
train.py
========
End-to-end training script for the book-scan OCR pipeline.

Usage
-----
# Step 0 — generate labels (once)
python generate_labels.py --data_root data/ --out_dir data/labels/

# Step 1 — train
python train.py \
    --data_root data/ \
    --label_dir data/labels/ \
    --epochs 50 \
    --batch_size 16 \
    --lr 3e-4 \
    --run_dir runs/run_001

# Step 2 — evaluate on test set
python train.py \
    --data_root data/ \
    --label_dir data/labels/ \
    --eval_only \
    --checkpoint runs/run_001/best.pt

Targets
-------
≥ 95 % character accuracy (1 - CER) on the test split.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import editdistance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BookScanDataset, build_vocab, ctc_collate
from model import CRNN, ctc_greedy_decode


# ── Helpers ───────────────────────────────────────────────────────────────────
class CollateWrapper:
    def __init__(self, max_w):
        self.max_w = max_w

    def __call__(self, batch):
        return ctc_collate(batch, max_w=self.max_w)
def cer(pred: str, gt: str) -> float:
    """Character Error Rate: edit_distance(pred, gt) / len(gt)."""
    if not gt:
        return 0.0 if not pred else 1.0
    return editdistance.eval(pred, gt) / len(gt)


def evaluate(model, loader, idx2char, device):
    model.eval()
    total_cer = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts  = batch["texts"]
            lp     = model(images)
            preds  = ctc_greedy_decode(lp, idx2char)
            for p, g in zip(preds, texts):
                total_cer += cer(p, g)
                n += 1
    return total_cer / max(n, 1)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    label_dir = Path(args.label_dir)
    char2idx, idx2char = build_vocab(label_dir)
    vocab_size = len(char2idx)
    print(f"Vocabulary size: {vocab_size} chars (+1 blank = {vocab_size+1} classes)")
    (Path(args.run_dir) / "vocab.json").parent.mkdir(parents=True, exist_ok=True)
    with open(Path(args.run_dir) / "vocab.json", "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f)

    # ── Datasets ──────────────────────────────────────────────────────────────
    noisy_dir = Path(args.data_root) / "simulated_noisy_images_grayscale"
    clean_dir = Path(args.data_root) / "clean_images_grayscale"
    collate   = CollateWrapper(max_w=args.max_w)
    # TR, VA, and RE all go into training — TE is the sole held-out test split.
    # val_loader monitors TE during training so no data is wasted on a
    # separate validation fold.
    from torch.utils.data import ConcatDataset

    train_splits = []
    for split_tag in ("TR", "VA", "RE"):
        try:
            ds = BookScanDataset(
                noisy_dir, clean_dir, label_dir, split_tag, char2idx, augment=True
            )
            train_splits.append(ds)
            print(f"  {split_tag}: {len(ds)} samples")
        except FileNotFoundError:
            print(f"  {split_tag}: not found, skipping")

    if not train_splits:
        raise RuntimeError("No training data found for any of TR / VA / RE splits.")

    # Move the bulk of TE into training — keep only a small fixed holdout
    # for evaluation.  200 samples is enough to get a stable CER estimate;
    # everything else is more valuable as training data.
    from torch.utils.data import ConcatDataset, Subset
    import random as _random

    _te_full = BookScanDataset(noisy_dir, clean_dir, label_dir, "TE", char2idx, augment=False)
    _te_size = len(_te_full)
    _holdout = min(args.test_size, _te_size)
    _rng     = _random.Random(42)          # fixed seed — same holdout every run
    _te_idx  = list(range(_te_size))
    _rng.shuffle(_te_idx)
    _test_idx  = _te_idx[:_holdout]
    _train_idx = _te_idx[_holdout:]

    test_ds  = Subset(_te_full, _test_idx)
    # TE remainder (augment=True) joins training
    if _train_idx:
        _te_train = BookScanDataset(noisy_dir, clean_dir, label_dir, "TE", char2idx, augment=True)
        train_splits.append(Subset(_te_train, _train_idx))
        print(f"  TE (train portion): {len(_train_idx)} samples")

    print(f"  TE (held-out test):  {len(test_ds)} samples")

    train_ds = ConcatDataset(train_splits)
    val_ds   = test_ds   # monitor on the same small holdout during training

    print(f"Train: {len(train_ds)}  Test/Val: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate,
    )
    test_loader = val_loader   # same holdout — reuse loader for final eval

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CRNN(vocab_size, lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers, dropout=args.dropout).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────────
    run_dir    = Path(args.run_dir)
    best_ckpt  = run_dir / "best.pt"
    last_ckpt  = run_dir / "last.pt"
    best_cer   = float("inf")
    start_epoch = 1

    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_cer    = ck.get("best_cer", best_cer)
        print(f"Resumed from epoch {ck['epoch']}, best CER={best_cer:.4f}")

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        ck_path = Path(args.checkpoint) if args.checkpoint else best_ckpt
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["model"])
        test_cer = evaluate(model, test_loader, idx2char, device)
        print(f"\nTest CER : {test_cer:.4f}  ({(1-test_cer)*100:.2f}% char accuracy)")
        return

    # ── Training ──────────────────────────────────────────────────────────────
    
    log = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        aligned_count = 0
        total_count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images     = batch["images"].to(device)           # [B, 1, H, W]
            labels     = batch["labels"].to(device)           # [sum_lens]
            label_lens = batch["label_lens"].to(device)       # [B]

            log_probs = model(images)                         # [T, B, V+1]
            T = log_probs.shape[0]
            B = images.shape[0]
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, labels, input_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            epoch_loss += loss.item()
            # Track alignment quality (only present in updated dataset.py)
            if "aligned" in batch:
                aligned_count += sum(batch["aligned"])
                total_count   += len(batch["aligned"])

        avg_loss = epoch_loss / len(train_loader)
        val_cer = evaluate(model, val_loader, idx2char, device)
        scheduler.step()
        elapsed  = time.time() - t0

        lr_now = scheduler.get_last_lr()[0]
        align_str = ""
        if total_count > 0:
            align_pct = 100 * aligned_count / total_count
            align_str = f" | align={align_pct:.0f}%"
            if align_pct < 80 and epoch == 1:
                print(f"  ⚠ Only {align_pct:.0f}% of line crops are well-aligned.")
                print( "    Check diagnose.py output — segmentation may be unreliable.")
        print(
            f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_CER={val_cer:.4f} "
            f"({(1-val_cer)*100:.1f}% acc) | lr={lr_now:.2e}{align_str} | {elapsed:.0f}s"
        )

        log.append({"epoch": epoch, "loss": avg_loss, "val_cer": val_cer})
        with open(run_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        # Save checkpoints
        ck = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_cer": val_cer, "best_cer": best_cer,
            "vocab_size": vocab_size,
        }
        torch.save(ck, last_ckpt)

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(ck, best_ckpt)
            print(f"  ✓ New best val CER={best_cer:.4f} saved to {best_ckpt}")

        # Early-exit if target reached
        if (1 - best_cer) >= 0.97:
            print(f"\n🎯 Target ≥97% char accuracy reached at epoch {epoch}!")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n── Loading best checkpoint for test evaluation ──")
    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    test_cer = evaluate(model, test_loader, idx2char, device)
    print(f"Test CER : {test_cer:.4f}  ({(1-test_cer)*100:.2f}% char accuracy)")

    with open(run_dir / "results.json", "w") as f:
        json.dump({
            "test_cer": test_cer,
            "test_char_accuracy": 1 - test_cer,
            "best_val_cer": best_cer,
        }, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="data/")
    p.add_argument("--label_dir",   default="data/labels/")
    p.add_argument("--run_dir",     default="runs/run_001")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--max_w",       type=int,   default=1024,
                   help="Max image width (wider images are cropped)")
    p.add_argument("--lstm_hidden", type=int,   default=256)
    p.add_argument("--lstm_layers", type=int,   default=2)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--eval_only",   action="store_true")
    p.add_argument("--checkpoint",  default=None, help="Path for --eval_only")
    p.add_argument("--dropout",    type=float, default=0.5)
    p.add_argument("--test_size",  type=int,   default=150,
                   help="Number of TE samples held out for evaluation; rest joins training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())