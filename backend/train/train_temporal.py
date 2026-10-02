"""
train_temporal.py
=================
Trains a bidirectional LSTM on per-frame 16-d temporal feature sequences
for deepfake temporal-consistency detection.

Input : dataset/processed/temporal_features.csv
         Each row: stem, label, features_json  (list-of-lists T×16)
Output: models/temporal_model.pth

Design:
  - BiLSTM(input=16, hidden=64, layers=2) + global-max-pool + MLP head
  - Variable-length sequences → padded & packed
  - AMP + gradient clipping
  - Weighted random sampling for class balance

Usage:
    python train_temporal.py --data_dir   /path/to/processed
                             --models_dir /path/to/models
                             --epochs 50
                             --batch  64
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_temporal")

N_FEAT  = 16     # feature dims per frame
MIN_SEQ = 4      # skip sequences shorter than this


# ─────────────────────────────── Dataset ───────────────────────────────────

class TemporalDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.sequences = []   # list of (tensor[T,16], label)

        for row in rows:
            try:
                feat_list = json.loads(row["features_json"])
            except Exception:
                continue
            if not feat_list or len(feat_list) < MIN_SEQ:
                continue

            arr = np.array(feat_list, dtype=np.float32)
            if arr.shape[1] != N_FEAT:
                continue

            # Normalise each feature dim independently
            mean = arr.mean(axis=0, keepdims=True)
            std  = arr.std(axis=0, keepdims=True) + 1e-8
            arr  = (arr - mean) / std

            label = int(row["label"])
            self.sequences.append(
                (torch.tensor(arr, dtype=torch.float32), label)
            )

        log.info("TemporalDataset: %d sequences loaded", len(self.sequences))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]
        return seq, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        return [s[1] for s in self.sequences]


def _collate(batch):
    """Pad variable-length sequences and track lengths for packing."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)   # (B, T_max, 16)
    labels  = torch.stack(labels)
    return padded, lengths, labels


# ─────────────────────────────── Model ─────────────────────────────────────

class TemporalBiLSTM(nn.Module):
    """
    BiLSTM with global max-pool over time + 2-layer MLP head.
    Handles variable-length sequences via pack_padded_sequence.
    """
    def __init__(self, n_feat: int = 16, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 16)
        # Sort by length descending (required by pack)
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]

        packed = pack_padded_sequence(x_sorted, sorted_lengths.cpu(),
                                      batch_first=True, enforce_sorted=True)
        out_packed, _ = self.lstm(packed)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
        # (B, T, 2H) → global max-pool
        pooled = out_padded.max(dim=1).values      # (B, 2H)

        # Unsort
        _, unsort_idx = sort_idx.sort()
        pooled = pooled[unsort_idx]

        return self.head(pooled).squeeze(1)


# ─────────────────────────────── Train loop ────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, scaler, device, train: bool):
    model.train(train)
    total_loss, preds_all, labels_all = 0.0, [], []
    ctx = torch.enable_grad if train else torch.no_grad

    with ctx():
        for seqs, lengths, labels in loader:
            seqs    = seqs.to(device)
            lengths = lengths.to(device)
            labels  = labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                out  = model(seqs, lengths)
                loss = criterion(out, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(labels)
            preds_all.extend(out.detach().cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(labels_all, [p > 0.5 for p in preds_all])
    try:
        auc = roc_auc_score(labels_all, preds_all)
    except Exception:
        auc = 0.5
    return avg_loss, acc, auc


# ─────────────────────────────── Data split ────────────────────────────────

def _load_and_split(data_dir: Path) -> tuple[list, list]:
    feat_file  = data_dir / "temporal_features.csv"
    split_file = data_dir / "split_manifest.csv"

    if not feat_file.exists():
        sys.exit(f"temporal_features.csv not found. Run prepare_dataset.py first.")

    # Read split manifest
    train_stems, val_stems = set(), set()
    with open(split_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train_stems.add(row["stem"])
            elif row["split"] == "val":
                val_stems.add(row["stem"])

    train_rows, val_rows = [], []
    with open(feat_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["stem"] in train_stems:
                train_rows.append(row)
            elif row["stem"] in val_stems:
                val_rows.append(row)

    # Fallback random split if stem matching fails
    if not train_rows or not val_rows:
        log.warning("Stem-split mismatch; using random 85/15 split.")
        all_rows = train_rows + val_rows
        random.shuffle(all_rows)
        n = int(len(all_rows) * 0.85)
        train_rows, val_rows = all_rows[:n], all_rows[n:]

    return train_rows, val_rows


# ─────────────────────────────── Main ──────────────────────────────────────

def main(args):
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    data_dir   = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = _load_and_split(data_dir)
    log.info("Rows → train=%d  val=%d", len(train_rows), len(val_rows))

    train_ds = TemporalDataset(train_rows)
    val_ds   = TemporalDataset(val_rows)

    if len(train_ds) == 0:
        sys.exit("No valid training samples. Check temporal_features.csv.")

    # Weighted sampler
    lbl = train_ds.get_labels()
    n1  = sum(lbl) or 1
    n0  = len(lbl) - n1 or 1
    w   = [1.0/n1 if l == 1 else 1.0/n0 for l in lbl]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              collate_fn=_collate, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              collate_fn=_collate, num_workers=0)

    model     = TemporalBiLSTM(n_feat=N_FEAT, hidden=64, n_layers=2).to(device)
    log.info("Params: {:,}".format(sum(p.numel() for p in model.parameters())))

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 5,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
        pct_start=0.1,
    )
    scaler    = GradScaler(enabled=device.type == "cuda")

    best_auc, no_improve = 0.0, 0
    ckpt_path = models_dir / "temporal_model.pth"

    log.info("─" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(
            model, train_loader, optimizer, criterion, scaler, device, True)
        va_loss, va_acc, va_auc = run_epoch(
            model, val_loader, optimizer, criterion, scaler, device, False)

        log.info("Ep %02d/%02d | TR l=%.4f acc=%.3f auc=%.3f | "
                 "VA l=%.4f acc=%.3f auc=%.3f | %.1fs",
                 epoch, args.epochs,
                 tr_loss, tr_acc, tr_auc,
                 va_loss, va_acc, va_auc, time.time() - t0)

        if va_auc > best_auc + 1e-4:
            best_auc, no_improve = va_auc, 0
            torch.save(model.state_dict(), ckpt_path)
            log.info("  ✓ Best AUC=%.4f → %s", best_auc, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info("Early stop at epoch %d.", epoch)
                break

    log.info("Done. Best AUC=%.4f  Model: %s", best_auc, ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch",      type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--patience",   type=int,   default=8)
    main(ap.parse_args())
