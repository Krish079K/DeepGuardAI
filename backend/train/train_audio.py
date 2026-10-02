"""
train_audio.py
==============
Trains a lightweight MLP on 128-d MFCC+spectral features for
synthetic-speech / deepfake audio detection.

Input : dataset/processed/audio_features.csv
Output: models/audio_model.pth

Architecture: MLP  128 → 256 → 128 → 64 → 1
Fast to train; runs fully on CPU if needed, GPU if available.

Usage:
    python train_audio.py --data_dir   /path/to/dataset/processed
                          --models_dir /path/to/models
                          --epochs 60
                          --batch  128
                          --lr     1e-3
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_audio")


# ─────────────────────────────── Dataset ───────────────────────────────────

class AudioFeatureDataset(Dataset):
    """
    Reads audio_features.csv.
    Columns: stem, label, f0, f1, ..., f127
    """
    def __init__(self, rows: list[dict], scaler: StandardScaler | None = None,
                 fit_scaler: bool = False):
        self.features = []
        self.labels   = []

        for row in rows:
            feat = np.array([float(row[f"f{i}"]) for i in range(128)],
                            dtype=np.float32)
            self.features.append(feat)
            self.labels.append(int(row["label"]))

        X = np.stack(self.features)
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            X = scaler.transform(X).astype(np.float32)
        else:
            self.scaler = None

        self.features = [X[i] for i in range(len(X))]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx],   dtype=torch.float32))


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _split_rows(rows: list[dict], split_file: Path) -> tuple[list, list]:
    """Return (train_rows, val_rows) filtered by split_manifest.csv."""
    train_stems, val_stems = set(), set()
    with open(split_file, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in ("train",):
                train_stems.add(r["stem"])
            elif r["split"] == "val":
                val_stems.add(r["stem"])

    train = [r for r in rows if r["stem"] in train_stems]
    val   = [r for r in rows if r["stem"] in val_stems]
    # If stems don't match (e.g. audio not extracted for all videos),
    # fall back to random 85/15 split
    if not train or not val:
        log.warning("Stem-based split produced empty set; falling back to random split.")
        idx = int(len(rows) * 0.85)
        np.random.shuffle(rows)
        train, val = rows[:idx], rows[idx:]
    return train, val


# ─────────────────────────────── Model ─────────────────────────────────────

class AudioMLP(nn.Module):
    def __init__(self, n_in: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ─────────────────────────────── Train loop ────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, scaler, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(X)
                loss   = criterion(logits, y)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(y)
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, [p > 0.5 for p in all_preds])
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
    return avg_loss, acc, auc


# ─────────────────────────────── Main ──────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    data_dir   = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    feat_file  = data_dir / "audio_features.csv"
    split_file = data_dir / "split_manifest.csv"

    if not feat_file.exists():
        raise FileNotFoundError(f"audio_features.csv not found in {data_dir}. "
                                "Run prepare_dataset.py first.")

    all_rows = _load_csv(feat_file)
    log.info("Total audio samples: %d", len(all_rows))

    train_rows, val_rows = _split_rows(all_rows, split_file)
    log.info("Train: %d  Val: %d", len(train_rows), len(val_rows))

    # Fit scaler on train only
    train_ds = AudioFeatureDataset(train_rows, fit_scaler=True)
    val_ds   = AudioFeatureDataset(val_rows, scaler=train_ds.scaler)

    # Save scaler for inference time
    scaler_path = models_dir / "audio_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(train_ds.scaler, f)
    log.info("Scaler saved → %s", scaler_path)

    # Weighted sampler for class balance
    labels   = train_ds.labels
    n_fake   = sum(labels)
    n_real   = len(labels) - n_fake
    weights  = [1.0 / n_fake if l == 1 else 1.0 / n_real for l in labels]
    sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0)

    model     = AudioMLP(n_in=128).to(device)
    log.info("Params: {:,}".format(sum(p.numel() for p in model.parameters())))

    pos_w    = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_auc   = 0.0
    patience   = args.patience
    no_improve = 0
    ckpt_path  = models_dir / "audio_model.pth"

    log.info("─" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(
            model, train_loader, optimizer, criterion, scaler, device, train=True)
        va_loss, va_acc, va_auc = run_epoch(
            model, val_loader, optimizer, criterion, scaler, device, train=False)
        scheduler.step(va_auc)

        log.info("Ep %02d/%02d | TR loss=%.4f acc=%.3f auc=%.3f | "
                 "VA loss=%.4f acc=%.3f auc=%.3f | %.1fs",
                 epoch, args.epochs,
                 tr_loss, tr_acc, tr_auc,
                 va_loss, va_acc, va_auc,
                 time.time() - t0)

        if va_auc > best_auc + 1e-4:
            best_auc = va_auc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            log.info("  ✓ Best AUC=%.4f — saved → %s", best_auc, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stop at epoch %d.", epoch)
                break

    log.info("Done. Best val AUC: %.4f  Model: %s", best_auc, ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train audio deepfake MLP")
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--epochs",     type=int,   default=60)
    ap.add_argument("--batch",      type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--patience",   type=int,   default=8)
    ap.add_argument("--pos_weight", type=float, default=1.0)
    main(ap.parse_args())
