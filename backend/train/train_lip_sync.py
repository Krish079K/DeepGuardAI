"""
train_lip_sync.py
=================
Trains a lip-sync discriminator that learns to score
the coherence between mouth motion and audio energy.

Architecture:
  Input: (mouth_motion_signal [T], audio_energy_signal [T])
         concatenated → shape (T, 2)
  Model: 1-layer BiLSTM → pooling → Linear → sigmoid
  Label: 0 = real (motion ↔ audio are in sync)
          1 = fake (motion ↔ audio are out of sync)

Because we don't need separate sync-labelled data, we generate
training pairs from the existing real/fake videos directly:
  - Real video pairs      → label 0
  - Fake video pairs      → label 1
  - Real-video cross-pairs (shuffled audio from different real video) → label 1
    (acts as hard negatives to teach the model what "out of sync" looks like)

Input : dataset/processed/split_manifest.csv
         (original videos in dataset_dir/real/ and dataset_dir/fake/)
Output: models/lip_sync_model.pth

Usage:
    python train_lip_sync.py
           --dataset_dir  /path/to/dataset       # folder with real/ fake/
           --data_dir     /path/to/processed
           --models_dir   /path/to/models
           --epochs 40
           --batch  32
"""

import argparse
import csv
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import subprocess
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_lip_sync")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp"}
SEQ_LEN = 64   # fixed sequence length (frames)


# ─────────────────────────────── Signal extractors ─────────────────────────

def _mouth_motion(video_path: str, seq_len: int = SEQ_LEN) -> np.ndarray | None:
    """Return normalised mouth-motion magnitude series [seq_len]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 4:
        cap.release()
        return None

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    indices = np.linspace(0, total - 1, num=seq_len, dtype=int)
    motion = []
    prev_mouth = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            motion.append(0.0)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            my1 = y + int(h * 0.60)
            mouth = gray[my1:y+h, x + int(w*0.1):x + w - int(w*0.1)]
            if mouth.size > 0:
                mouth = cv2.resize(mouth, (32, 16))
                if prev_mouth is not None and prev_mouth.shape == mouth.shape:
                    diff = np.abs(mouth.astype(np.float32) - prev_mouth.astype(np.float32))
                    motion.append(float(diff.mean()))
                else:
                    motion.append(0.0)
                prev_mouth = mouth
            else:
                motion.append(0.0)
        else:
            motion.append(0.0)
            prev_mouth = None

    cap.release()

    arr = np.array(motion, dtype=np.float32)
    mx = arr.max()
    if mx > 1e-6:
        arr /= mx
    return arr


def _audio_energy(video_path: str, seq_len: int = SEQ_LEN,
                   sr: int = 16000) -> np.ndarray | None:
    """Return normalised audio RMS energy series [seq_len]."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
             tmp_wav, "-loglevel", "error"],
            capture_output=True, timeout=120
        )
        if r.returncode != 0 or not os.path.exists(tmp_wav):
            return np.zeros(seq_len, dtype=np.float32)

        y, _ = librosa.load(tmp_wav, sr=sr, mono=True)
        hop = max(1, len(y) // seq_len)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        x_old = np.linspace(0, 1, len(rms))
        x_new = np.linspace(0, 1, seq_len)
        envelope = np.interp(x_new, x_old, rms).astype(np.float32)
        mx = envelope.max()
        if mx > 1e-6:
            envelope /= mx
        return envelope
    except Exception:
        return np.zeros(seq_len, dtype=np.float32)
    finally:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass


# ─────────────────────────────── Dataset ───────────────────────────────────

class LipSyncDataset(Dataset):
    """
    Each sample: (motion_signal, audio_signal, label)
    Positives (label=1, fake):
      - Videos from fake/ folder
      - Cross-pairs: motion from real A, audio from real B (A≠B)
    Negatives (label=0, real):
      - Videos from real/ folder
    """

    def __init__(self, video_pairs: list[tuple[str, str, int]]):
        # pairs: (video_path_for_motion, video_path_for_audio, label)
        self.pairs = video_pairs
        self.cache: dict[str, np.ndarray | None] = {}

    def _get_motion(self, path: str) -> np.ndarray:
        if path not in self.cache:
            self.cache[path] = _mouth_motion(path)
        m = self.cache[path]
        return m if m is not None else np.zeros(SEQ_LEN, dtype=np.float32)

    def _get_audio(self, path: str) -> np.ndarray:
        key = path + "__audio"
        if key not in self.cache:
            self.cache[key] = _audio_energy(path)
        a = self.cache[key]
        return a if a is not None else np.zeros(SEQ_LEN, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        motion_path, audio_path, label = self.pairs[idx]
        motion = self._get_motion(motion_path)
        audio  = self._get_audio(audio_path)
        # Stack → (SEQ_LEN, 2)
        x = np.stack([motion, audio], axis=1).astype(np.float32)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


def _build_pairs(dataset_dir: Path, split_file: Path,
                 split: str, max_per_class: int = 300
                 ) -> list[tuple[str, str, int]]:
    """Build (motion_vid, audio_vid, label) pairs for a given split."""
    stems_in_split = set()
    with open(split_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                stems_in_split.add((row["stem"], int(row["label"])))

    def find_video(stem: str, label: int) -> Path | None:
        sub = "real" if label == 0 else "fake"
        # stem format: real_<original_stem> or fake_<original_stem>
        orig_stem = stem[len(sub)+1:]   # strip "real_" or "fake_"
        folder = dataset_dir / sub
        for ext in VIDEO_EXTS:
            p = folder / (orig_stem + ext)
            if p.exists():
                return p
        # try glob
        matches = list(folder.glob(orig_stem + ".*"))
        return matches[0] if matches else None

    real_paths, fake_paths = [], []
    for stem, label in stems_in_split:
        p = find_video(stem, label)
        if p and p.exists():
            (real_paths if label == 0 else fake_paths).append(str(p))

    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    real_paths = real_paths[:max_per_class]
    fake_paths = fake_paths[:max_per_class]

    pairs = []
    # Real pairs (label=0) - motion and audio from same video
    for p in real_paths:
        pairs.append((p, p, 0))

    # Fake pairs (label=1) - from fake videos
    for p in fake_paths:
        pairs.append((p, p, 1))

    # Hard negatives: cross real pairs (label=1, out-of-sync)
    n_cross = min(len(real_paths), max_per_class // 2)
    shuffled_real = real_paths[:n_cross].copy()
    random.shuffle(shuffled_real)
    for i, p in enumerate(real_paths[:n_cross]):
        audio_p = shuffled_real[(i + 1) % len(shuffled_real)]
        if p != audio_p:
            pairs.append((p, audio_p, 1))

    random.shuffle(pairs)
    log.info("[%s] built %d pairs (real=%d fake=%d cross=%d)",
             split, len(pairs), len(real_paths), len(fake_paths), n_cross)
    return pairs


# ─────────────────────────────── Model ─────────────────────────────────────

class LipSyncLSTM(nn.Module):
    """
    BiLSTM over (motion, audio) 2-channel time series.
    Input: (batch, SEQ_LEN, 2)
    Output: (batch,) fake probability
    """
    def __init__(self, input_size: int = 2, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.3 if layers > 1 else 0.0)
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                    # (B, T, 2H)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context = (out * attn_w).sum(dim=1)      # (B, 2H)
        return self.head(context).squeeze(1)


# ─────────────────────────────── Train loop ────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, scaler, device, train: bool):
    model.train(train)
    total_loss, preds_all, labels_all = 0.0, [], []
    ctx = torch.enable_grad if train else torch.no_grad

    with ctx():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out  = model(X)
                loss = criterion(out, y)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(y)
            preds_all.extend(out.detach().cpu().numpy().tolist())
            labels_all.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(labels_all, [p > 0.5 for p in preds_all])
    try:
        auc = roc_auc_score(labels_all, preds_all)
    except Exception:
        auc = 0.5
    return avg_loss, acc, auc


# ─────────────────────────────── Main ──────────────────────────────────────

def main(args):
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    dataset_dir = Path(args.dataset_dir)
    data_dir    = Path(args.data_dir)
    models_dir  = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    split_file = data_dir / "split_manifest.csv"
    if not split_file.exists():
        sys.exit(f"split_manifest.csv not found. Run prepare_dataset.py first.")

    train_pairs = _build_pairs(dataset_dir, split_file, "train",
                                max_per_class=args.max_per_class)
    val_pairs   = _build_pairs(dataset_dir, split_file, "val",
                                max_per_class=args.max_per_class // 3)

    train_ds = LipSyncDataset(train_pairs)
    val_ds   = LipSyncDataset(val_pairs)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0)

    model = LipSyncLSTM(input_size=2, hidden=64, layers=2).to(device)
    log.info("Params: {:,}".format(sum(p.numel() for p in model.parameters())))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler(enabled=device.type == "cuda")

    best_auc, no_improve = 0.0, 0
    ckpt_path = models_dir / "lip_sync_model.pth"

    log.info("─" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(
            model, train_loader, optimizer, criterion, scaler, device, True)
        va_loss, va_acc, va_auc = run_epoch(
            model, val_loader,   optimizer, criterion, scaler, device, False)
        scheduler.step()

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
                log.info("Early stop.")
                break

    log.info("Done. Best AUC=%.4f  Model: %s", best_auc, ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",    required=True,
                    help="Root folder with real/ and fake/ video sub-folders")
    ap.add_argument("--data_dir",       required=True,
                    help="Processed folder (split_manifest.csv location)")
    ap.add_argument("--models_dir",     required=True)
    ap.add_argument("--epochs",         type=int,   default=40)
    ap.add_argument("--batch",          type=int,   default=32)
    ap.add_argument("--lr",             type=float, default=5e-4)
    ap.add_argument("--patience",       type=int,   default=7)
    ap.add_argument("--max_per_class",  type=int,   default=300,
                    help="Max videos per class for pair construction")
    main(ap.parse_args())
