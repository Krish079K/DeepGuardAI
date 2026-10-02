"""
train_fusion.py
===============
Trains a fusion MLP that combines the 4 branch scores into a final
fake-probability prediction.

Strategy:
  - Run all 4 branch models over every video in the training set
    to collect (visual_score, audio_score, lip_sync_score, temporal_score, label)
  - Train a small MLP on these vectors
  - This is the final model that replaces the heuristic ensemble in fusion_model.py

Output: models/fusion_model.pth

Usage:
    python train_fusion.py
           --dataset_dir  /path/to/dataset         # real/ and fake/ videos
           --data_dir     /path/to/processed
           --models_dir   /path/to/models           # where branch models live
           --epochs 100
           --batch  64
           --lr     1e-3
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_fusion")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp"}
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────── Score collector ────────────────────────────

def _collect_scores(video_paths: list[Path], labels: list[int],
                    models_dir: Path) -> list[dict]:
    """
    Run all 4 branch models over each video and collect 4-score vectors.
    Skips videos that fail.
    """
    from visual_model   import analyze_visual
    from audio_model    import analyze_audio
    from lip_sync_model import analyze_lip_sync
    from temporal_model import analyze_temporal

    def _find_model(name: str) -> str | None:
        p = models_dir / name
        return str(p) if p.exists() else None

    vis_model  = _find_model("visual_model.pth")
    aud_model  = _find_model("audio_model.pth")
    lip_model  = _find_model("lip_sync_model.pth")
    tmp_model  = _find_model("temporal_model.pth")

    rows = []
    for i, (vpath, label) in enumerate(zip(video_paths, labels)):
        log.info("[%d/%d] %s", i+1, len(video_paths), vpath.name)
        try:
            vis = analyze_visual(str(vpath),   vis_model)
            aud = analyze_audio(str(vpath),    aud_model)
            lip = analyze_lip_sync(str(vpath), lip_model)
            tmp = analyze_temporal(str(vpath), tmp_model)
            rows.append({
                "visual_score":   vis.get("visual_score",   0.5),
                "audio_score":    aud.get("audio_score",    0.5),
                "lip_sync_score": lip.get("lip_sync_score", 0.5),
                "temporal_score": tmp.get("temporal_score", 0.5),
                "label":          label,
            })
        except Exception as exc:
            log.warning("  Skipped %s: %s", vpath.name, exc)

    return rows


# ─────────────────────────────── Dataset ───────────────────────────────────

class FusionDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.X = torch.tensor(
            [[r["visual_score"], r["audio_score"],
              r["lip_sync_score"], r["temporal_score"]] for r in rows],
            dtype=torch.float32,
        )
        self.y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_labels(self) -> list[int]:
        return self.y.long().tolist()


# ─────────────────────────────── Model ─────────────────────────────────────

class FusionMLP(nn.Module):
    """
    Input: 4-d score vector
    Output: fake probability
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


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

def _load_video_list(dataset_dir: Path, split_file: Path,
                     split: str) -> tuple[list[Path], list[int]]:
    """Return video paths and labels for a given split."""
    stem_label = {}
    with open(split_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                stem_label[row["stem"]] = int(row["label"])

    paths, labels = [], []
    for stem, label in stem_label.items():
        sub = "real" if label == 0 else "fake"
        orig_stem = stem[len(sub)+1:]
        folder = dataset_dir / sub
        for ext in VIDEO_EXTS:
            p = folder / (orig_stem + ext)
            if p.exists():
                paths.append(p)
                labels.append(label)
                break
        else:
            # Try glob
            matches = list(folder.glob(orig_stem + ".*"))
            if matches:
                paths.append(matches[0])
                labels.append(label)

    return paths, labels


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
        sys.exit("split_manifest.csv not found. Run prepare_dataset.py first.")

    # ── Collect branch scores ────────────────────────────────────────────
    cache_train = data_dir / "fusion_train_scores.json"
    cache_val   = data_dir / "fusion_val_scores.json"

    if args.use_cache and cache_train.exists() and cache_val.exists():
        log.info("Loading cached scores (skip re-inference).")
        with open(cache_train) as f:
            train_rows = json.load(f)
        with open(cache_val) as f:
            val_rows = json.load(f)
    else:
        log.info("Collecting branch scores for train set …")
        tr_paths, tr_labels = _load_video_list(dataset_dir, split_file, "train")
        train_rows = _collect_scores(tr_paths, tr_labels, models_dir)
        with open(cache_train, "w") as f:
            json.dump(train_rows, f)

        log.info("Collecting branch scores for val set …")
        va_paths, va_labels = _load_video_list(dataset_dir, split_file, "val")
        val_rows = _collect_scores(va_paths, va_labels, models_dir)
        with open(cache_val, "w") as f:
            json.dump(val_rows, f)

    log.info("Fusion train=%d  val=%d", len(train_rows), len(val_rows))

    if not train_rows:
        sys.exit("No training rows. Ensure branch models are trained and videos are accessible.")

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_ds = FusionDataset(train_rows)
    val_ds   = FusionDataset(val_rows) if val_rows else train_ds

    lbl = train_ds.get_labels()
    n1 = sum(lbl) or 1
    n0 = len(lbl) - n1 or 1
    w  = [1.0/n1 if l == 1 else 1.0/n0 for l in lbl]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    model = FusionMLP().to(device)
    log.info("FusionMLP params: {:,}".format(
        sum(p.numel() for p in model.parameters())))

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, verbose=True
    )
    scaler    = GradScaler(enabled=device.type == "cuda")

    best_auc, no_improve = 0.0, 0
    ckpt_path = models_dir / "fusion_model.pth"

    log.info("─" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(
            model, train_loader, optimizer, criterion, scaler, device, True)
        va_loss, va_acc, va_auc = run_epoch(
            model, val_loader, optimizer, criterion, scaler, device, False)
        scheduler.step(va_auc)

        log.info("Ep %03d/%03d | TR l=%.4f acc=%.3f auc=%.3f | "
                 "VA l=%.4f acc=%.3f auc=%.3f | %.2fs",
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

    # ── Final evaluation on val set ──────────────────────────────────────
    if val_rows:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        X_val = torch.tensor(
            [[r["visual_score"], r["audio_score"],
              r["lip_sync_score"], r["temporal_score"]] for r in val_rows],
            dtype=torch.float32,
        ).to(device)
        with torch.no_grad():
            preds = model(X_val).cpu().numpy()
        true  = [r["label"] for r in val_rows]
        log.info("\n%s", classification_report(
            true, [p > 0.5 for p in preds], target_names=["real", "fake"]
        ))
        log.info("Final val AUC: %.4f", roc_auc_score(true, preds))

    log.info("Done. Model saved: %s", ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",  required=True,
                    help="Folder with real/ and fake/ video sub-dirs")
    ap.add_argument("--data_dir",     required=True)
    ap.add_argument("--models_dir",   required=True)
    ap.add_argument("--epochs",       type=int,   default=100)
    ap.add_argument("--batch",        type=int,   default=64)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--patience",     type=int,   default=12)
    ap.add_argument("--use_cache",    action="store_true",
                    help="Re-use cached branch scores (skip re-inference)")
    main(ap.parse_args())
