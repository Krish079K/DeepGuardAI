"""
train_visual.py
===============
Fine-tunes MobileNetV3-Small (pretrained on ImageNet) for deepfake
face-crop classification.

Input : dataset/processed/frame_manifest.csv
         dataset/processed/frames/<real|fake>/<stem>/frame_XXXX.png
Output: models/visual_model.pth

Design choices (RTX 3050 4 GB):
  - MobileNetV3-Small  ~1.5 M params → fits in 4 GB even at batch 64
  - Mixed-precision (AMP) fp16 forward / fp32 master weights
  - Cosine-annealing LR with warm restarts
  - Extensive augmentation to prevent overfit on small datasets
  - Early stopping (patience=5)

Usage:
    python train_visual.py --data_dir  /path/to/dataset/processed
                           --models_dir /path/to/models
                           --epochs 30
                           --batch  64
                           --lr     3e-4
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_visual")


# ─────────────────────────────── Dataset ───────────────────────────────────

class FaceFrameDataset(Dataset):
    """
    Loads face-crop PNG files listed in frame_manifest.csv.
    Each row: path (relative to data_dir), label (0=real, 1=fake).
    """

    TRAIN_TF = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])

    VAL_TF = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_dir: Path, stems: set[str], is_train: bool):
        self.data_dir = data_dir
        self.is_train = is_train
        self.tf = self.TRAIN_TF if is_train else self.VAL_TF
        self.samples = []   # (abs_path, label)

        manifest = data_dir / "frame_manifest.csv"
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel_path = Path(row["path"])
                label    = int(row["label"])
                # Derive stem: frames/<class>/<STEM>/frame_XXXX.png → <STEM>
                stem = rel_path.parts[2] if len(rel_path.parts) >= 3 else ""
                if stem in stems or stems is None:
                    abs_p = data_dir / rel_path
                    if abs_p.exists():
                        self.samples.append((abs_p, label))

        if not self.samples:
            raise RuntimeError(f"No samples found. Check manifest and stems.")
        log.info("FaceFrameDataset [%s] → %d samples",
                 "train" if is_train else "val", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.tf(img)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        return [s[1] for s in self.samples]


def _make_weighted_sampler(dataset: FaceFrameDataset) -> WeightedRandomSampler:
    labels = dataset.get_labels()
    class_count = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_count[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────── Model ─────────────────────────────────────

def build_model(pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with a 2-layer classification head."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    in_features = net.classifier[3].in_features
    net.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 64),
        nn.Hardswish(),
        nn.Linear(64, 1),
    )
    return net


# ─────────────────────────────── Training ──────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, [p > 0.5 for p in all_preds])
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.0
    return avg_loss, acc, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, [p > 0.5 for p in all_preds])
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.0
    return avg_loss, acc, auc


# ─────────────────────────────── Main ──────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  |  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    data_dir  = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Read split manifest ───────────────────────────────────────────────
    split_file = data_dir / "split_manifest.csv"
    if not split_file.exists():
        sys.exit(f"[ERROR] split_manifest.csv not found in {data_dir}. "
                 "Run prepare_dataset.py first.")

    train_stems, val_stems = set(), set()
    with open(split_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] in ("train",):
                train_stems.add(row["stem"])
            elif row["split"] == "val":
                val_stems.add(row["stem"])

    log.info("Stems → train=%d  val=%d", len(train_stems), len(val_stems))

    train_ds = FaceFrameDataset(data_dir, train_stems, is_train=True)
    val_ds   = FaceFrameDataset(data_dir, val_stems,   is_train=False)

    sampler = _make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=sampler,
                              num_workers=args.workers,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=(device.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)
    log.info("Parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    # Freeze backbone for first 3 epochs (feature extraction phase)
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight]).to(device)
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_auc   = 0.0
    patience   = args.patience
    no_improve = 0
    ckpt_path  = models_dir / "visual_model.pth"

    log.info("─" * 60)
    log.info("Starting training  epochs=%d  batch=%d  lr=%.0e",
             args.epochs, args.batch, args.lr)
    log.info("─" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after epoch 3
        if epoch == 4:
            log.info("Unfreezing backbone for fine-tuning.")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, args.epochs - 3), eta_min=1e-7
            )

        tr_loss, tr_acc, tr_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)
        va_loss, va_acc, va_auc = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        log.info(
            "Epoch %02d/%02d | "
            "TR loss=%.4f acc=%.3f auc=%.3f | "
            "VA loss=%.4f acc=%.3f auc=%.3f | "
            "%.1fs",
            epoch, args.epochs,
            tr_loss, tr_acc, tr_auc,
            va_loss, va_acc, va_auc,
            elapsed,
        )

        if va_auc > best_auc + 1e-4:
            best_auc = va_auc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            log.info("  ✓ New best val AUC=%.4f — checkpoint saved → %s",
                     best_auc, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (patience=%d).",
                         epoch, patience)
                break

    log.info("Training complete. Best val AUC: %.4f", best_auc)
    log.info("Model saved: %s", ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train visual deepfake detector")
    ap.add_argument("--data_dir",   required=True,
                    help="Path to dataset/processed/ folder")
    ap.add_argument("--models_dir", required=True,
                    help="Where to save .pth files")
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--batch",      type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--workers",    type=int,   default=4)
    ap.add_argument("--patience",   type=int,   default=5,
                    help="Early-stopping patience (epochs)")
    ap.add_argument("--pos_weight", type=float, default=1.0,
                    help="BCEWithLogitsLoss pos_weight for class imbalance")
    main(ap.parse_args())
