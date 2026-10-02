"""
evaluate.py
===========
Comprehensive evaluation of all trained branch models + fusion model.
Computes:
  - Accuracy, AUC-ROC, F1, Precision, Recall
  - Confusion matrix
  - Per-threshold analysis
  - Combined score table

Usage:
    python evaluate.py --dataset_dir  /path/to/dataset
                       --data_dir     /path/to/processed
                       --models_dir   /path/to/models
"""

import argparse
import csv
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp"}


# ─────────────────────────────── Helpers ───────────────────────────────────

def _load_test_videos(dataset_dir: Path, split_file: Path
                      ) -> tuple[list[Path], list[int]]:
    stem_label = {}
    with open(split_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                stem_label[row["stem"]] = int(row["label"])

    paths, labels = [], []
    for stem, label in stem_label.items():
        sub = "real" if label == 0 else "fake"
        orig = stem[len(sub)+1:]
        folder = dataset_dir / sub
        for ext in VIDEO_EXTS:
            p = folder / (orig + ext)
            if p.exists():
                paths.append(p)
                labels.append(label)
                break
        else:
            m = list(folder.glob(orig + ".*"))
            if m:
                paths.append(m[0])
                labels.append(label)

    log.info("Test videos: %d (real=%d fake=%d)",
             len(paths), labels.count(0), labels.count(1))
    return paths, labels


def _metrics(y_true: list, y_score: list, name: str):
    y_pred = [1 if s > 0.5 else 0 for s in y_score]
    acc  = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Confusion matrix:")
    print(f"         Pred REAL  Pred FAKE")
    print(f"  True REAL   {cm[0,0]:6d}     {cm[0,1]:6d}")
    print(f"  True FAKE   {cm[1,0]:6d}     {cm[1,1]:6d}")
    print(classification_report(y_true, y_pred, target_names=["REAL","FAKE"]))

    # Best threshold by Youden's J
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j = tpr - fpr
        best_t = thresholds[np.argmax(j)]
        y_best = [1 if s > best_t else 0 for s in y_score]
        print(f"  Best threshold: {best_t:.3f}")
        print(f"  Accuracy @ best: {accuracy_score(y_true, y_best):.4f}")
    except Exception:
        pass

    return {"name": name, "acc": acc, "auc": auc, "f1": f1}


# ─────────────────────────────── Branch evaluation ─────────────────────────

def _eval_branch(videos, labels, models_dir, branch):
    from visual_model   import analyze_visual
    from audio_model    import analyze_audio
    from lip_sync_model import analyze_lip_sync
    from temporal_model import analyze_temporal

    fn_map = {
        "visual":   (analyze_visual,   "visual_score",   "visual_model.pth"),
        "audio":    (analyze_audio,    "audio_score",    "audio_model.pth"),
        "lip_sync": (analyze_lip_sync, "lip_sync_score", "lip_sync_model.pth"),
        "temporal": (analyze_temporal, "temporal_score", "temporal_model.pth"),
    }
    fn, key, mfile = fn_map[branch]
    mp = str(models_dir / mfile) if (models_dir / mfile).exists() else None

    scores = []
    for i, vp in enumerate(videos):
        try:
            r = fn(str(vp), mp)
            scores.append(float(r.get(key, 0.5)))
        except Exception as exc:
            log.warning("[%s] %s: %s", branch, vp.name, exc)
            scores.append(0.5)
        if (i + 1) % 10 == 0:
            log.info("  %s  %d/%d", branch, i+1, len(videos))
    return scores


def _eval_fusion(videos, labels, models_dir):
    from deepfake_detector import DeepfakeDetector
    det = DeepfakeDetector(models_dir=str(models_dir))
    scores = []
    for i, vp in enumerate(videos):
        try:
            r = det.analyze(str(vp))
            scores.append(float(r["ai_generated_probability"]))
        except Exception as exc:
            log.warning("[fusion] %s: %s", vp.name, exc)
            scores.append(0.5)
        if (i + 1) % 5 == 0:
            log.info("  fusion  %d/%d", i+1, len(videos))
    return scores


# ─────────────────────────────── Main ──────────────────────────────────────

def main(args):
    dataset_dir = Path(args.dataset_dir)
    data_dir    = Path(args.data_dir)
    models_dir  = Path(args.models_dir)

    split_file = data_dir / "split_manifest.csv"
    if not split_file.exists():
        sys.exit("split_manifest.csv not found.")

    videos, labels = _load_test_videos(dataset_dir, split_file)
    if not videos:
        sys.exit("No test videos found. Check split_manifest.csv.")

    results = []

    for branch in ["visual", "audio", "lip_sync", "temporal"]:
        if args.branch and branch not in args.branch:
            continue
        log.info("Evaluating %s branch …", branch)
        scores = _eval_branch(videos, labels, models_dir, branch)
        r = _metrics(labels, scores, f"{branch.upper()} BRANCH")
        results.append(r)

    if not args.branch or "fusion" in args.branch:
        log.info("Evaluating fusion model …")
        scores = _eval_fusion(videos, labels, models_dir)
        r = _metrics(labels, scores, "FUSION (FINAL)")
        results.append(r)

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Model':<22}  {'ACC':>6}  {'AUC':>6}  {'F1':>6}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*6}")
    for r in results:
        print(f"  {r['name']:<22}  {r['acc']:>6.4f}  {r['auc']:>6.4f}  {r['f1']:>6.4f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--data_dir",    required=True)
    ap.add_argument("--models_dir",  required=True)
    ap.add_argument("--branch", nargs="*",
                    choices=["visual","audio","lip_sync","temporal","fusion"],
                    help="Which branches to evaluate (default: all)")
    main(ap.parse_args())
