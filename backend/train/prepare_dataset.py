"""
prepare_dataset.py
==================
Converts a folder of real and fake videos into training-ready datasets.

Expected input layout (FaceForensics++ or your own data):
    dataset/
        real/      ← genuine videos (.mp4 / .avi / .mov)
        fake/      ← deepfake videos (.mp4 / .avi / .mov)

Outputs (written to dataset/processed/):
    frames/
        real/<video_stem>/  ← face-crop PNG frames
        fake/<video_stem>/
    audio_features.csv      ← 128-d MFCC vector per video + label
    temporal_features.csv   ← 16-d * T per video + label (stored as JSON)
    frame_manifest.csv      ← path, label for visual DataLoader
    split_manifest.csv      ← train/val/test split for all modalities

Usage:
    python prepare_dataset.py --dataset_dir /path/to/dataset
                              --out_dir     /path/to/dataset/processed
                              --max_frames  16
                              --max_videos  500    # per class; 0 = all
                              --val_split   0.15
                              --test_split  0.10
"""

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import librosa

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("prepare")

# ─────────────────────────────── helpers ───────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp"}


def _list_videos(folder: Path, limit: int = 0) -> list[Path]:
    vids = [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    random.shuffle(vids)
    return vids[:limit] if limit > 0 else vids


def _open_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def _extract_face_crops(video_path: Path, out_dir: Path,
                         n_frames: int, cascade) -> list[Path]:
    """Sample n_frames from video, detect face, save cropped PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    saved = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Add 10% margin
            pad_x, pad_y = int(w * 0.10), int(h * 0.10)
            x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame   # fallback: full frame

        crop = cv2.resize(crop, (224, 224))
        out_path = out_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(out_path), crop)
        saved.append(out_path)

    cap.release()
    return saved


def _extract_wav(video_path: Path, out_wav: Path, sr: int = 16000) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
             str(out_wav), "-loglevel", "error"],
            capture_output=True, timeout=120
        )
        return r.returncode == 0 and out_wav.exists()
    except Exception:
        return False


def _audio_features_128(video_path: Path, sr: int = 16000) -> np.ndarray | None:
    """Return 128-d MFCC+spectral feature vector (same as audio_model.py)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        ok = _extract_wav(video_path, tmp_wav, sr)
        if ok:
            y, _ = librosa.load(str(tmp_wav), sr=sr, mono=True, duration=30)
        else:
            try:
                y, _ = librosa.load(str(video_path), sr=sr, mono=True, duration=30)
            except Exception:
                return None

        if y is None or len(y) < sr * 0.5:
            return None

        feats = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feats.extend(np.mean(mfcc, axis=1).tolist())
        feats.extend(np.std(mfcc, axis=1).tolist())

        for fn in [librosa.feature.spectral_centroid,
                   librosa.feature.spectral_bandwidth,
                   librosa.feature.spectral_rolloff,
                   librosa.feature.zero_crossing_rate]:
            f = fn(y=y, sr=sr) if fn != librosa.feature.zero_crossing_rate else fn(y)
            feats += [float(np.mean(f)), float(np.std(f))]

        rms = librosa.feature.rms(y=y)
        feats += [float(np.mean(rms)), float(np.std(rms))]

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats.extend(np.mean(chroma, axis=1).tolist())
        feats.extend(np.std(chroma, axis=1).tolist())

        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            feats.extend(np.mean(contrast, axis=1).tolist())
        except Exception:
            feats.extend([0.0] * 7)

        try:
            harm = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)
            feats.extend(np.mean(tonnetz, axis=1).tolist())
        except Exception:
            feats.extend([0.0] * 6)

        arr = np.array(feats, dtype=np.float32)
        if len(arr) < 128:
            arr = np.pad(arr, (0, 128 - len(arr)))
        return arr[:128]
    except Exception as e:
        log.warning("Audio features failed for %s: %s", video_path.name, e)
        return None
    finally:
        try:
            tmp_wav.unlink()
        except Exception:
            pass


def _temporal_features(video_path: Path, n_frames: int = 32) -> list[list[float]] | None:
    """Return list of 16-d feature vectors (one per frame)."""
    # Reuse logic from temporal_model.py
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from temporal_model import _per_frame_features
    except ImportError:
        log.warning("temporal_model not importable; skipping temporal features.")
        return None

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    cascade = _open_face_cascade()
    feats = []
    prev_frame = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(60, 60))
        bbox = tuple(max(faces, key=lambda f: f[2]*f[3]).tolist()) if len(faces) > 0 else None
        f = _per_frame_features(frame, prev_frame, face_bbox=bbox)
        feats.append(f.tolist())
        prev_frame = frame

    cap.release()
    return feats if feats else None


# ─────────────────────────────── main logic ────────────────────────────────

def prepare(dataset_dir: Path, out_dir: Path, max_frames: int,
            max_videos: int, val_split: float, test_split: float):
    random.seed(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = out_dir / "frames"
    cascade = _open_face_cascade()

    frame_rows = []         # for frame_manifest.csv
    audio_rows = []         # for audio_features.csv
    temporal_rows = []      # for temporal_features.csv
    video_ids = []          # (video_stem, label) for split

    for label_name, label_int in [("real", 0), ("fake", 1)]:
        src_dir = dataset_dir / label_name
        if not src_dir.exists():
            log.warning("Folder not found: %s  (skipping %s)", src_dir, label_name)
            continue

        videos = _list_videos(src_dir, limit=max_videos)
        log.info("[%s] %d videos found", label_name, len(videos))

        for i, vpath in enumerate(videos):
            stem = f"{label_name}_{vpath.stem}"
            log.info("  [%d/%d] %s", i+1, len(videos), vpath.name)

            # ── Face crops ────────────────────────────────────────────────
            crop_dir = frames_root / label_name / stem
            crops = _extract_face_crops(vpath, crop_dir, max_frames, cascade)
            for cp in crops:
                frame_rows.append({
                    "path": str(cp.relative_to(out_dir)),
                    "label": label_int,
                })

            # ── Audio features ────────────────────────────────────────────
            af = _audio_features_128(vpath)
            if af is not None:
                row = {"stem": stem, "label": label_int}
                for j, v in enumerate(af.tolist()):
                    row[f"f{j}"] = round(v, 6)
                audio_rows.append(row)

            # ── Temporal features ─────────────────────────────────────────
            tf = _temporal_features(vpath, n_frames=max_frames * 2)
            if tf is not None:
                temporal_rows.append({
                    "stem": stem,
                    "label": label_int,
                    "features_json": json.dumps(tf),
                })

            video_ids.append((stem, label_int))

    # ── Write CSVs ─────────────────────────────────────────────────────────
    _write_csv(out_dir / "frame_manifest.csv", frame_rows,
               fieldnames=["path", "label"])
    log.info("frame_manifest.csv → %d face-crop entries", len(frame_rows))

    if audio_rows:
        audio_fields = ["stem", "label"] + [f"f{j}" for j in range(128)]
        _write_csv(out_dir / "audio_features.csv", audio_rows,
                   fieldnames=audio_fields)
        log.info("audio_features.csv → %d video entries", len(audio_rows))

    if temporal_rows:
        _write_csv(out_dir / "temporal_features.csv", temporal_rows,
                   fieldnames=["stem", "label", "features_json"])
        log.info("temporal_features.csv → %d video entries", len(temporal_rows))

    # ── Train / Val / Test split ───────────────────────────────────────────
    random.shuffle(video_ids)
    n = len(video_ids)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    n_train = n - n_test - n_val

    splits = (
        [("train", v) for v in video_ids[:n_train]] +
        [("val",   v) for v in video_ids[n_train:n_train+n_val]] +
        [("test",  v) for v in video_ids[n_train+n_val:]]
    )
    split_rows = [{"split": s, "stem": v[0], "label": v[1]} for s, v in splits]
    _write_csv(out_dir / "split_manifest.csv", split_rows,
               fieldnames=["split", "stem", "label"])
    log.info("split_manifest.csv → train=%d  val=%d  test=%d",
             n_train, n_val, n_test)

    log.info("✓ Dataset preparation complete → %s", out_dir)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    if not rows:
        log.warning("No rows for %s — skipping", path.name)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────── CLI ───────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DeepGuard AI — dataset preparation")
    ap.add_argument("--dataset_dir", required=True,
                    help="Root folder with real/ and fake/ sub-folders")
    ap.add_argument("--out_dir", default=None,
                    help="Output folder (default: dataset_dir/processed)")
    ap.add_argument("--max_frames", type=int, default=16,
                    help="Face-crop frames to extract per video")
    ap.add_argument("--max_videos", type=int, default=0,
                    help="Max videos per class (0 = all)")
    ap.add_argument("--val_split",  type=float, default=0.15)
    ap.add_argument("--test_split", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "processed"

    prepare(dataset_dir, out_dir,
            max_frames=args.max_frames,
            max_videos=args.max_videos,
            val_split=args.val_split,
            test_split=args.test_split)
