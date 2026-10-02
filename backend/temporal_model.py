"""
temporal_model.py
-----------------
Temporal consistency analysis for deepfake detection.

Deepfake generation (especially face-swap) often introduces temporal
inconsistencies:
  - Sudden identity flickers between frames
  - Unnatural blinking patterns
  - Inconsistent facial boundary (blending mask) over time
  - Abnormal inter-frame difference statistics

This module computes a temporal fake-probability score.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Temporal LSTM / MLP (if trained model provided)
# ─────────────────────────────────────────────
class TemporalDeepfakeNet(nn.Module):
    """
    Expects a sequence feature vector of shape (seq_len, n_features).
    Uses a bidirectional LSTM followed by a classification head.
    """
    def __init__(self, n_features: int = 16, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Global max-pool over time
        pooled = out.max(dim=1).values
        return self.head(pooled)


# ─────────────────────────────────────────────
# Frame feature extraction
# ─────────────────────────────────────────────
def _per_frame_features(frame: np.ndarray, prev_frame: np.ndarray | None,
                         face_bbox=None) -> np.ndarray:
    """
    Compute a 16-dim feature vector for one frame:
      [0]  Mean pixel difference from previous frame (face region)
      [1]  Std  pixel difference
      [2]  Max  pixel difference
      [3]  Gradient magnitude mean (Sobel)
      [4]  Gradient magnitude std
      [5]  HSV saturation mean (face)
      [6]  HSV value mean      (face)
      [7]  Laplacian variance  (sharpness)
      [8]  Edge density (Canny)
      [9]  Optical flow magnitude mean  (if prev available)
      [10] Optical flow magnitude std
      [11] Inter-frame SSIM proxy       (normalised diff)
      [12] DCT high-frequency energy ratio
      [13] Chroma channel correlation (B-G)
      [14] Face boundary blur estimate
      [15] Frame-level noise estimate (residual)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Crop to face if available
    if face_bbox is not None:
        x, y, w, h = face_bbox
        face = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        face_hsv  = hsv[y:y+h, x:x+w]
    else:
        face = frame
        face_gray = gray
        face_hsv  = hsv

    feats = np.zeros(16, dtype=np.float32)

    # Pixel difference
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if face_bbox is not None:
            x, y, w, h = face_bbox
            prev_crop = prev_gray[y:y+h, x:x+w]
        else:
            prev_crop = prev_gray

        if prev_crop.shape == face_gray.shape:
            diff = np.abs(face_gray - prev_crop)
            feats[0] = float(diff.mean())
            feats[1] = float(diff.std())
            feats[2] = float(diff.max())
        else:
            feats[0:3] = 0.0

        # Optical flow skipped (too RAM-heavy for 512MB free tier)
        # Use gradient-difference proxy instead (cheap, similar signal)
        if prev_crop.shape == face_gray.shape:
            gx1 = cv2.Sobel(face_gray.astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
            gx2 = cv2.Sobel(prev_crop.astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
            motion_proxy = np.abs(gx1 - gx2)
            feats[9]  = float(motion_proxy.mean())
            feats[10] = float(motion_proxy.std())

        # SSIM proxy
        if prev_crop.shape == face_gray.shape:
            mu1 = face_gray.mean(); mu2 = prev_crop.mean()
            sigma1 = face_gray.std(); sigma2 = prev_crop.std()
            cov = float(np.mean((face_gray - mu1) * (prev_crop - mu2)))
            ssim_proxy = (2*mu1*mu2 + 1e-4) * (2*cov + 9e-4) / \
                         ((mu1**2 + mu2**2 + 1e-4) * (sigma1**2 + sigma2**2 + 9e-4))
            feats[11] = 1.0 - float(np.clip(ssim_proxy, 0.0, 1.0))

    # Gradient magnitude
    sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    feats[3] = float(grad_mag.mean())
    feats[4] = float(grad_mag.std())

    # HSV
    feats[5] = float(face_hsv[..., 1].mean())  # saturation
    feats[6] = float(face_hsv[..., 2].mean())  # value

    # Laplacian variance (sharpness)
    feats[7] = float(cv2.Laplacian(face_gray.astype(np.uint8), cv2.CV_64F).var())

    # Edge density
    edges = cv2.Canny(face_gray.astype(np.uint8), 50, 150)
    feats[8] = float(edges.mean())

    # DCT high-freq ratio
    h, w = face_gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 > 0 and w8 > 0:
        block = face_gray[:h8, :w8]
        dct = cv2.dct(block.copy())
        hf = float(np.sum(dct[h8//2:, w8//2:]**2))
        total = float(np.sum(dct**2)) + 1e-8
        feats[12] = hf / total

    # Chroma correlation
    b = face[..., 0].flatten().astype(np.float32)
    g = face[..., 1].flatten().astype(np.float32)
    feats[13] = float(np.corrcoef(b, g)[0, 1]) if len(b) > 1 else 0.0

    # Boundary blur (perimeter of face crop vs interior sharpness)
    if face_gray.shape[0] > 10 and face_gray.shape[1] > 10:
        border = np.concatenate([
            face_gray[:5, :].flatten(),
            face_gray[-5:, :].flatten(),
            face_gray[:, :5].flatten(),
            face_gray[:, -5:].flatten(),
        ])
        interior = face_gray[5:-5, 5:-5].flatten()
        feats[14] = float(np.std(border)) / (float(np.std(interior)) + 1e-8)

    # Noise estimate
    try:
        blur = cv2.GaussianBlur(face_gray, (5, 5), 0)
        noise = face_gray - blur
        feats[15] = float(noise.std())
    except Exception:
        feats[15] = 0.0

    return feats


# ─────────────────────────────────────────────
# Temporal heuristic scoring
# ─────────────────────────────────────────────
def _temporal_heuristic(feature_seq: np.ndarray) -> float:
    """
    Given (T, 16) feature matrix, compute temporal inconsistency score.
    Focuses on:
    - Variance of inter-frame difference over time
    - Sudden spikes in boundary blur
    - Anomalous noise level changes
    """
    if len(feature_seq) < 3:
        return 0.5

    # Inter-frame differences (feature 0 = pixel diff mean)
    diff_series  = feature_seq[:, 0]
    blur_series  = feature_seq[:, 14]
    noise_series = feature_seq[:, 15]

    # Spike detection via local z-score
    def spike_score(series):
        if series.std() < 1e-6:
            return 0.0
        z = (series - series.mean()) / series.std()
        spike_fraction = float(np.mean(np.abs(z) > 2.5))
        return min(spike_fraction * 3.0, 1.0)

    s_diff  = spike_score(diff_series)
    s_blur  = spike_score(blur_series)
    s_noise = spike_score(noise_series)

    # Temporal consistency: high variance in inter-frame diff is suspicious
    consistency = float(np.std(np.diff(diff_series))) / \
                  (float(np.mean(diff_series)) + 1e-8)
    consistency_score = float(np.clip(consistency / 5.0, 0.0, 1.0))

    combined = 0.3 * s_diff + 0.25 * s_blur + 0.2 * s_noise + 0.25 * consistency_score
    return float(np.clip(combined, 0.0, 1.0))


# ─────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────
def analyze_temporal(video_path: str, model_path: str | None = None,
                     num_frames: int = 10) -> dict:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return {"temporal_score": 0.5, "frames_analysed": total, "method": "no_frames"}

    num_frames = min(num_frames, 10)
    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frames_data = []
    prev_frame = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Resize to max 480px
        if frame.shape[1] > 480:
            scale = 480.0 / frame.shape[1]
            frame = cv2.resize(frame, (480, int(frame.shape[0] * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        scale_det = 240.0 / max(fw, fh) if max(fw, fh) > 240 else 1.0
        small_gray = cv2.resize(gray, (int(fw * scale_det), int(fh * scale_det))) if scale_det < 1.0 else gray
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.2,
                                              minNeighbors=4, minSize=(24, 24))
        face_bbox = None
        if len(faces) > 0:
            bx, by, bw, bh = max(faces, key=lambda f: f[2] * f[3])
            if scale_det < 1.0:
                bx, by, bw, bh = int(bx / scale_det), int(by / scale_det), int(bw / scale_det), int(bh / scale_det)
            face_bbox = (bx, by, bw, bh)

        feats = _per_frame_features(frame, prev_frame, face_bbox=face_bbox)
        frames_data.append(feats)
        prev_frame = frame

    cap.release()


    if not frames_data:
        return {"temporal_score": 0.5, "frames_analysed": 0, "method": "no_frames"}

    feature_seq = np.array(frames_data, dtype=np.float32)

    # ── Try LSTM model ───────────────────────────────────────────────────
    device = torch.device("cpu")
    net = None
    method = "heuristic"

    if model_path and Path(model_path).exists():
        try:
            net = TemporalDeepfakeNet(n_features=16, hidden=64).to(device)
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
            method = "lstm_model"
            logger.info("Temporal model loaded from %s", model_path)
        except Exception as exc:
            logger.warning("Cannot load temporal model (%s); using heuristics.", exc)
            net = None

    if net is not None:
        tensor = torch.tensor(feature_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            score = net(tensor).item()
    else:
        score = _temporal_heuristic(feature_seq)

    return {
        "temporal_score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "frames_analysed": len(frames_data),
        "method": method,
    }
