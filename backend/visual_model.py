"""
visual_model.py  (updated)
--------------------------
Visual deepfake detection using fine-tuned MobileNetV3-Small.
This file is the inference module used by deepfake_detector.py.
The matching training script is: train/train_visual.py

Model architecture:
    MobileNetV3-Small (ImageNet pretrained)
    → classifier head replaced with: Dropout(0.3) → Linear(in, 64)
                                      → Hardswish → Linear(64, 1)

When no trained .pth is found, falls back to handcrafted pixel-artifact
heuristics so the pipeline still produces real (non-hardcoded) scores.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model definition  (must match train_visual.py)
# ─────────────────────────────────────────────
def build_visual_model() -> nn.Module:
    """Construct the same architecture used during training."""
    net = models.mobilenet_v3_small(weights=None)
    in_features = net.classifier[3].in_features
    net.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 64),
        nn.Hardswish(),
        nn.Linear(64, 1),
    )
    return net


# ─────────────────────────────────────────────
# Image pre-processing (inference-time only)
# ─────────────────────────────────────────────
INFER_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Heuristic fallback (no trained model)
# ─────────────────────────────────────────────
def _pixel_artifact_score(frame_bgr: np.ndarray) -> float:
    """
    Estimate deepfake probability from raw pixel statistics:
    • Laplacian variance  (edge blurring / sharpening artefacts)
    • DCT high-freq ratio (block compression anomalies)
    • Inter-channel correlation (GAN colour plane misalignment)
    Returns score ∈ [0, 1] (higher = more likely fake).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_score = 1.0 - float(np.clip((lap_var - 50) / 750, 0.0, 1.0))

    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 > 0 and w8 > 0:
        g32 = gray[:h8, :w8].astype(np.float32)
        dct_scores = []
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                dct = cv2.dct(g32[i:i+8, j:j+8])
                hf  = np.sum(dct[4:, 4:]**2)
                tot = np.sum(dct**2) + 1e-8
                dct_scores.append(hf / tot)
        dct_score = float(np.mean(dct_scores))
    else:
        dct_score = 0.5

    b = frame_bgr[:, :, 0].flatten().astype(np.float32)
    g = frame_bgr[:, :, 1].flatten().astype(np.float32)
    r = frame_bgr[:, :, 2].flatten().astype(np.float32)
    corr_rg = float(np.corrcoef(r, g)[0, 1])
    corr_rb = float(np.corrcoef(r, b)[0, 1])
    channel_score = 1.0 - (abs(corr_rg) + abs(corr_rb)) / 2.0

    return float(np.clip(0.4*lap_score + 0.35*dct_score + 0.25*channel_score,
                         0.0, 1.0))


# ─────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────
def analyze_visual(video_path: str, model_path: str | None = None,
                   num_frames: int = 20) -> dict:
    """
    Analyse the visual stream of a video.

    Parameters
    ----------
    video_path  : Path to input video.
    model_path  : Optional path to trained visual_model.pth.
                  If None or not found → uses heuristic fallback.
    num_frames  : Frames to sample (default 20).

    Returns
    -------
    {
        visual_score     : float  – fake probability in [0,1]
        frame_scores     : list   – per-frame scores
        faces_detected   : int
        frames_analyzed  : int
        method           : "cnn_model" | "heuristic"
    }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        raise ValueError("Video has no frames.")

    # Extract up to 8 evenly spaced frames
    num_frames = min(num_frames, 8)
    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if ret and fr is not None:
            # Resize frame to max width 640 to prevent massive memory/CPU usage
            if fr.shape[1] > 640:
                scale = 640.0 / fr.shape[1]
                fr = cv2.resize(fr, (640, int(fr.shape[0] * scale)))
            frames.append(fr)
    cap.release()

    if not frames:
        raise ValueError("Could not read any frames.")

    # ── Load model (if available) ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = None
    method = "heuristic"

    if model_path and Path(model_path).exists():
        try:
            net = build_visual_model().to(device)
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
            method = "cnn_model"
            logger.info("Visual model loaded: %s", model_path)
        except Exception as exc:
            logger.warning("Cannot load visual model (%s) → heuristic", exc)
            net = None

    # ── Score each frame ─────────────────────────────────────────────────
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    frame_scores  = []
    faces_detected = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fast face detection on downscaled thumbnail
        fh, fw = gray.shape
        scale_det = 240.0 / max(fw, fh) if max(fw, fh) > 240 else 1.0
        small_gray = cv2.resize(gray, (int(fw * scale_det), int(fh * scale_det))) if scale_det < 1.0 else gray
        faces = cascade.detectMultiScale(small_gray, 1.2, 4, minSize=(24, 24))

        if len(faces) > 0:
            faces_detected += 1
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            if scale_det < 1.0:
                x, y, w, h = int(x / scale_det), int(y / scale_det), int(w / scale_det), int(h / scale_det)
            pad_x, pad_y = int(w * 0.10), int(h * 0.10)
            x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame


        if net is not None:
            try:
                rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = INFER_TF(rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    logit = net(tensor)
                    score = torch.sigmoid(logit).item()
                frame_scores.append(float(score))
            except Exception:
                frame_scores.append(_pixel_artifact_score(crop))
        else:
            frame_scores.append(_pixel_artifact_score(crop))

    if not frame_scores:
        visual_score = 0.5
    else:
        arr     = np.array(frame_scores)
        weights = np.where(arr > 0.65, 2.0,
                  np.where(arr < 0.30, 0.5, 1.0))
        visual_score = float(np.average(arr, weights=weights))

    return {
        "visual_score":    round(float(np.clip(visual_score, 0.0, 1.0)), 4),
        "frame_scores":    [round(s, 4) for s in frame_scores],
        "faces_detected":  faces_detected,
        "frames_analyzed": len(frames),
        "method":          method,
    }
