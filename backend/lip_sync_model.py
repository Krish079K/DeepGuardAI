"""
lip_sync_model.py  (updated)
-----------------------------
Lip-sync consistency analysis.
Uses a trained LipSyncLSTM when models/lip_sync_model.pth is available,
otherwise falls back to audio/motion cross-correlation heuristic.

Training script: train/train_lip_sync.py
"""

import cv2
import numpy as np
import librosa
import subprocess
import tempfile
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SEQ_LEN = 64   # must match train_lip_sync.py


# ─────────────────────────────────────────────
# Model definition (must match train_lip_sync.py)
# ─────────────────────────────────────────────
class LipSyncLSTM(nn.Module):
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
        out, _  = self.lstm(x)
        attn_w  = torch.softmax(self.attn(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.head(context).squeeze(1)


# ─────────────────────────────────────────────
# Signal extractors
# ─────────────────────────────────────────────
def _mouth_motion_signal(video_path: str, seq_len: int = 16) -> np.ndarray:
    seq_len = min(seq_len, 16)
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return np.zeros(seq_len, dtype=np.float32)

    indices = np.linspace(0, total - 1, num=seq_len, dtype=int)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    motion, prev_mouth = [], None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            motion.append(0.0)
            continue
        
        # Fast downscaled face and mouth detection
        fh, fw = frame.shape[:2]
        scale = 240.0 / max(fw, fh) if max(fw, fh) > 240 else 1.0
        small = cv2.resize(frame, (int(fw * scale), int(fh * scale))) if scale < 1.0 else frame
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 4, minSize=(24, 24))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            mouth = gray[y+int(h*0.6):y+h, x+int(w*0.1):x+w-int(w*0.1)]
            if mouth.size > 0:
                mouth = cv2.resize(mouth, (32, 16))
                if prev_mouth is not None and prev_mouth.shape == mouth.shape:
                    motion.append(float(np.abs(
                        mouth.astype(np.float32) - prev_mouth.astype(np.float32)
                    ).mean()))
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
    mx  = arr.max()
    return arr / mx if mx > 1e-6 else arr


def _audio_energy_envelope(video_path: str, seq_len: int = 16,
                            sr: int = 16000) -> np.ndarray:
    seq_len = min(seq_len, 16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-t", "15",
             "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
             tmp_wav, "-loglevel", "error"],
            capture_output=True, timeout=25
        )
        if r.returncode != 0 or not os.path.exists(tmp_wav):
            return np.zeros(seq_len, dtype=np.float32)
        y, _ = librosa.load(tmp_wav, sr=sr, mono=True, duration=15)

        hop  = max(1, len(y) // seq_len)
        rms  = librosa.feature.rms(y=y, hop_length=hop)[0]
        x_old = np.linspace(0, 1, len(rms))
        x_new = np.linspace(0, 1, seq_len)
        env   = np.interp(x_new, x_old, rms).astype(np.float32)
        mx    = env.max()
        return env / mx if mx > 1e-6 else env
    except Exception as exc:
        logger.warning("Audio envelope failed: %s", exc)
        return np.zeros(seq_len, dtype=np.float32)
    finally:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass


# ─────────────────────────────────────────────
# Cross-correlation heuristic fallback
# ─────────────────────────────────────────────
def _correlation_score(motion: np.ndarray, audio: np.ndarray) -> float:
    n = min(len(motion), len(audio))
    if n < 5:
        return 0.5
    m = (motion[:n] - motion[:n].mean()) / (motion[:n].std() + 1e-8)
    a = (audio[:n]  - audio[:n].mean())  / (audio[:n].std()  + 1e-8)
    max_lag = min(5, n // 4)
    corrs   = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.corrcoef(m[lag:], a[:n-lag])[0, 1] if n > lag else 0.0
        else:
            c = np.corrcoef(m[:n+lag], a[-lag:])[0, 1] if n > -lag else 0.0
        corrs.append(c if np.isfinite(c) else 0.0)
    return float(np.clip(1.0 - max(corrs), 0.0, 1.0))


# ─────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────
def analyze_lip_sync(video_path: str, model_path: str | None = None) -> dict:
    """
    Analyse lip-sync consistency between visual mouth movement and audio.

    Returns
    -------
    {
        lip_sync_score : float – fake probability in [0,1]
        motion_frames  : int
        has_motion     : bool
        method         : str
    }
    """
    motion = _mouth_motion_signal(str(video_path))
    audio  = _audio_energy_envelope(str(video_path))

    has_motion = bool(np.any(motion > 0.01))
    if not has_motion:
        return {"lip_sync_score": 0.5, "motion_frames": len(motion),
                "has_motion": False, "method": "no_face_detected"}

    # ── Try LSTM model ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = None
    method = "cross_correlation"

    if model_path and Path(model_path).exists():
        try:
            net = LipSyncLSTM(input_size=2, hidden=64, layers=2).to(device)
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
            method = "lstm_model"
            logger.info("LipSync model loaded: %s", model_path)
        except Exception as exc:
            logger.warning("Cannot load lip-sync model (%s) → heuristic", exc)
            net = None

    if net is not None:
        x = np.stack([motion, audio], axis=1).astype(np.float32)
        tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            score = net(tensor).item()
    else:
        score = _correlation_score(motion, audio)

    return {
        "lip_sync_score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "motion_frames":  len(motion),
        "has_motion":     has_motion,
        "method":         method,
    }
