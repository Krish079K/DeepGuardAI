"""
audio_model.py  (updated)
--------------------------
Audio deepfake detection using a trained MLP on 128-d MFCC + spectral
features. Inference module for deepfake_detector.py.

Matching training script: train/train_audio.py
Matching model file     : models/audio_model.pth
Scaler file             : models/audio_scaler.pkl  (StandardScaler fit on train)
"""

import subprocess
import tempfile
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model definition  (must match train_audio.py)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Audio extraction helper
# ─────────────────────────────────────────────
def _extract_audio(video_path: str, out_wav: str, sr: int = 16000) -> bool:
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path,
               "-vn", "-acodec", "pcm_s16le",
               "-ar", str(sr), "-ac", "1",
               out_wav, "-loglevel", "error"]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0 and os.path.exists(out_wav)
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.warning("ffmpeg failed: %s", e)
        return False


# ─────────────────────────────────────────────
# 128-d feature extraction (same as prepare_dataset.py)
# ─────────────────────────────────────────────
def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    feats = []

    # 40 MFCCs × (mean + std) = 80
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())

    # Spectral features × 5 × (mean + std) = 10
    for fn in [librosa.feature.spectral_centroid,
               librosa.feature.spectral_bandwidth,
               librosa.feature.spectral_rolloff,
               librosa.feature.zero_crossing_rate]:
        f = fn(y=y, sr=sr) if fn != librosa.feature.zero_crossing_rate else fn(y)
        feats += [float(np.mean(f)), float(np.std(f))]

    rms = librosa.feature.rms(y=y)
    feats += [float(np.mean(rms)), float(np.std(rms))]

    # Chroma  12 × (mean + std) = 24
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1).tolist())
    feats.extend(np.std(chroma, axis=1).tolist())

    # Spectral contrast  7
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.extend(np.mean(contrast, axis=1).tolist())
    except Exception:
        feats.extend([0.0] * 7)

    # Tonnetz  6
    try:
        harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)
        feats.extend(np.mean(tonnetz, axis=1).tolist())
    except Exception:
        feats.extend([0.0] * 6)

    arr = np.array(feats, dtype=np.float32)
    arr = arr[:128] if len(arr) >= 128 else np.pad(arr, (0, 128 - len(arr)))
    return arr


# ─────────────────────────────────────────────
# Heuristic fallback score
# ─────────────────────────────────────────────
def _heuristic_audio_score(y: np.ndarray, sr: int) -> float:
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500,
                                           sr=sr, frame_length=2048)
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_std   = float(np.std(voiced_f0)) if len(voiced_f0) > 5 else 20.0
        pitch_score = max(0.0, 1.0 - pitch_std / 30.0)

        intervals      = librosa.effects.split(y, top_db=30)
        voiced_samples = sum(e - s for s, e in intervals)
        silence_ratio  = 1.0 - voiced_samples / (len(y) + 1e-8)
        silence_score  = float(np.clip(silence_ratio * 2.0, 0.0, 1.0))

        spec = np.abs(librosa.stft(y))
        smoothness = float(np.mean(np.diff(spec, axis=1) ** 2))
        smooth_score = max(0.0, 1.0 - min(smoothness / 0.5, 1.0))

        return float(np.clip(0.4*pitch_score + 0.3*silence_score + 0.3*smooth_score,
                             0.0, 1.0))
    except Exception as exc:
        logger.warning("Heuristic audio score failed: %s", exc)
        return 0.5


# ─────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────
def analyze_audio(video_path: str, model_path: str | None = None) -> dict:
    """
    Analyse audio track of a video for synthetic-speech artefacts.

    Returns
    -------
    {
        audio_score   : float – fake probability in [0,1]
        has_audio     : bool
        method        : "cnn_model" | "heuristic" | "no_audio"
        features_dim  : int
    }
    """
    sr = 16000
    y  = None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        # ── Extract audio ────────────────────────────────────────────────
        ok = _extract_audio(str(video_path), tmp_wav, sr)
        if ok and os.path.exists(tmp_wav):
            try:
                y, _ = librosa.load(tmp_wav, sr=sr, mono=True, duration=30)
            except Exception:
                y = None
        if y is None:
            try:
                y, _ = librosa.load(str(video_path), sr=sr,
                                     mono=True, duration=30)
            except Exception:
                y = None

        if y is None or len(y) < sr * 0.5:
            return {"audio_score": 0.5, "has_audio": False,
                    "method": "no_audio", "features_dim": 0}

        features = _extract_features(y, sr)

        # ── Load model ───────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net    = None
        method = "heuristic"
        scaler = None

        if model_path and Path(model_path).exists():
            try:
                net = AudioMLP(n_in=128).to(device)
                state = torch.load(model_path, map_location=device)
                net.load_state_dict(state)
                net.eval()
                method = "cnn_model"
                # Try loading matching StandardScaler
                scaler_path = Path(model_path).parent / "audio_scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                logger.info("Audio model loaded: %s", model_path)
            except Exception as exc:
                logger.warning("Cannot load audio model (%s) → heuristic", exc)
                net = None

        # ── Inference ────────────────────────────────────────────────────
        if net is not None:
            feat_in = features.reshape(1, -1)
            if scaler is not None:
                feat_in = scaler.transform(feat_in).astype(np.float32)
            tensor = torch.tensor(feat_in, dtype=torch.float32).to(device)
            with torch.no_grad():
                logit = net(tensor)
                score = torch.sigmoid(logit).item()
        else:
            score = _heuristic_audio_score(y, sr)

        return {
            "audio_score":  round(float(np.clip(score, 0.0, 1.0)), 4),
            "has_audio":    True,
            "method":       method,
            "features_dim": len(features),
        }

    finally:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass
