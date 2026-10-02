"""
fusion_model.py
---------------
Multimodal fusion layer that combines scores from the four analysis branches:
  - visual_score
  - audio_score
  - lip_sync_score
  - temporal_score

Supports two modes:
  1. Trained MLP fusion model (if a .pth file is provided)
  2. Learned-weight ensemble with confidence-aware weighting (default)

The final output is:
  - ai_generated_probability  (probability video is deepfake)
  - real_probability           (probability video is genuine)
  - prediction                 (string label)
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Trained fusion MLP (optional)
# ─────────────────────────────────────────────
class FusionMLP(nn.Module):
    """
    A small MLP that takes the 4 branch scores as input.
    Input shape: (batch, 4)
    Output: (batch, 1) – fake probability
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Branch confidence estimation
# ─────────────────────────────────────────────
def _branch_confidence(branch_result: dict, branch_name: str) -> float:
    """
    Estimate how reliable a branch result is, based on metadata flags.
    Returns a weight multiplier in (0, 1].
    """
    if branch_name == "audio":
        if not branch_result.get("has_audio", True):
            return 0.1   # Very low weight if no audio found
        if branch_result.get("method", "") == "no_audio":
            return 0.05
        return 1.0

    if branch_name == "lip_sync":
        if not branch_result.get("has_motion", True):
            return 0.2
        if branch_result.get("method", "") == "no_face_detected":
            return 0.1
        return 1.0

    if branch_name == "visual":
        faces = branch_result.get("faces_detected", 0)
        total = branch_result.get("frames_analyzed", 1)
        if total == 0:
            return 0.3
        face_ratio = faces / total
        return max(0.3, face_ratio)

    if branch_name == "temporal":
        frames = branch_result.get("frames_analysed", 0)
        if frames < 4:
            return 0.3
        return 1.0

    return 1.0


# ─────────────────────────────────────────────
# Default ensemble weights
# (calibrated heuristically for face-swap deepfakes)
# ─────────────────────────────────────────────
BASE_WEIGHTS = {
    "visual":   0.40,
    "audio":    0.20,
    "lip_sync": 0.20,
    "temporal": 0.20,
}

THRESHOLD_FAKE  = 0.55   # score > 0.55 → LIKELY AI-GENERATED
THRESHOLD_REAL  = 0.45   # score < 0.45 → LIKELY REAL
# between → UNCERTAIN (treated as whichever side is closer)


# ─────────────────────────────────────────────
# Main fusion function
# ─────────────────────────────────────────────
def fuse_scores(
    visual_result: dict,
    audio_result: dict,
    lip_sync_result: dict,
    temporal_result: dict,
    model_path: str | None = None,
) -> dict:
    """
    Fuse the four branch results into a final verdict.

    Parameters
    ----------
    visual_result    : Output of visual_model.analyze_visual()
    audio_result     : Output of audio_model.analyze_audio()
    lip_sync_result  : Output of lip_sync_model.analyze_lip_sync()
    temporal_result  : Output of temporal_model.analyze_temporal()
    model_path       : Optional path to trained FusionMLP .pth file.

    Returns
    -------
    dict:
        visual_score           – visual branch fake prob
        audio_score            – audio branch fake prob
        lip_sync_score         – lip-sync branch fake prob
        temporal_score         – temporal branch fake prob
        ai_generated_probability
        real_probability
        prediction             – "LIKELY REAL" | "LIKELY AI-GENERATED"
    """
    scores = {
        "visual":   float(visual_result.get("visual_score", 0.5)),
        "audio":    float(audio_result.get("audio_score", 0.5)),
        "lip_sync": float(lip_sync_result.get("lip_sync_score", 0.5)),
        "temporal": float(temporal_result.get("temporal_score", 0.5)),
    }

    # ── Try MLP fusion model ─────────────────────────────────────────────
    device = torch.device("cpu")
    net = None

    if model_path and Path(model_path).exists():
        try:
            net = FusionMLP().to(device)
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
            logger.info("Fusion MLP model loaded from %s", model_path)
        except Exception as exc:
            logger.warning("Cannot load fusion model (%s); using ensemble.", exc)
            net = None

    if net is not None:
        x = torch.tensor(
            [scores["visual"], scores["audio"], scores["lip_sync"], scores["temporal"]],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            fake_prob = float(net(x).item())
    else:
        # ── Confidence-weighted ensemble ─────────────────────────────────
        confidences = {
            "visual":   _branch_confidence(visual_result, "visual"),
            "audio":    _branch_confidence(audio_result, "audio"),
            "lip_sync": _branch_confidence(lip_sync_result, "lip_sync"),
            "temporal": _branch_confidence(temporal_result, "temporal"),
        }

        weighted_sum  = 0.0
        total_weight  = 0.0
        for branch in BASE_WEIGHTS:
            w = BASE_WEIGHTS[branch] * confidences[branch]
            weighted_sum  += w * scores[branch]
            total_weight  += w

        if total_weight < 1e-8:
            fake_prob = 0.5
        else:
            fake_prob = weighted_sum / total_weight

    fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
    real_prob = 1.0 - fake_prob

    if fake_prob >= THRESHOLD_FAKE:
        prediction = "LIKELY AI-GENERATED"
    else:
        prediction = "LIKELY REAL"

    return {
        "visual_score":             round(scores["visual"],   4),
        "audio_score":              round(scores["audio"],    4),
        "lip_sync_score":           round(scores["lip_sync"], 4),
        "temporal_score":           round(scores["temporal"], 4),
        "real_probability":         round(real_prob,  4),
        "ai_generated_probability": round(fake_prob,  4),
        "prediction":               prediction,
    }
