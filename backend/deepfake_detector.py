"""
deepfake_detector.py
--------------------
Orchestrator that calls all four analysis branches in parallel and
combines results through the fusion model.

Usage:
    from deepfake_detector import DeepfakeDetector
    detector = DeepfakeDetector()
    result = detector.analyze("path/to/video.mp4")
"""

import concurrent.futures
import logging
import time
from pathlib import Path

from visual_model   import analyze_visual
from audio_model    import analyze_audio
from lip_sync_model import analyze_lip_sync
from temporal_model import analyze_temporal
from fusion_model   import fuse_scores

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    High-level deepfake detection pipeline.

    Parameters
    ----------
    models_dir : Directory containing optional .pth model files.
                 If a file is present it will be used; otherwise the
                 module falls back to its built-in heuristics.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)

        self.visual_model_path   = self._find_model("visual_model.pth")
        self.audio_model_path    = self._find_model("audio_model.pth")
        self.lip_sync_model_path = self._find_model("lip_sync_model.pth")
        self.temporal_model_path = self._find_model("temporal_model.pth")
        self.fusion_model_path   = self._find_model("fusion_model.pth")

        logger.info("DeepfakeDetector initialised.")
        logger.info("  Visual   model : %s", self.visual_model_path or "heuristic")
        logger.info("  Audio    model : %s", self.audio_model_path   or "heuristic")
        logger.info("  Lip-sync model : %s", self.lip_sync_model_path or "heuristic")
        logger.info("  Temporal model : %s", self.temporal_model_path or "heuristic")
        logger.info("  Fusion   model : %s", self.fusion_model_path   or "ensemble")

    def _find_model(self, filename: str) -> str | None:
        p = self.models_dir / filename
        return str(p) if p.exists() else None

    # ─────────────────────────────────────────────────────────────────────
    def analyze(self, video_path: str) -> dict:
        """
        Run the full multimodal deepfake detection pipeline.

        Parameters
        ----------
        video_path : Absolute or relative path to the input video.

        Returns
        -------
        dict with keys (all from fuse_scores):
            visual_score, audio_score, lip_sync_score, temporal_score,
            real_probability, ai_generated_probability, prediction,
            + metadata: filename, processing_time_seconds, branch_details
        """
        video_path = str(video_path)
        t0 = time.time()

        logger.info("Starting analysis: %s", video_path)

        # ── Run all four branches in parallel (thread pool) ───────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fut_visual   = pool.submit(analyze_visual,   video_path,
                                        self.visual_model_path)
            fut_audio    = pool.submit(analyze_audio,    video_path,
                                        self.audio_model_path)
            fut_lip_sync = pool.submit(analyze_lip_sync, video_path,
                                        self.lip_sync_model_path)
            fut_temporal = pool.submit(analyze_temporal, video_path,
                                        self.temporal_model_path)

            # Collect results (exceptions are surfaced here)
            visual_result   = self._safe_get(fut_visual,   "visual",   video_path)
            audio_result    = self._safe_get(fut_audio,    "audio",    video_path)
            lip_sync_result = self._safe_get(fut_lip_sync, "lip_sync", video_path)
            temporal_result = self._safe_get(fut_temporal, "temporal", video_path)

        logger.info("Branch scores → visual=%.3f  audio=%.3f  lip=%.3f  temp=%.3f",
                    visual_result.get("visual_score", -1),
                    audio_result.get("audio_score", -1),
                    lip_sync_result.get("lip_sync_score", -1),
                    temporal_result.get("temporal_score", -1))

        # ── Fusion ───────────────────────────────────────────────────────
        fusion_result = fuse_scores(
            visual_result, audio_result, lip_sync_result, temporal_result,
            model_path=self.fusion_model_path,
        )

        elapsed = round(time.time() - t0, 2)
        logger.info("Analysis complete in %.2fs — prediction: %s",
                    elapsed, fusion_result["prediction"])

        # ── Build full response dict ─────────────────────────────────────
        return {
            **fusion_result,
            "filename": Path(video_path).name,
            "processing_time_seconds": elapsed,
            "branch_details": {
                "visual":   visual_result,
                "audio":    audio_result,
                "lip_sync": lip_sync_result,
                "temporal": temporal_result,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_get(future: concurrent.futures.Future,
                  branch: str, video_path: str) -> dict:
        """Return future result or a safe fallback on exception."""
        try:
            return future.result(timeout=300)
        except Exception as exc:
            logger.error("Branch '%s' failed for %s: %s", branch, video_path, exc)
            fallback_key = f"{branch}_score" if branch != "lip_sync" else "lip_sync_score"
            return {fallback_key: 0.5, "error": str(exc), "method": "fallback"}
