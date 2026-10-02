"""
test_pipeline.py
================
End-to-end CLI test for the DeepGuard AI pipeline.

Analyzes one video through all 4 branches + fusion model and
prints a full detailed report.

Usage:
    python test_pipeline.py --video /path/to/video.mp4
                            --models_dir ./models

    python test_pipeline.py --video sample.mp4    # uses ./models by default
"""

import argparse
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_pipeline")

# Ensure backend/ is on path
sys.path.insert(0, str(Path(__file__).parent))

# Handle Windows cp1252 console encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def _bar(score: float, width: int = 30) -> str:
    """ASCII progress bar for a score in [0, 1]."""
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score*100:5.1f}%"


def _label_colour(prediction: str) -> str:
    if "REAL" in prediction:
        return "✅ " + prediction
    return "⚠️  " + prediction


def main(args):
    video_path = Path(args.video)
    models_dir = Path(args.models_dir)

    if not video_path.exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          DEEPGUARD AI  —  Deepfake Analysis          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Video : {video_path.name}")
    print(f"  Models: {models_dir.resolve()}\n")

    # ── Import all modules ───────────────────────────────────────────────
    from visual_model   import analyze_visual
    from audio_model    import analyze_audio
    from lip_sync_model import analyze_lip_sync
    from temporal_model import analyze_temporal
    from fusion_model   import fuse_scores

    def _find_model(name: str) -> str | None:
        p = models_dir / name
        if p.exists():
            log.info("  Model loaded: %s", name)
            return str(p)
        log.warning("  Model NOT found: %s — using heuristic", name)
        return None

    vis_model = _find_model("visual_model.pth")
    aud_model = _find_model("audio_model.pth")
    lip_model = _find_model("lip_sync_model.pth")
    tmp_model = _find_model("temporal_model.pth")
    fus_model = _find_model("fusion_model.pth")

    video_str = str(video_path)
    t_total = time.time()

    # ── Branch 1: Visual ─────────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  [1/5]  Analyzing visual features …                 │")
    print("  └─────────────────────────────────────────────────────┘")
    t0 = time.time()
    try:
        vis_result = analyze_visual(video_str, model_path=vis_model, num_frames=20)
        vis_ok = True
    except Exception as e:
        log.error("Visual branch failed: %s", e)
        vis_result = {"visual_score": 0.5, "faces_detected": 0, "method": "error"}
        vis_ok = False

    print(f"     Score  : {_bar(vis_result['visual_score'])}")
    print(f"     Faces  : {vis_result.get('faces_detected', '?')} "
          f"/ {vis_result.get('frames_analyzed', '?')} frames")
    print(f"     Method : {vis_result.get('method', '?')}")
    print(f"     Time   : {time.time()-t0:.1f}s\n")

    # ── Branch 2: Audio ──────────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  [2/5]  Analyzing audio track …                     │")
    print("  └─────────────────────────────────────────────────────┘")
    t0 = time.time()
    try:
        aud_result = analyze_audio(video_str, model_path=aud_model)
        aud_ok = True
    except Exception as e:
        log.error("Audio branch failed: %s", e)
        aud_result = {"audio_score": 0.5, "has_audio": False, "method": "error"}
        aud_ok = False

    print(f"     Score  : {_bar(aud_result['audio_score'])}")
    print(f"     Audio  : {'✓ extracted' if aud_result.get('has_audio') else '✗ not found'}")
    print(f"     Method : {aud_result.get('method', '?')}")
    print(f"     Time   : {time.time()-t0:.1f}s\n")

    # ── Branch 3: Lip-Sync ───────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  [3/5]  Checking lip synchronisation …              │")
    print("  └─────────────────────────────────────────────────────┘")
    t0 = time.time()
    try:
        lip_result = analyze_lip_sync(video_str, model_path=lip_model)
        lip_ok = True
    except Exception as e:
        log.error("Lip-sync branch failed: %s", e)
        lip_result = {"lip_sync_score": 0.5, "has_motion": False, "method": "error"}
        lip_ok = False

    print(f"     Score  : {_bar(lip_result['lip_sync_score'])}")
    print(f"     Motion : {'✓ detected' if lip_result.get('has_motion') else '✗ no mouth motion'}")
    print(f"     Frames : {lip_result.get('motion_frames', '?')}")
    print(f"     Method : {lip_result.get('method', '?')}")
    print(f"     Time   : {time.time()-t0:.1f}s\n")

    # ── Branch 4: Temporal ───────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  [4/5]  Analyzing temporal consistency …            │")
    print("  └─────────────────────────────────────────────────────┘")
    t0 = time.time()
    try:
        tmp_result = analyze_temporal(video_str, model_path=tmp_model, num_frames=48)
        tmp_ok = True
    except Exception as e:
        log.error("Temporal branch failed: %s", e)
        tmp_result = {"temporal_score": 0.5, "frames_analysed": 0, "method": "error"}
        tmp_ok = False

    print(f"     Score  : {_bar(tmp_result['temporal_score'])}")
    print(f"     Frames : {tmp_result.get('frames_analysed', '?')}")
    print(f"     Method : {tmp_result.get('method', '?')}")
    print(f"     Time   : {time.time()-t0:.1f}s\n")

    # ── Fusion ───────────────────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  [5/5]  Running multimodal fusion …                 │")
    print("  └─────────────────────────────────────────────────────┘")
    t0 = time.time()
    fusion = fuse_scores(vis_result, aud_result, lip_result, tmp_result,
                         model_path=fus_model)
    print(f"     Time   : {time.time()-t0:.2f}s\n")

    total_time = time.time() - t_total

    # ── Final Report ─────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║                    ANALYSIS RESULT                   ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  File    : {video_path.name:<41} ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Visual   {_bar(fusion['visual_score'],  28)}  ║")
    print(f"║  Audio    {_bar(fusion['audio_score'],   28)}  ║")
    print(f"║  Lip-Sync {_bar(fusion['lip_sync_score'],28)}  ║")
    print(f"║  Temporal {_bar(fusion['temporal_score'],28)}  ║")
    print("╠══════════════════════════════════════════════════════╣")
    rp = fusion['real_probability']
    fp = fusion['ai_generated_probability']
    print(f"║  REAL      probability : {rp*100:5.1f}%                     ║")
    print(f"║  AI-GEN    probability : {fp*100:5.1f}%                     ║")
    print("╠══════════════════════════════════════════════════════╣")
    pred_str = _label_colour(fusion['prediction'])
    print(f"║  VERDICT  : {pred_str:<43} ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Total analysis time   : {total_time:.1f}s                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Return exit code: 0=real, 1=fake (useful in shell scripts)
    return 0 if "REAL" in fusion["prediction"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="DeepGuard AI — end-to-end deepfake detection CLI"
    )
    ap.add_argument("--video",      required=True,
                    help="Path to video file to analyze")
    ap.add_argument("--models_dir", default="models",
                    help="Directory containing trained .pth files")
    sys.exit(main(ap.parse_args()))
