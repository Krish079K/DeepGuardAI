"""
main.py
-------
FastAPI backend for DeepGuard AI.

Endpoints:
    GET  /health
    POST /analyze
    GET  /history
    GET  /history/{id}
    DELETE /history/{id}
    POST /history         (manual save, for testing)
"""

import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Ensure the backend directory is on sys.path ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from database         import init_db, save_analysis, get_all_analyses, \
                              get_analysis_by_id, delete_analysis
from deepfake_detector import DeepfakeDetector

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deepguard.api")

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="DeepGuard AI",
    description="Multimodal AI-Based Deepfake Detection API",
    version="1.0.0",
)

# Allow all origins (suitable for development / local phone testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR  = Path(__file__).parent / "uploads"
MODELS_DIR  = Path(__file__).parent / "models"
STATIC_DIR  = Path(__file__).parent / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Detector (singleton) ─────────────────────────────────────────────────
detector = DeepfakeDetector(models_dir=str(MODELS_DIR))

# ── DB init ──────────────────────────────────────────────────────────────
init_db()

# ── Mount static files (logo, favicon, etc.) ─────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=FileResponse, tags=["Web"])
def serve_web_ui():
    """Serves the DeepGuard AI Web Application."""
    index_file = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_file)

@app.get("/health", tags=["System"])
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "service": "DeepGuard AI Backend", "version": "1.0.0"}


# ── POST /analyze ─────────────────────────────────────────────────────────
@app.post("/analyze", tags=["Detection"])
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video and run multimodal deepfake analysis.

    Returns JSON with branch scores and final prediction.
    """
    # Validate extension
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp", ".flv"}
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # Save uploaded file with a unique name
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name

    try:
        with open(save_path, "wb") as fp:
            shutil.copyfileobj(file.file, fp)
        logger.info("Received upload: %s → %s", file.filename, save_path)

        # ── Run AI analysis ───────────────────────────────────────────────
        result = detector.analyze(str(save_path))

        # ── Persist to SQLite ─────────────────────────────────────────────
        analysis_id = save_analysis(
            filename                 = file.filename or unique_name,
            visual_score             = result["visual_score"],
            audio_score              = result["audio_score"],
            lip_sync_score           = result["lip_sync_score"],
            temporal_score           = result["temporal_score"],
            real_probability         = result["real_probability"],
            ai_generated_probability = result["ai_generated_probability"],
            prediction               = result["prediction"],
        )

        return JSONResponse({
            "id":                       analysis_id,
            "visual_score":             result["visual_score"],
            "audio_score":              result["audio_score"],
            "lip_sync_score":           result["lip_sync_score"],
            "temporal_score":           result["temporal_score"],
            "real_probability":         result["real_probability"],
            "ai_generated_probability": result["ai_generated_probability"],
            "prediction":               result["prediction"],
            "processing_time_seconds":  result.get("processing_time_seconds", 0),
            "filename":                 file.filename,
        })

    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"Analysis failed: {str(exc)}")
    finally:
        # Clean up uploaded file to save disk space
        try:
            if save_path.exists():
                os.unlink(save_path)
        except Exception:
            pass


# ── GET /history ──────────────────────────────────────────────────────────
@app.get("/history", tags=["History"])
def get_history():
    """Return all saved analyses, most recent first."""
    return get_all_analyses()


# ── GET /history/{id} ─────────────────────────────────────────────────────
@app.get("/history/{analysis_id}", tags=["History"])
def get_history_item(analysis_id: int):
    """Return a single analysis by ID."""
    record = get_analysis_by_id(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record


# ── DELETE /history/{id} ──────────────────────────────────────────────────
@app.delete("/history/{analysis_id}", tags=["History"])
def delete_history_item(analysis_id: int):
    """Delete a single analysis by ID."""
    deleted = delete_analysis(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"message": f"Analysis {analysis_id} deleted."}


# ── POST /history ─────────────────────────────────────────────────────────
@app.post("/history", tags=["History"])
def manual_save(payload: dict):
    """Manually save an analysis record (for testing)."""
    try:
        aid = save_analysis(
            filename                 = payload.get("filename", "unknown"),
            visual_score             = payload.get("visual_score", 0.5),
            audio_score              = payload.get("audio_score", 0.5),
            lip_sync_score           = payload.get("lip_sync_score", 0.5),
            temporal_score           = payload.get("temporal_score", 0.5),
            real_probability         = payload.get("real_probability", 0.5),
            ai_generated_probability = payload.get("ai_generated_probability", 0.5),
            prediction               = payload.get("prediction", "UNKNOWN"),
        )
        return {"id": aid, "message": "Saved successfully."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
