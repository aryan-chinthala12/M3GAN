from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile

from backend.audio_processor import extract_acoustic_features
from backend.svi_engine import calculate_ml_svi

app = FastAPI(title="NHAA ML-Powered SVI Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    model_status = os.path.exists(os.path.join("backend", "emotion_model.pkl"))
    return {
        "status": "healthy",
        "engine": "NHAA-ML-Acoustic-SVI-v2.0",
        "model_loaded": model_status
    }

@app.post("/api/v1/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    filename = str(file.filename or "")
    if not filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload WAV, MP3, or M4A."
        )

    tmp_path = None
    try:
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        metrics = extract_acoustic_features(tmp_path)
        svi_results = calculate_ml_svi(tmp_path, metrics)

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {
            "status": "success",
            "filename": filename,
            "duration_seconds": metrics["duration_seconds"],
            "acoustic_metrics": metrics,
            "svi_result": svi_results
        }

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))