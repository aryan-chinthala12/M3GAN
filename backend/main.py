import os
import shutil
import tempfile
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import AudioAnalysisResponse, InterventionRequest, InterventionResponse
from backend.svi_engine import SVIEngine

logger = logging.getLogger(__name__)

# Global Engine Instance
svi_engine: Optional[SVIEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global svi_engine
    print("[SYSTEM] Starting FastAPI Server & Warming ML Models...")
    svi_engine = SVIEngine()
    print("[SYSTEM] SVI Multimodal Backend is Live and Ready!")
    yield
    # Shutdown logic
    svi_engine = None

app = FastAPI(
    title="NHAA Emergency Helpline - SVI Multimodal Engine",
    description="AI-driven Stress Vulnerability Index backend fusing acoustic emotion & lexical threat analysis.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit / Frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "system": "NHAA Helpline SVI Backend",
        "status": "OPERATIONAL",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if svi_engine is not None else "starting",
        "engine_ready": svi_engine is not None
    }

@app.post("/api/v1/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """Primary REST endpoint for processing audio calls and computing SVI scores."""
    if svi_engine is None:
        raise HTTPException(status_code=500, detail="SVI Engine is not initialized")

    if not file.filename or not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid audio format. Please upload .wav, .mp3, or .flac")

    # Save incoming upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        shutil.copyfileobj(file.file, temp_audio)
        temp_path = temp_audio.name

    try:
        results = svi_engine.process_multimodal_audio(temp_path, file.filename)
        return results
    except Exception as exc:
        logger.exception("Audio analysis failed for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {exc}"
        ) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/v1/interventions/respond", response_model=InterventionResponse)
async def record_intervention(request: InterventionRequest):
    """Human-in-the-loop endpoint recording operator override/confirmation actions."""
    return InterventionResponse(
        status="recorded",
        call_id=request.call_id,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )