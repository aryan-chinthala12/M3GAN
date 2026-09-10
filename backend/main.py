"""
backend/main.py

FastAPI application for the NHAA (14566) AI Stress & Trauma Assessment
Module. Exposes audio + text analysis, case listing and aggregate stats.

Privacy / ethics:
  - Every narrative is PII-redacted BEFORE storage or response.
  - `consent` flag is persisted with each case (informed consent).
  - Assessments are decision-support only; human oversight is required
    (enforced by frontend copy and the /interventions endpoint).
"""

import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AudioAnalysisResponse,
    TextAnalysisRequest,
    TextAnalysisResponse,
    InterventionRequest,
    InterventionResponse,
    CaseSummary,
    StatsResponse,
)
from backend.svi_engine import SVIEngine

# Global Engine Instance
svi_engine: Optional[SVIEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svi_engine
    print("[SYSTEM] Starting NHAA SVI Assessment Module...")
    svi_engine = SVIEngine()
    print("[SYSTEM] Backend live. Text channels ready; audio models load on demand.")
    yield
    svi_engine = None


app = FastAPI(
    title="NHAA (14566) - AI Stress & Trauma Assessment Module",
    description=(
        "Multimodal Stress Vulnerability Index engine: speech emotion + "
        "acoustic biomarkers + multilingual distress NLP for voice, chat, "
        "portal and chatbot channels."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# CORS for the React dashboard / portal integrations
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
        "system": "NHAA Stress & Trauma Assessment Module",
        "status": "OPERATIONAL",
        "docs": "/docs",
        "channels": ["voice", "chat", "portal", "chatbot", "ivrs"],
    }


@app.get("/health")
def health_check():
    if svi_engine is None:
        return {"status": "initializing", "engine_ready": False}

    return {
        "status": "healthy",
        "engine_ready": True,
        "models": {
            "speech_emotion": svi_engine._models_loaded,
            "whisper": svi_engine._models_loaded,
        },
        "case_store": "connected",
    }


# =============================================================
# AUDIO ANALYSIS
# =============================================================

@app.post("/api/v1/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    language: str = Form("English"),
    consent: bool = Form(False),
):
    """Multimodal analysis of a voice call recording (SER + STT + SVI)."""
    if svi_engine is None:
        raise HTTPException(status_code=500, detail="SVI Engine is not initialized")

    allowed_extensions = (".wav", ".mp3", ".flac", ".m4a", ".webm", ".ogg")

    if not file.filename or not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Invalid audio format. Allowed: wav, mp3, flac, m4a, webm, ogg.",
        )

    extension = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_audio:
        shutil.copyfileobj(file.file, temp_audio)
        temp_path = temp_audio.name

    try:
        return svi_engine.process_multimodal_audio(
            temp_path,
            file.filename,
            language,
            consent,
        )
    except ValueError as validation_error:
        raise HTTPException(status_code=400, detail=str(validation_error))
    except Exception as processing_error:
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {processing_error}",
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =============================================================
# TEXT ANALYSIS (chat / portal / chatbot / IVRS transcript)
# =============================================================

@app.post("/api/v1/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze a written narrative from chat, portal, or chatbot channels."""
    if svi_engine is None:
        raise HTTPException(status_code=500, detail="SVI Engine is not initialized")

    try:
        return svi_engine.process_text(request)
    except ValueError as validation_error:
        raise HTTPException(status_code=400, detail=str(validation_error))
    except Exception as processing_error:
        raise HTTPException(
            status_code=500,
            detail=f"Text analysis failed: {processing_error}",
        )


# =============================================================
# CASES & STATS
# =============================================================

@app.get("/api/v1/cases", response_model=list[CaseSummary])
async def list_cases(limit: int = 100):
    """Recent assessments, newest first (dashboard / case queue)."""
    if svi_engine is None:
        raise HTTPException(status_code=500, detail="SVI Engine is not initialized")
    return svi_engine.list_cases(limit)


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Aggregate risk/channel statistics (dashboard cards)."""
    if svi_engine is None:
        raise HTTPException(status_code=500, detail="SVI Engine is not initialized")
    return svi_engine.get_stats()


@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: str):
    """Full case record including redacted narrative and components."""
    if svi_engine is None:
        raise HTTPException(status_code=500, detail="SVI Engine is not initialized")

    case = svi_engine.case_store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return case


# =============================================================
# HUMAN-IN-THE-LOOP
# =============================================================

@app.post("/api/v1/interventions/respond", response_model=InterventionResponse)
async def record_intervention(request: InterventionRequest):
    """
    Human-in-the-loop endpoint: operator acknowledges/overrides the
    AI recommendation. Updates case status and records the action.
    """
    if svi_engine is None:
        raise HTTPException(status_code=500, detail="SVI Engine is not initialized")

    svi_engine.case_store.update_status(request.call_id, request.action_taken)

    return InterventionResponse(
        status="recorded",
        call_id=request.call_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
