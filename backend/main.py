"""
backend/main.py

FastAPI application for the NHAA (14566) AI Stress & Trauma Assessment Module.

Exposes:
  - Batch Audio Analysis: POST /api/v1/analyze-audio (100% backwards compatible fallback)
  - Batch Text Analysis: POST /api/v1/analyze-text
  - Cases & Stats: GET /api/v1/cases, GET /api/v1/stats, GET /api/v1/cases/{case_id}
  - Human Interventions: POST /api/v1/interventions/respond
  - Real-Time Session Lifecycle:
      POST /api/v1/session/start
      POST /api/v1/session/{session_id}/stop
      GET  /api/v1/session/{session_id}/summary
  - Real-Time WebSocket Streaming: WS /ws/session/{session_id}/stream

Privacy & Ethics:
  - PII-redacted before storage/response.
  - Informed consent persisted.
  - Decision-support only (human oversight required).
"""

import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import (
    AudioAnalysisResponse,
    TextAnalysisRequest,
    TextAnalysisResponse,
    InterventionRequest,
    InterventionResponse,
    CaseSummary,
    StatsResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionStopResponse,
    SessionSummaryResponse,
)
from backend.core.svi_engine import SVIEngine
from backend.api.websocket_handler import StreamingSessionManager, handle_websocket_stream

# Global Engine & Session Manager Instances
svi_engine: Optional[SVIEngine] = None
session_manager: Optional[StreamingSessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svi_engine, session_manager
    print("[SYSTEM] Starting NHAA SVI Assessment Module...")
    svi_engine = SVIEngine()
    session_manager = StreamingSessionManager(svi_engine=svi_engine)
    print("[SYSTEM] Backend live. Text & WebSocket channels ready; audio models load on demand.")
    yield
    svi_engine = None
    session_manager = None


app = FastAPI(
    title="NHAA (14566) - AI Stress & Trauma Assessment Module",
    description=(
        "Multimodal Stress Vulnerability Index engine: speech emotion + "
        "acoustic biomarkers + multilingual distress NLP for real-time voice streams, "
        "chat, portal, and chatbot channels."
    ),
    version="3.1.0",
    lifespan=lifespan,
)

# CORS for React dashboard / portal integrations
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
        "version": "3.1.0",
        "docs": "/docs",
        "channels": ["voice_stream", "voice_batch", "chat", "portal", "chatbot", "ivrs"],
    }


@app.get("/health")
def health_check():
    if svi_engine is None:
        return {"status": "initializing", "engine_ready": False}

    active_sessions_count = len(session_manager.active_sessions) if session_manager else 0

    return {
        "status": "healthy",
        "engine_ready": True,
        "models": {
            "speech_emotion": svi_engine._models_loaded,
            "whisper": svi_engine._models_loaded,
        },
        "streaming_sessions_active": active_sessions_count,
        "case_store": "connected",
    }


# =============================================================
# REAL-TIME STREAMING SESSION LIFECYCLE & WEBSOCKET
# =============================================================

@app.post("/api/v1/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new real-time streaming assessment session."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session Manager is not initialized")
    return session_manager.create_session(request)


@app.post("/api/v1/session/{session_id}/stop", response_model=SessionStopResponse)
async def stop_session(session_id: str):
    """Stop an active streaming session, persist final case record, and clean up."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session Manager is not initialized")

    res = session_manager.stop_session(session_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or already stopped.")
    return res


@app.get("/api/v1/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str):
    """Get real-time timeline summary and indicators for an active or recent session."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session Manager is not initialized")

    res = session_manager.get_session_summary(session_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return res


@app.websocket("/ws/session/{session_id}/stream")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """Real-time WebSocket endpoint for continuous audio streaming."""
    if session_manager is None:
        await websocket.close(code=1011, reason="Session Manager not initialized")
        return
    await handle_websocket_stream(websocket, session_id, session_manager)


# =============================================================
# BATCH AUDIO ANALYSIS (100% Backwards Compatible Fallback)
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


# =============================================================
# HARDWARE / PHONE / USB LINE INGESTION (PHASE 5A)
# =============================================================

from backend.audio.audio_hardware_ingest import HardwareAudioIngestManager, list_input_devices


@app.get("/api/v1/hardware/devices")
async def get_hardware_devices():
    """List host system audio input devices (USB headsets, soundcards, virtual audio cables)."""
    return {"devices": list_input_devices()}


@app.post("/api/v1/hardware/start-ingest")
async def start_hardware_ingestion(payload: dict):
    """Start hardware soundcard audio ingestion for a session."""
    session_id = payload.get("session_id")
    device_index = payload.get("device_index")
    gain_db = float(payload.get("gain_db", 0.0))

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    session_obj = session_manager.get_session(session_id)
    if not session_obj:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        ingest_mgr = HardwareAudioIngestManager.get_instance()

        # Thread-safe broadcaster callback forwarding hardware window updates to active WebSocket
        def websocket_broadcaster(msg_dict: dict):
            if session_manager is not None:
                session_manager.broadcast_to_websocket(session_id, msg_dict)

        success = ingest_mgr.start_ingestion(
            session_id=session_id,
            session_obj=session_obj,
            device_index=device_index,
            gain_db=gain_db,
            websocket_broadcaster=websocket_broadcaster,
        )
        return {
            "status": "started",
            "session_id": session_id,
            "device_index": device_index,
            "success": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start hardware ingestion: {str(e)}")


@app.post("/api/v1/hardware/stop-ingest")
async def stop_hardware_ingestion(payload: dict):
    """Stop active hardware soundcard audio ingestion for a session."""
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    ingest_mgr = HardwareAudioIngestManager.get_instance()
    stopped = ingest_mgr.stop_ingestion(session_id)
    return {"status": "stopped", "session_id": session_id, "stopped": stopped}


@app.get("/api/v1/hardware/status/{session_id}")
async def get_hardware_ingestion_status(session_id: str):
    """Get status of hardware ingestion stream for a session."""
    ingest_mgr = HardwareAudioIngestManager.get_instance()
    return ingest_mgr.get_status(session_id)


@app.get("/api/v1/hardware/telemetry/{session_id}")
async def get_hardware_telemetry(session_id: str):
    """Get diagnostic telemetry for an active or past hardware ingestion stream."""
    ingest_mgr = HardwareAudioIngestManager.get_instance()
    status = ingest_mgr.get_status(session_id)
    if not status.get("active", False):
        return {
            "session_id": session_id,
            "active": False,
            "status": "DISCONNECTED",
            "telemetry": None,
        }
    return {
        "session_id": session_id,
        "active": True,
        "status": status.get("state", "RECORDING"),
        "telemetry": status,
    }


