"""
backend/websocket_handler.py

WebSocket Protocol Handler for M3GAN Real-Time Streaming Assessment.

Responsibilities:
  - Manages active WebSocket connection lifecycle.
  - Ingests binary PCM audio frames (16kHz Int16/Float32) or JSON Base64 payloads.
  - Executes window processing off the main event loop via asyncio.to_thread.
  - Broadcasts versioned structured JSON updates to client.
  - Handles client disconnects, timeouts, and errors gracefully with session cleanup.
"""

import asyncio
import base64
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect

from backend.streaming.streaming_session import StreamingAssessmentSession
from backend.api.schemas import (
    SessionStartRequest,
    SessionStartResponse,
    SessionStopResponse,
    SessionSummaryResponse,
    StreamingAssessmentMessage,
    StreamingErrorMessage,
)


class StreamingSessionManager:
    """In-memory session registry and lifetime manager."""

    def __init__(self, svi_engine: Optional[Any] = None):
        self.svi_engine = svi_engine
        self.active_sessions: Dict[str, StreamingAssessmentSession] = {}
        self.finalized_stops: Dict[str, SessionStopResponse] = {}
        self.finalized_summaries: Dict[str, SessionSummaryResponse] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.active_websockets: Dict[str, WebSocket] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def register_websocket(self, session_id: str, websocket: WebSocket):
        """Register active WebSocket connection and capture running event loop."""
        self.active_websockets[session_id] = websocket
        try:
            self.loop = asyncio.get_running_loop()
        except Exception:
            pass

    def unregister_websocket(self, session_id: str):
        """Unregister active WebSocket connection."""
        self.active_websockets.pop(session_id, None)

    def broadcast_to_websocket(self, session_id: str, message_dict: dict):
        """Thread-safe WebSocket broadcaster for external ingestion threads."""
        websocket = self.active_websockets.get(session_id)
        if websocket is None:
            return

        loop = self.loop
        if loop is None or not loop.is_running():
            return

        async def _send():
            try:
                json_text = json.dumps(message_dict)
                await websocket.send_text(json_text)
            except Exception as ex:
                print(f"[WEBSOCKET BROADCAST ERROR] {session_id}: {ex}")

        asyncio.run_coroutine_threadsafe(_send(), loop)

    def create_session(
        self,
        request: SessionStartRequest,
    ) -> SessionStartResponse:
        """Create a new streaming session."""
        session = StreamingAssessmentSession(
            language=request.language,
            consent=request.consent,
            window_duration_sec=request.window_duration_sec,
            hop_duration_sec=request.hop_duration_sec,
            svi_engine=self.svi_engine,
        )

        self.active_sessions[session.session_id] = session
        self.session_locks[session.session_id] = asyncio.Lock()

        print(f"[SESSION CREATE] session_id={session.session_id} (Language: {session.language}) active_keys={list(self.active_sessions.keys())}")

        return SessionStartResponse(
            session_id=session.session_id,
            status="created",
            created_at=session.created_at,
            websocket_url=f"/ws/session/{session.session_id}/stream",
            config={
                "language": session.language,
                "window_duration_sec": session.window_duration_sec,
                "hop_duration_sec": session.hop_duration_sec,
                "sample_rate": session.sample_rate,
            },
        )

    def get_session(self, session_id: str) -> Optional[StreamingAssessmentSession]:
        return self.active_sessions.get(session_id)

    def stop_session(self, session_id: str) -> Optional[SessionStopResponse]:
        """Stop session, persist final state, and preserve summary in registry."""
        print(f"[STOP ENDPOINT] Requested session_id={session_id} active_keys={list(self.active_sessions.keys())} finalized_keys={list(self.finalized_stops.keys())}")

        # 1. Idempotent check: return preserved stop response if already finalized
        if session_id in self.finalized_stops:
            print(f"[STOP ENDPOINT] Session {session_id} already finalized, returning cached stop response.")
            return self.finalized_stops[session_id]

        # 2. Finalize active session
        session = self.active_sessions.pop(session_id, None)
        self.session_locks.pop(session_id, None)

        if session is None:
            print(f"[STOP ENDPOINT] 404 Session {session_id} not found in active or finalized registries.")
            return None

        final_svi = session.last_smoothed_svi
        final_risk = session._get_risk_band(final_svi)
        duration = round(len(session.audio_buffer) / session.sample_rate, 2)

        case_id = "NH-NOAUDIO"
        if duration >= 0.5 and self.svi_engine and self.svi_engine.case_store:
            risk_color = "#2E7D32"
            from backend.core.config import RISK_RULES
            if final_risk in RISK_RULES:
                risk_color = RISK_RULES[final_risk]["color"]

            case_id = self.svi_engine.case_store.create_case(
                channel="voice",
                language=session.language,
                svi_score=final_svi,
                risk_band=final_risk,
                risk_color=risk_color,
                status="logged",
                text=session.cumulative_transcript or "[LIVE VOICE STREAM]",
            )

        from backend.core.config import RISK_RULES
        actions = RISK_RULES.get(final_risk, {}).get("actions", [])

        stop_resp = SessionStopResponse(
            session_id=session_id,
            case_id=case_id,
            status="stopped",
            duration_seconds=duration,
            final_svi_score=final_svi,
            final_risk_band=final_risk,
            final_transcript=session.cumulative_transcript or "[NO SPEECH DETECTED]",
            safety_flags=session.safety_flags,
            created_at=session.created_at,
        )

        summary_resp = SessionSummaryResponse(
            session_id=session_id,
            status="stopped",
            duration_seconds=duration,
            created_at=session.created_at,
            final_svi_score=final_svi,
            final_risk_band=final_risk,
            final_transcript=session.cumulative_transcript or "[NO SPEECH DETECTED]",
            svi_timeline=session.svi_timeline,
            risk_history=[{"time_sec": t, "risk_band": r} for t, r in session.risk_history],
            safety_flags=session.safety_flags,
            confidence_metrics=session.confidence_metrics,
            recommended_interventions=actions,
        )

        # Store in finalized registries
        self.finalized_stops[session_id] = stop_resp
        self.finalized_summaries[session_id] = summary_resp

        print(f"[STOP ENDPOINT] Successfully finalized session {session_id} (Case: {case_id}, SVI: {final_svi:.1f})")
        return stop_resp

    def get_session_summary(self, session_id: str) -> Optional[SessionSummaryResponse]:
        """Get summary payload for an active or finalized session."""
        print(f"[SUMMARY ENDPOINT] Requested session_id={session_id} active_keys={list(self.active_sessions.keys())} finalized_keys={list(self.finalized_summaries.keys())}")

        # 1. Check finalized summaries registry first
        if session_id in self.finalized_summaries:
            print(f"[SUMMARY ENDPOINT] Returning finalized summary for {session_id}.")
            return self.finalized_summaries[session_id]

        # 2. Check active sessions if streaming is ongoing
        session = self.active_sessions.get(session_id)
        if session is None:
            print(f"[SUMMARY ENDPOINT] 404 Session {session_id} not found.")
            return None

        duration = round(len(session.audio_buffer) / session.sample_rate, 2)
        final_svi = session.last_smoothed_svi
        final_risk = session._get_risk_band(final_svi)

        from backend.config import RISK_RULES
        actions = RISK_RULES.get(final_risk, {}).get("actions", [])

        return SessionSummaryResponse(
            session_id=session_id,
            status="active",
            duration_seconds=duration,
            created_at=session.created_at,
            final_svi_score=final_svi,
            final_risk_band=final_risk,
            final_transcript=session.cumulative_transcript or "[NO SPEECH DETECTED]",
            svi_timeline=session.svi_timeline,
            risk_history=[{"time_sec": t, "risk_band": r} for t, r in session.risk_history],
            safety_flags=session.safety_flags,
            confidence_metrics=session.confidence_metrics,
            recommended_interventions=actions,
        )


def decode_audio_bytes(data: bytes) -> np.ndarray:
    """Convert raw binary PCM bytes (16-bit signed PCM or Float32) to float32 numpy array."""
    if not data:
        return np.array([], dtype=np.float32)

    try:
        # Check if int16 PCM
        int16_arr = np.frombuffer(data, dtype=np.int16)
        float32_arr = int16_arr.astype(np.float32) / 32768.0
        return float32_arr
    except Exception:
        try:
            return np.frombuffer(data, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)


async def handle_websocket_stream(
    websocket: WebSocket,
    session_id: str,
    session_manager: StreamingSessionManager,
):
    """
    WebSocket streaming endpoint handler.
    Exchanges versioned JSON payloads and non-blocking streaming updates.
    """
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if session is None:
        err = StreamingErrorMessage(
            type="error",
            version=1,
            session_id=session_id,
            code="SESSION_NOT_FOUND",
            message=f"Streaming session '{session_id}' does not exist or has expired.",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        await websocket.send_text(err.model_dump_json())
        await websocket.close(code=4004)
        return

    session_manager.register_websocket(session_id, websocket)
    print(f"[WEBSOCKET CONNECT] Connected stream client for session {session_id} (active_keys={list(session_manager.active_sessions.keys())})")

    # Send session_started handshake ack
    start_msg = {
        "type": "session_started",
        "version": 1,
        "session_id": session_id,
        "config": {
            "language": session.language,
            "window_duration_sec": session.window_duration_sec,
            "hop_duration_sec": session.hop_duration_sec,
            "sample_rate": session.sample_rate,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    await websocket.send_text(json.dumps(start_msg))

    try:
        while True:
            # Receive text (JSON base64 or commands) or binary PCM frames
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                chunk = decode_audio_bytes(message["bytes"])
            elif "text" in message and message["text"]:
                raw_text = message["text"]

                try:
                    payload = json.loads(raw_text)
                    cmd_type = payload.get("type", "")

                    if cmd_type == "stop":
                        break
                    elif "audio_base64" in payload:
                        raw_bytes = base64.b64decode(payload["audio_base64"])
                        chunk = decode_audio_bytes(raw_bytes)
                    else:
                        chunk = np.array([], dtype=np.float32)
                except Exception:
                    chunk = np.array([], dtype=np.float32)
            else:
                chunk = np.array([], dtype=np.float32)

            if len(chunk) > 0:
                print(f"[AUDIO PROCESSING] Processing audio chunk for session {session_id} (samples={len(chunk)})")
                should_hop = session.ingest_audio_chunk(chunk)

                if should_hop:
                    # Run CPU-heavy window processing in worker thread (non-blocking)
                    update_msg = await asyncio.to_thread(session.process_latest_window)

                    if update_msg is not None:
                        await websocket.send_text(update_msg.model_dump_json())

    except WebSocketDisconnect:
        print(f"[WEBSOCKET DISCONNECT] Client disconnected from session {session_id}")
    except Exception as ex:
        print(f"[WEBSOCKET DISCONNECT] Error in session stream {session_id}: {ex}")
        try:
            err = StreamingErrorMessage(
                type="error",
                version=1,
                session_id=session_id,
                code="PROCESSING_ERROR",
                message=f"Streaming error: {str(ex)}",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            await websocket.send_text(err.model_dump_json())
        except Exception:
            pass
    finally:
        session_manager.unregister_websocket(session_id)
        # Clean up hardware soundcard ingestion if active, but DO NOT delete session registry!
        try:
            from backend.audio_hardware_ingest import HardwareAudioIngestManager
            HardwareAudioIngestManager.get_instance().stop_ingestion(session_id)
        except Exception:
            pass
        print(f"[WEBSOCKET DISCONNECT] Session {session_id} WebSocket closed cleanly without destroying session registry (active_keys={list(session_manager.active_sessions.keys())}).")


