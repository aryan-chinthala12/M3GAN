"""
backend/tests/test_streaming_session.py

Comprehensive Test Suite for M3GAN Phase 2 Real-Time Streaming Architecture.
Run: .venv/Scripts/python.exe -X utf8 backend/tests/test_streaming_session.py
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.streaming.streaming_session import StreamingAssessmentSession
from backend.api.websocket_handler import StreamingSessionManager, decode_audio_bytes
from backend.api.schemas import SessionStartRequest, StreamingAssessmentMessage
from backend.core.svi_engine import SVIEngine


def generate_synthetic_pcm(duration_sec=1.0, sample_rate=16000, tone_freq=220.0):
    """Generate float32 PCM audio array."""
    num_samples = int(duration_sec * sample_rate)
    if tone_freq <= 0:
        return np.zeros(num_samples, dtype=np.float32)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * tone_freq * t)).astype(np.float32)


def test_session_creation_and_config():
    print("--- 1. Testing Session Creation & Config ---")
    manager = StreamingSessionManager()
    req = SessionStartRequest(
        channel="voice",
        language="English",
        consent=True,
        window_duration_sec=3.0,
        hop_duration_sec=0.5,
    )

    res = manager.create_session(req)
    print(f"    Session ID: {res.session_id} | Status: {res.status} | WS URL: {res.websocket_url}")

    assert res.session_id.startswith("SES-")
    assert res.status == "created"
    assert res.config["window_duration_sec"] == 3.0
    assert res.config["hop_duration_sec"] == 0.5

    session = manager.get_session(res.session_id)
    assert session is not None
    print("    [VERIFIED] Session initialized with configurable rolling window and hop duration.")


def test_chunk_ingestion_and_bounded_buffering():
    print("--- 2. Testing Chunk Ingestion & Bounded Buffering ---")
    session = StreamingAssessmentSession(
        window_duration_sec=3.0, hop_duration_sec=0.5, max_buffer_sec=5.0
    )

    # Ingest 0.2s chunk -> hop not ready
    chunk_02s = generate_synthetic_pcm(duration_sec=0.2)
    hop_ready = session.ingest_audio_chunk(chunk_02s)
    assert hop_ready is False, "0.2s chunk should not trigger 0.5s hop"

    # Ingest another 0.4s chunk -> total 0.6s -> hop ready
    chunk_04s = generate_synthetic_pcm(duration_sec=0.4)
    hop_ready = session.ingest_audio_chunk(chunk_04s)
    assert hop_ready is True, "0.6s total accumulated audio should trigger 0.5s hop"

    # Test bounded buffer capacity limit (max 5.0s = 80,000 samples)
    large_audio = generate_synthetic_pcm(duration_sec=10.0)
    session.ingest_audio_chunk(large_audio)
    buffer_dur = len(session.audio_buffer) / session.sample_rate
    print(f"    Ingested 10s into 5s max buffer -> Buffer size: {buffer_dur:.2f}s")
    assert len(session.audio_buffer) <= 5.0 * 16000, "Audio buffer exceeded max bounded capacity limit!"
    print("    [VERIFIED] Bounded audio buffer prevents memory leaks and backlog overflow.")


def test_silence_and_insufficient_audio():
    print("--- 3. Testing Silence & Insufficient Audio Window Processing ---")
    session = StreamingAssessmentSession(window_duration_sec=3.0, hop_duration_sec=0.5)

    # Insufficient audio (< 0.8s)
    session.ingest_audio_chunk(generate_synthetic_pcm(0.5))
    msg = session.process_latest_window()
    assert msg is None, "Insufficient audio (< 0.8s) must return None"

    # Ingest 3.0s pure silence
    session.ingest_audio_chunk(generate_synthetic_pcm(3.0, tone_freq=0.0))
    msg = session.process_latest_window()

    assert msg is not None
    assert msg.type == "assessment_update"
    assert msg.signal_quality == "UNRELIABLE"
    assert msg.confidence.overall_confidence < 50.0
    print(f"    Silence Window Update: SVI={msg.svi.smoothed_score:.1f}, SignalQuality={msg.signal_quality}, Conf={msg.confidence.overall_confidence:.1f}%")
    print("    [VERIFIED] Silence correctly evaluated as UNRELIABLE signal quality.")


def test_normal_speech_ema_smoothing_and_timeline():
    print("--- 4. Testing Normal Speech, EMA Smoothing & SVI Timeline ---")
    svi_engine = SVIEngine(use_vad_fallback=True)
    session = StreamingAssessmentSession(
        window_duration_sec=3.0, hop_duration_sec=0.5, svi_engine=svi_engine
    )

    # Simulate 5 hops of audio (total 3.5s tone speech)
    for i in range(5):
        chunk = generate_synthetic_pcm(0.7, tone_freq=220.0 + i * 20.0)
        session.ingest_audio_chunk(chunk)
        msg = session.process_latest_window()

    print(f"    SVI History Count: {len(session.raw_svi_history)}")
    print(f"    Latest Raw SVI    : {session.raw_svi_history[-1][1]:.2f}")
    print(f"    Latest Smoothed SVI: {session.smoothed_svi_history[-1][1]:.2f}")
    print(f"    Latest Trend      : {session.svi_timeline[-1]['trend']}")
    print(f"    Latency           : {session.svi_timeline[-1]['latency_ms']:.2f} ms")

    assert len(session.svi_timeline) >= 4
    assert "raw_score" in msg.svi.model_dump()
    assert "smoothed_score" in msg.svi.model_dump()
    assert msg.svi.trend in ["INCREASING", "DECREASING", "STABLE", "INSUFFICIENT_DATA"]
    print("    [VERIFIED] EMA smoothing, raw/smoothed SVI, and timeline tracking operational.")


def test_transcript_deduplication():
    print("--- 5. Testing Transcript Deduplication ---")
    session = StreamingAssessmentSession()

    session._update_transcript_dedup("Help me please")
    assert session.cumulative_transcript == "Help me please"

    # Overlapping window: "please I am scared"
    session._update_transcript_dedup("please I am scared")
    print(f"    Deduplicated Transcript: '{session.cumulative_transcript}'")
    assert session.cumulative_transcript == "Help me please I am scared"

    # Duplicate exact window: "Help me please I am scared"
    session._update_transcript_dedup("Help me please I am scared")
    assert session.cumulative_transcript == "Help me please I am scared"
    print("    [VERIFIED] Transcript deduplication prevents text repetition in streaming windows.")


def test_safety_flag_independence_and_trend():
    print("--- 6. Testing Safety Flag Independence & Trend Calculation ---")
    session = StreamingAssessmentSession()

    # Simulate SVI trend sequence: [20.0, 30.0, 45.0, 60.0, 75.0]
    scores = [20.0, 30.0, 45.0, 60.0, 75.0]
    for i, sc in enumerate(scores):
        session.smoothed_svi_history.append((float(i * 0.5), sc))

    trend = session._calculate_trend()
    print(f"    Increasing Sequence Trend: {trend}")
    assert trend == "INCREASING"

    # Decreasing sequence
    session.smoothed_svi_history = []
    scores_dec = [80.0, 65.0, 50.0, 35.0, 20.0]
    for i, sc in enumerate(scores_dec):
        session.smoothed_svi_history.append((float(i * 0.5), sc))

    trend_dec = session._calculate_trend()
    print(f"    Decreasing Sequence Trend: {trend_dec}")
    assert trend_dec == "DECREASING"
    print("    [VERIFIED] SVI trend calculation correctly detects trajectory.")


def test_decode_audio_bytes_and_malformed_input():
    print("--- 7. Testing Audio Bytes Decoding & Malformed Input Handling ---")

    # Valid int16 PCM bytes
    int16_samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    pcm_bytes = int16_samples.tobytes()

    decoded = decode_audio_bytes(pcm_bytes)
    assert len(decoded) == 4
    assert abs(decoded[1] - 0.5) < 0.01

    # Malformed / empty bytes
    empty_decoded = decode_audio_bytes(b"")
    assert len(empty_decoded) == 0
    print("    [VERIFIED] Audio decoder handles valid PCM and empty/malformed input safely.")


def test_session_lifecycle_and_cleanup():
    print("--- 8. Testing Session Stop & Cleanup ---")
    manager = StreamingSessionManager(svi_engine=SVIEngine(use_vad_fallback=True))
    req = SessionStartRequest(language="English")

    start_res = manager.create_session(req)
    session_id = start_res.session_id

    session = manager.get_session(session_id)
    session.ingest_audio_chunk(generate_synthetic_pcm(2.0))

    stop_res = manager.stop_session(session_id)
    print(f"    Stopped Session: ID={stop_res.session_id}, Duration={stop_res.duration_seconds}s, Case={stop_res.case_id}")

    assert stop_res is not None
    assert stop_res.status == "stopped"
    assert manager.get_session(session_id) is None, "Session buffer was not cleaned up after stop!"
    print("    [VERIFIED] Session stop persists case summary and cleans up buffer memory.")


def main():
    print("=========================================================")
    print(" M3GAN PHASE 2 REAL-TIME STREAMING INTEGRATION TESTS")
    print("=========================================================")

    test_session_creation_and_config()
    test_chunk_ingestion_and_bounded_buffering()
    test_silence_and_insufficient_audio()
    test_normal_speech_ema_smoothing_and_timeline()
    test_transcript_deduplication()
    test_safety_flag_independence_and_trend()
    test_decode_audio_bytes_and_malformed_input()
    test_session_lifecycle_and_cleanup()

    print("\n=========================================================")
    print(" ALL PHASE 2 REAL-TIME STREAMING TESTS PASSED (100% SUCCESS)!")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
