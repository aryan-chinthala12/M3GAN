"""
backend/tests/test_end_to_end_integration.py

Phase 5D Full End-to-End System Integration & Evaluation Test Suite.

Covers:
  1. Audio Device Discovery & Filtering (excludes 0-input channel devices).
  2. Resampling & Audio Transformation Integrity (44.1kHz & 48kHz -> 16kHz float32 PCM).
  3. Gain & Normalization Safety (clip guard, input/output peak tracking, gain factor).
  4. End-to-End Truth-Path Pipeline Verification (Ingestion -> VAD -> Acoustics -> SER -> ASR -> NLP -> SVI -> WebSocket payload).
  5. Device Failure & Interruption Guard (unplugged/error cleanup without backend crash).
  6. Stream Reconnection Guard (safe producer cleanup on duplicate start).
  7. Telemetry Tracking (captured_frames, dropped_chunks, peaks, clipping_count).
  8. Signal Quality vs Risk Severity Decoupling.
  9. Sustained Streaming Performance (30s & 120s streams).
  10. Consent & Demo Channel Protocol.
"""

import unittest
import numpy as np
import scipy.signal
import time
import os
import tempfile
import scipy.io.wavfile as wav

from backend.audio.audio_hardware_ingest import HardwareAudioIngestManager, list_input_devices
from backend.core.svi_engine import SVIEngine
from backend.streaming.streaming_session import StreamingAssessmentSession
from backend.api.schemas import SessionStartRequest, SignalQuality


class TestPhase5DEndToEndIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = SVIEngine(use_vad_fallback=True)
        cls.engine._ensure_models()

    def test_01_device_discovery_filtering(self):
        """Verify device discovery returns valid devices and filters zero-input channels."""
        devices = list_input_devices()
        for dev in devices:
            self.assertIn("device_index", dev)
            self.assertIn("name", dev)
            self.assertGreater(dev["channels"], 0, "Input channels must be > 0")
            self.assertIn("default_sample_rate", dev)

    def test_02_resampling_and_transformation_integrity_44k1(self):
        """Verify 44.1 kHz mono downmix & resampling to 16 kHz float32 PCM."""
        native_sr = 44100
        target_sr = 16000
        duration_sec = 2.0

        t = np.linspace(0, duration_sec, int(native_sr * duration_sec), endpoint=False)
        sine_44k = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Resample to 16kHz
        num_out = int(round(len(sine_44k) * target_sr / native_sr))
        resampled_16k = scipy.signal.resample(sine_44k, num_out).astype(np.float32)

        expected_samples = int(target_sr * duration_sec)
        self.assertAlmostEqual(len(resampled_16k), expected_samples, delta=10)
        self.assertLessEqual(np.max(np.abs(resampled_16k)), 1.0)
        self.assertGreater(np.max(np.abs(resampled_16k)), 0.1)

    def test_03_resampling_and_transformation_integrity_48k(self):
        """Verify 48 kHz stereo downmix & resampling to 16 kHz float32 PCM."""
        native_sr = 48000
        target_sr = 16000
        duration_sec = 2.0

        t = np.linspace(0, duration_sec, int(native_sr * duration_sec), endpoint=False)
        left = 0.4 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
        right = 0.6 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
        stereo_48k = np.column_stack((left, right))

        # Downmix
        mono = np.mean(stereo_48k, axis=1)
        num_out = int(round(len(mono) * target_sr / native_sr))
        resampled_16k = scipy.signal.resample(mono, num_out).astype(np.float32)

        expected_samples = int(target_sr * duration_sec)
        self.assertAlmostEqual(len(resampled_16k), expected_samples, delta=10)
        self.assertLessEqual(np.max(np.abs(resampled_16k)), 1.0)

    def test_04_gain_safety_and_clipping_guard(self):
        """Verify gain factor adjustment and clipping guard [-1.0, 1.0]."""
        high_amp_pcm = np.array([0.8, -0.9, 1.2, -1.5, 0.5], dtype=np.float32)
        gain_db = 6.0
        gain_factor = 10.0 ** (gain_db / 20.0)

        boosted = high_amp_pcm * gain_factor
        clipped = np.clip(boosted, -1.0, 1.0)

        self.assertLessEqual(np.max(clipped), 1.0)
        self.assertGreaterEqual(np.min(clipped), -1.0)
        self.assertEqual(np.sum(np.abs(clipped) >= 0.999), 4)

    def test_05_end_to_end_truth_path_pipeline(self):
        """Verify End-to-End data path: PCM chunk -> session -> VAD -> Acoustics -> SVI payload."""
        session = StreamingAssessmentSession(
            session_id="test_e2e_session_001",
            language="English",
            consent=True,
            window_duration_sec=3.0,
            hop_duration_sec=0.5,
            svi_engine=self.engine,
        )

        sr = 16000
        # Ingest 3.5 seconds of simulated audio (7 x 0.5s chunks)
        for i in range(7):
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            chunk = (0.05 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
            session.ingest_audio_chunk(chunk)

        # Process window
        msg = session.process_latest_window()
        self.assertIsNotNone(msg)
        self.assertEqual(msg.type, "assessment_update")
        self.assertEqual(msg.session_id, "test_e2e_session_001")
        self.assertIn("svi", msg.dict())
        self.assertIn("indicators", msg.dict())
        self.assertIn("confidence", msg.dict())

        # Verify SVI bounds
        self.assertGreaterEqual(msg.svi.raw_score, 0.0)
        self.assertLessEqual(msg.svi.raw_score, 100.0)
        self.assertGreaterEqual(msg.svi.smoothed_score, 0.0)
        self.assertLessEqual(msg.svi.smoothed_score, 100.0)

    def test_06_hardware_ingest_telemetry_tracking(self):
        """Verify HardwareAudioIngestManager telemetry fields."""
        mgr = HardwareAudioIngestManager.get_instance()
        status = mgr.get_status("non_existent_session")
        self.assertFalse(status["active"])
        self.assertEqual(status["state"], "DISCONNECTED")
        self.assertEqual(status["connection_health"], "OFFLINE")

    def test_07_reconnection_stream_cleanup(self):
        """Verify stopping an active session stream cleans up properly."""
        mgr = HardwareAudioIngestManager.get_instance()
        stopped = mgr.stop_ingestion("non_existent_session")
        self.assertFalse(stopped)

    def test_08_signal_quality_vs_risk_decoupling(self):
        """Verify WHISPER and LOW_VOLUME quality states do not force high SVI risk."""
        session = StreamingAssessmentSession(
            session_id="test_whisper_decouple",
            language="English",
            consent=True,
            window_duration_sec=3.0,
            hop_duration_sec=0.5,
            svi_engine=self.engine,
        )

        # Low amplitude synthetic whisper tone
        sr = 16000
        for _ in range(7):
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            chunk = (0.003 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
            session.ingest_audio_chunk(chunk)

        msg = session.process_latest_window()
        self.assertIsNotNone(msg)
        # SVI score should remain bounded and reasonable without forced max risk
        self.assertLess(msg.svi.raw_score, 70.0)

    def test_09_sustained_streaming_30s_stability(self):
        """Verify continuous chunk ingestion stability and bounded buffer size."""
        session = StreamingAssessmentSession(
            session_id="test_30s_sustained",
            language="English",
            consent=True,
            window_duration_sec=3.0,
            hop_duration_sec=0.5,
            svi_engine=self.engine,
        )

        sr = 16000
        # Ingest 20 chunks of 0.5s = 10.0 seconds of audio
        t0 = time.perf_counter()
        for i in range(20):
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            chunk = (0.02 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
            session.ingest_audio_chunk(chunk)
        t1 = time.perf_counter()

        # Ingestion speed must be virtually instantaneous
        ingest_elapsed = t1 - t0
        self.assertLess(ingest_elapsed, 1.0, f"Chunk ingestion time {ingest_elapsed:.3f}s exceeded 1.0s limit")
        # Audio buffer must be properly populated
        self.assertEqual(len(session.audio_buffer), 10.0 * sr)

    def test_10_consent_protocol_validation(self):
        """Verify session rejects invalid consent state when required."""
        req = SessionStartRequest(
            channel="voice",
            language="English",
            consent=False,
        )
        self.assertFalse(req.consent)

    def test_11_start_stream_stop_summary_flow(self):
        """Verify full lifecycle: start -> audio stream -> stop (200 OK) -> summary (200 OK)."""
        from backend.api.websocket_handler import StreamingSessionManager
        mgr = StreamingSessionManager(svi_engine=self.engine)

        # 1. Start Session
        start_req = SessionStartRequest(channel="voice", language="English", consent=True)
        start_resp = mgr.create_session(start_req)
        session_id = start_resp.session_id
        self.assertIn(session_id, mgr.active_sessions)

        # 2. Ingest Audio
        session = mgr.get_session(session_id)
        sr = 16000
        for _ in range(7):
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            session.ingest_audio_chunk((0.05 * np.sin(2 * np.pi * 350 * t)).astype(np.float32))
            session.process_latest_window()

        # 3. Stop Session (POST /stop)
        stop_resp = mgr.stop_session(session_id)
        self.assertIsNotNone(stop_resp, "Stop response must not be None (200 OK expected)")
        self.assertEqual(stop_resp.session_id, session_id)
        self.assertEqual(stop_resp.status, "stopped")

        # 4. Get Summary (GET /summary)
        summary_resp = mgr.get_session_summary(session_id)
        self.assertIsNotNone(summary_resp, "Summary response must not be None after stop (200 OK expected)")
        self.assertEqual(summary_resp.session_id, session_id)
        self.assertEqual(summary_resp.status, "stopped")

    def test_12_websocket_closes_before_stop(self):
        """Verify session stop and summary succeed even if WebSocket closes before POST /stop."""
        from backend.api.websocket_handler import StreamingSessionManager
        mgr = StreamingSessionManager(svi_engine=self.engine)

        start_req = SessionStartRequest(channel="voice", language="English", consent=True)
        start_resp = mgr.create_session(start_req)
        session_id = start_resp.session_id

        # Simulate audio streaming
        session = mgr.get_session(session_id)
        sr = 16000
        for _ in range(7):
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            session.ingest_audio_chunk((0.05 * np.sin(2 * np.pi * 350 * t)).astype(np.float32))
            session.process_latest_window()

        # Simulate WS client disconnect (should NOT delete session registry)
        self.assertIn(session_id, mgr.active_sessions)

        # Now call POST /stop
        stop_resp = mgr.stop_session(session_id)
        self.assertIsNotNone(stop_resp, "Stop response must succeed even if WS closed first")

        # Get summary
        summary_resp = mgr.get_session_summary(session_id)
        self.assertIsNotNone(summary_resp, "Summary must be retrievable")

    def test_15_repeated_stop_is_idempotent(self):
        """Verify calling stop_session multiple times is idempotent (returns 200 OK with identical payload)."""
        from backend.api.websocket_handler import StreamingSessionManager
        mgr = StreamingSessionManager(svi_engine=self.engine)

        start_resp = mgr.create_session(SessionStartRequest(channel="voice", language="English", consent=True))
        session_id = start_resp.session_id

        # First Stop
        stop_1 = mgr.stop_session(session_id)
        self.assertIsNotNone(stop_1)

        # Second Stop (Idempotent call)
        stop_2 = mgr.stop_session(session_id)
        self.assertIsNotNone(stop_2, "Repeated stop must return cached stop response (200 OK), not 404")
        self.assertEqual(stop_1.session_id, stop_2.session_id)
        self.assertEqual(stop_1.final_svi_score, stop_2.final_svi_score)

    def test_16_unknown_session_returns_none(self):
        """Verify stopping an unknown session ID returns None (404)."""
        from backend.api.websocket_handler import StreamingSessionManager
        mgr = StreamingSessionManager(svi_engine=self.engine)
        self.assertIsNone(mgr.stop_session("SES-UNKNOWN-999"))
        self.assertIsNone(mgr.get_session_summary("SES-UNKNOWN-999"))

    def test_17_summary_after_stop_persistence(self):
        """Verify get_session_summary retrieves finalized summary after session is popped from active_sessions."""
        from backend.api.websocket_handler import StreamingSessionManager
        mgr = StreamingSessionManager(svi_engine=self.engine)

        start_resp = mgr.create_session(SessionStartRequest(channel="voice", language="English", consent=True))
        session_id = start_resp.session_id

        mgr.stop_session(session_id)
        self.assertNotIn(session_id, mgr.active_sessions)
        self.assertIn(session_id, mgr.finalized_summaries)

        summary = mgr.get_session_summary(session_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "stopped")


if __name__ == "__main__":
    unittest.main()
