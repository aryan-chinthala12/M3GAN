"""
backend/streaming_session.py

Stateful Real-Time Streaming Assessment Session Manager.

Responsibilities:
  - Manages live audio buffer (16kHz float32 mono PCM) with bounded buffer limit.
  - Executes rolling window analysis (default 3.0s window, 0.5s hop).
  - Performs VAD, acoustic feature extraction, SER classification, ASR transcription, and lexical NLP.
  - Applies Exponential Moving Average (EMA) SVI smoothing.
  - Calculates real-time risk trend ('INCREASING', 'DECREASING', 'STABLE', 'INSUFFICIENT_DATA').
  - Deduplicates incremental transcript windows.
  - Measures end-to-end processing latency and realtime ratio.
"""

import time
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from backend.core.config import RISK_RULES, THREAT_LEVEL_MAP
from backend.audio.audio_processor import AudioProcessor
from backend.nlp.text_analyzer import analyze_transcript, redact, threat_level_from_band
from backend.api.schemas import (
    SafetyFlagItem,
    ConfidenceMetrics,
    StreamingSVI,
    StreamingAssessmentMessage,
)


class StreamingAssessmentSession:
    """Stateful assessment session processing continuous PCM audio streams."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        language: str = "English",
        consent: bool = True,
        window_duration_sec: float = 3.0,
        hop_duration_sec: float = 0.5,
        sample_rate: int = 16000,
        max_buffer_sec: float = 10.0,
        svi_engine: Any = None,
    ):
        self.session_id = session_id or f"SES-{uuid.uuid4().hex[:8].upper()}"
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.language = language
        self.consent = consent

        self.window_duration_sec = float(window_duration_sec)
        self.hop_duration_sec = float(hop_duration_sec)
        self.sample_rate = int(sample_rate)

        self.window_samples = int(self.window_duration_sec * self.sample_rate)
        self.hop_samples = int(self.hop_duration_sec * self.sample_rate)
        self.max_buffer_samples = int(max_buffer_sec * self.sample_rate)

        self.svi_engine = svi_engine

        # Audio PCM float32 buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_samples = 0
        self.last_hop_sample_count = 0

        # State tracking
        self.raw_svi_history: List[Tuple[float, float]] = []       # (time_sec, raw_score)
        self.smoothed_svi_history: List[Tuple[float, float]] = []  # (time_sec, smoothed_score)
        self.risk_history: List[Tuple[float, str]] = []            # (time_sec, risk_band)
        self.svi_timeline: List[Dict[str, Any]] = []

        self.cumulative_transcript = ""
        self.finalized_transcript = ""
        self.partial_transcript = ""

        self.latest_acoustic_metrics: Dict[str, Any] = {}
        self.latest_emotion_result: Dict[str, Any] = {"label": "NEUTRAL", "confidence": 0.0, "stress_score": 0.05}
        self.latest_nlp_result: Dict[str, Any] = {"lexical_score": 0.0, "hits": [], "categories": []}
        self.safety_flags: List[SafetyFlagItem] = []

        self.signal_quality = "GOOD"
        self.confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.0,
            confidence_rating="LOW",
            signal_quality="UNRELIABLE",
            speech_duration_seconds=0.0,
            voiced_ratio=0.0,
            pause_ratio=0.0,
            pitch_reliable=False,
        )

        self.last_smoothed_svi = 0.0
        self.ema_alpha = 0.35
        self.last_heavy_inference_time = 0.0
        self.heavy_inference_interval_sec = 1.0  # Schedule heavy ML every 1s to save CPU

    def ingest_audio_chunk(self, chunk: Any) -> bool:
        """
        Append PCM float32 16kHz mono samples (or raw 16-bit PCM bytes) to audio buffer.
        Enforces bounded buffer capacity (drops oldest samples if buffer exceeds max_buffer_sec).
        Returns True if a new hop analysis step should be triggered.
        """
        if chunk is None or len(chunk) == 0:
            return False

        if isinstance(chunk, (bytes, bytearray)):
            int16_samples = np.frombuffer(chunk, dtype=np.int16)
            chunk = int16_samples.astype(np.float32) / 32768.0

        if not isinstance(chunk, np.ndarray):
            chunk = np.array(chunk, dtype=np.float32)

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.processed_samples += len(chunk)

        # Bounded buffer capacity guardrail
        if len(self.audio_buffer) > self.max_buffer_samples:
            overflow = len(self.audio_buffer) - self.max_buffer_samples
            self.audio_buffer = self.audio_buffer[overflow:]

        # Check if enough samples accumulated for next hop
        samples_since_last_hop = len(self.audio_buffer) - self.last_hop_sample_count
        return samples_since_last_hop >= self.hop_samples

    def process_latest_window(self) -> Optional[StreamingAssessmentMessage]:
        """
        Processes the rolling analysis window over current audio buffer.
        Returns a structured versioned StreamingAssessmentMessage.
        """
        if len(self.audio_buffer) < int(0.8 * self.sample_rate):
            return None  # Insufficient audio for window analysis

        t_start = time.perf_counter()
        current_time_sec = round(len(self.audio_buffer) / self.sample_rate, 2)

        # Extract current rolling window audio slice (up to window_duration_sec)
        window_len = min(len(self.audio_buffer), self.window_samples)
        window_audio = self.audio_buffer[-window_len:]

        # ---------------------------------------------------------
        # 1. Fast Path: VAD & Acoustic Feature Normalization
        # ---------------------------------------------------------
        t_fast_start = time.perf_counter()
        audio_processor = getattr(self.svi_engine, "audio_processor", None) or AudioProcessor(use_vad_fallback=True)
        biomarkers = audio_processor.extract_voice_biomarkers(window_audio, sr=self.sample_rate)
        t_fast_end = time.perf_counter()
        fast_path_latency_ms = round((t_fast_end - t_fast_start) * 1000.0, 2)

        self.latest_acoustic_metrics = biomarkers
        self.signal_quality = biomarkers["signal_quality"]

        # ---------------------------------------------------------
        # 2. Heavy Path: SER, STT, NLP (Scheduled / Speech-triggered)
        # ---------------------------------------------------------
        now = time.time()
        should_run_heavy = (
            biomarkers.get("has_sufficient_speech", biomarkers.get("voiced_ratio", 0) >= 0.10)
            or (now - self.last_heavy_inference_time) >= self.heavy_inference_interval_sec
            or len(self.raw_svi_history) == 0
        )

        heavy_path_latency_ms = 0.0
        if should_run_heavy and self.svi_engine is not None and getattr(self.svi_engine, "_models_loaded", False):
            t_heavy_start = time.perf_counter()
            try:
                self._run_heavy_inference_window(window_audio)
                self.last_heavy_inference_time = now
            except Exception as ex:
                print(f"[WARNING] Heavy streaming inference error: {ex}")
            t_heavy_end = time.perf_counter()
            heavy_path_latency_ms = round((t_heavy_end - t_heavy_start) * 1000.0, 2)

        # ---------------------------------------------------------
        # 3. Multimodal SVI Fusion & EMA Smoothing
        # ---------------------------------------------------------
        acoustic_score = biomarkers["acoustic_score"]
        emotion_stress = self.latest_emotion_result["stress_score"]
        lexical_score = self.latest_nlp_result["lexical_score"]

        if self.svi_engine and hasattr(self.svi_engine, "get_fusion_weights"):
            w_ac, w_em, w_lex = self.svi_engine.get_fusion_weights(self.signal_quality)
        else:
            w_ac, w_em, w_lex = 0.30, 0.45, 0.25

        components = [
            ("acoustic", acoustic_score, w_ac, biomarkers["acoustic_detail"]),
            ("emotion", emotion_stress, w_em, f"Top emotion {self.latest_emotion_result['label']}"),
            ("lexical", lexical_score, w_lex, f"{len(self.latest_nlp_result['hits'])} lexical hits"),
        ]

        if self.svi_engine:
            raw_svi, safety_flags, override_trig, override_reason = self.svi_engine._fuse_with_safety_flags(
                components,
                self.latest_nlp_result["hits"],
                self.latest_nlp_result.get("floor_categories", []),
                self.latest_nlp_result.get("severity_floor", 0.0),
            )
        else:
            raw_svi = round((w_ac * acoustic_score + w_em * emotion_stress + w_lex * lexical_score) * 100.0, 2)
            safety_flags = []

        self.safety_flags = safety_flags

        # Apply Exponential Moving Average (EMA) smoothing
        if not self.smoothed_svi_history:
            smoothed_svi = raw_svi
        else:
            prev_smoothed = self.smoothed_svi_history[-1][1]
            smoothed_svi = round(self.ema_alpha * raw_svi + (1.0 - self.ema_alpha) * prev_smoothed, 2)

        self.last_smoothed_svi = smoothed_svi

        # Record histories
        self.raw_svi_history.append((current_time_sec, raw_svi))
        self.smoothed_svi_history.append((current_time_sec, smoothed_svi))

        risk_band = self._get_risk_band(smoothed_svi)
        self.risk_history.append((current_time_sec, risk_band))

        # ---------------------------------------------------------
        # 4. Trend Calculation
        # ---------------------------------------------------------
        trend = self._calculate_trend()

        # ---------------------------------------------------------
        # 5. Confidence Metrics
        # ---------------------------------------------------------
        qual_conf_map = {
            "GOOD": 100.0,
            "DEGRADED": 70.0,
            "LOW_SNR": 60.0,
            "LOW_VOLUME": 55.0,
            "WHISPER": 50.0,
            "UNRELIABLE": 10.0,
        }
        qual_conf = qual_conf_map.get(self.signal_quality, 60.0)
        emo_conf = float(self.latest_emotion_result["confidence"])
        text_words = len(self.cumulative_transcript.split())
        text_conf = min(100.0, text_words * 10.0) if text_words > 0 else 0.0

        overall_conf = round(0.40 * qual_conf + 0.35 * emo_conf + 0.25 * text_conf, 1)
        conf_rating = "HIGH" if overall_conf >= 75.0 else ("MEDIUM" if overall_conf >= 45.0 else "LOW")

        self.confidence_metrics = ConfidenceMetrics(
            overall_confidence=overall_conf,
            confidence_rating=conf_rating,
            signal_quality=self.signal_quality,
            speech_duration_seconds=biomarkers["speech_duration"],
            voiced_ratio=biomarkers["voiced_ratio"],
            pause_ratio=biomarkers["pause_ratio"],
            pitch_reliable=biomarkers["pitch_reliable"],
            snr_state=biomarkers["snr_state"],
            snr_db=biomarkers["snr_db"],
            provenance={"streaming_engine": "M3GAN Real-Time Session Processor"},
        )

        t_end = time.perf_counter()
        end_to_end_latency_ms = round((t_end - t_start) * 1000.0, 2)
        realtime_ratio = round(end_to_end_latency_ms / (self.hop_duration_sec * 1000.0), 3)

        self.last_hop_sample_count = len(self.audio_buffer)

        # Timeline entry
        timeline_entry = {
            "timestamp_sec": current_time_sec,
            "raw_svi": raw_svi,
            "smoothed_svi": smoothed_svi,
            "risk_band": risk_band,
            "trend": trend,
            "latency_ms": end_to_end_latency_ms,
            "fast_path_latency_ms": fast_path_latency_ms,
            "heavy_path_latency_ms": heavy_path_latency_ms,
        }
        self.svi_timeline.append(timeline_entry)

        # Build versioned response message
        return StreamingAssessmentMessage(
            type="assessment_update",
            version=1,
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            audio_duration=current_time_sec,
            transcript=self.cumulative_transcript or "[NO SPEECH DETECTED]",
            svi=StreamingSVI(
                raw_score=raw_svi,
                smoothed_score=smoothed_svi,
                risk_band=risk_band,
                risk_color=RISK_RULES[risk_band]["color"],
                trend=trend,
            ),
            confidence=self.confidence_metrics,
            signal_quality=self.signal_quality,
            emotion={
                "top_emotion": self.latest_emotion_result["label"],
                "confidence": self.latest_emotion_result["confidence"],
                "stress_score": emotion_stress,
            },
            acoustics={
                "pitch_volatility": biomarkers["pitch_volatility"],
                "rms_energy": biomarkers["rms_energy"],
                "energy_variation": biomarkers["energy_variation"],
                "pause_ratio": biomarkers["pause_ratio"],
                "voiced_ratio": biomarkers["voiced_ratio"],
                "pitch_reliable": biomarkers["pitch_reliable"],
            },
            indicators={
                "threat_level": threat_level_from_band(risk_band),
                "flagged_keywords": self.latest_nlp_result.get("flagged_keywords", []),
                "distress_categories": self.latest_nlp_result.get("categories", []),
            },
            safety_flags=self.safety_flags,
            fast_path_latency_ms=fast_path_latency_ms,
            heavy_path_latency_ms=heavy_path_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            latency_ms=end_to_end_latency_ms,
            realtime_ratio=realtime_ratio,
        )

    def _run_heavy_inference_window(self, window_audio: np.ndarray):
        """Run Whisper STT and Wav2Vec2 SER on current audio window."""
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, window_audio, self.sample_rate)
            tmp_path = tmp.name

        try:
            # 1. SER Prediction
            if hasattr(self.svi_engine, "_classify_emotion"):
                preds = self.svi_engine._classify_emotion(tmp_path)
                top_label = preds[0]["label"].upper()
                top_conf = round(preds[0]["score"] * 100.0, 1)
                stress_sc = self.svi_engine._emotion_stress_score(preds)
                self.latest_emotion_result = {
                    "label": top_label,
                    "confidence": top_conf,
                    "stress_score": stress_sc,
                }

            # 2. Whisper STT
            if hasattr(self.svi_engine, "whisper_model") and self.svi_engine.whisper_model:
                w_lang = self.svi_engine.language_map.get(self.language)
                segments, _ = self.svi_engine.whisper_model.transcribe(
                    tmp_path, beam_size=3, language=w_lang
                )
                window_text = " ".join(seg.text for seg in segments).strip()
                if window_text:
                    self._update_transcript_dedup(window_text)

            # 3. Lexical NLP
            if self.cumulative_transcript:
                red_text, _ = redact(self.cumulative_transcript)
                self.latest_nlp_result = analyze_transcript(red_text)

        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _update_transcript_dedup(self, new_text: str):
        """Deduplicate overlapping transcription windows."""
        new_text = new_text.strip()
        if not new_text:
            return

        if not self.cumulative_transcript:
            self.cumulative_transcript = new_text
            return

        # Simple overlap deduplication
        existing_words = self.cumulative_transcript.split()
        new_words = new_text.split()

        # Check if new_text is substring or appended extension
        if new_text in self.cumulative_transcript:
            return

        # Check suffix overlap
        max_overlap = min(len(existing_words), len(new_words))
        overlap_found = 0

        for k in range(max_overlap, 0, -1):
            if existing_words[-k:] == new_words[:k]:
                overlap_found = k
                break

        if overlap_found > 0:
            appended = " ".join(new_words[overlap_found:])
            if appended:
                self.cumulative_transcript = f"{self.cumulative_transcript} {appended}"
        else:
            self.cumulative_transcript = f"{self.cumulative_transcript} {new_text}"

    def _calculate_trend(self) -> str:
        """
        Calculate trend state from recent smoothed SVI history.
        States: 'INCREASING', 'DECREASING', 'STABLE', 'INSUFFICIENT_DATA'.
        Requires minimum 3 observation history.
        """
        if len(self.smoothed_svi_history) < 3:
            return "INSUFFICIENT_DATA"

        recent = [score for _, score in self.smoothed_svi_history[-5:]]
        x = np.arange(len(recent))
        y = np.array(recent)

        # Simple linear regression slope
        if len(x) >= 3:
            slope = float(np.polyfit(x, y, 1)[0])
            if slope > 1.2:
                return "INCREASING"
            elif slope < -1.2:
                return "DECREASING"
            return "STABLE"

        return "STABLE"

    @staticmethod
    def _get_risk_band(score: float) -> str:
        if score >= RISK_RULES["CRITICAL"]["min_score"]:
            return "CRITICAL"
        if score >= RISK_RULES["HIGH"]["min_score"]:
            return "HIGH"
        if score >= RISK_RULES["MODERATE"]["min_score"]:
            return "MODERATE"
        return "LOW"


def __getattr__(name: str):
    """Dynamic re-export of authoritative StreamingSessionManager to prevent circular imports."""
    if name == "StreamingSessionManager":
        from backend.api.websocket_handler import StreamingSessionManager
        return StreamingSessionManager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

