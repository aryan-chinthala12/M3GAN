import os
import torch
import numpy as np

# Disable symlinks for Hugging Face on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline
from faster_whisper import WhisperModel
from backend.audio_processor import AudioProcessor
from backend.schemas import (
    AudioAnalysisResponse, 
    AcousticIndicators, 
    NLPIndicators, 
    SVIMetrics, 
    Explainability
)

class SVIEngine:
    def __init__(self):
        print("[INFO] Initializing SVI Multimodal Engine...")
        
        # Audio Preprocessing Module
        self.audio_processor = AudioProcessor()

        # Speech Emotion Recognition (SER) Pipeline
        print("[INFO] Loading Wav2Vec 2.0 Speech Emotion Transformer...")
        self.speech_classifier = pipeline(
            "audio-classification", 
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )

        # Faster-Whisper Speech-to-Text Model
        print("[INFO] Loading Faster-Whisper STT Engine...")
        self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

        # High-risk trigger keywords for lexical analysis
        self.risk_keywords = [
            "suicide", "kill", "die", "depressed", "help", "pain", 
            "end it", "hopeless", "harm", "bleeding", "overdose", "alone"
        ]

    def process_multimodal_audio(self, file_path: str, filename: str) -> AudioAnalysisResponse:
        """Fuses acoustic emotion detection and lexical transcription analysis."""
        
        # 1. Acoustic Signal Processing
        acoustic_raw = self.audio_processor.extract_acoustic_features(file_path)

        # 2. Wav2Vec 2.0 Emotion Classification
        emotion_predictions = self.speech_classifier(file_path)
        top_emotion = emotion_predictions[0]["label"].upper()
        emotion_score = float(emotion_predictions[0]["score"]) * 100

        # 3. Whisper Speech-to-Text
        segments, info = self.whisper_model.transcribe(file_path, beam_size=5)
        transcript = " ".join([segment.text for segment in segments]).strip()
        duration = float(info.duration) if hasattr(info, 'duration') else 0.0

        # 4. Lexical Threat Keyword Matching
        matched_triggers = [kw for kw in self.risk_keywords if kw in transcript.lower()]
        keyword_density = len(matched_triggers) / (len(transcript.split()) + 1)

        # 5. Stress Vulnerability Index (SVI) Formula
        emotion_weights = {
            "SAD": 0.85, "SADNESS": 0.85,
            "FEAR": 0.90, "FEARFUL": 0.90,
            "ANGRY": 0.70, "ANGER": 0.70,
            "NEUTRAL": 0.20,
            "HAPPY": 0.05
        }
        
        base_emotion_weight = emotion_weights.get(top_emotion, 0.50)
        acoustic_stress = min(acoustic_raw.pitch_std / 100.0, 1.0) * 0.3 + min(acoustic_raw.energy_rms * 5.0, 1.0) * 0.7

        raw_svi = (
            (base_emotion_weight * 0.4) +
            (acoustic_stress * 0.3) +
            (min(keyword_density * 3.0, 1.0) * 0.3)
        )
        svi_score = round(float(np.clip(raw_svi * 100, 0, 100)), 2)

        # Keyword Override Check
        override_triggered = len(matched_triggers) > 0
        override_reason = "High-risk distress keyword detected in transcript" if override_triggered else None

        # Risk Banding & Colors
        if svi_score >= 75 or override_triggered:
            risk_band = "CRITICAL"
            risk_color = "#D32F2F"
            threat_level = "HIGH"
            interventions = ["IMMEDIATE_HUMAN_DISPATCH", "ALERT_SAFETY_TEAM"]
        elif svi_score >= 45:
            risk_band = "MODERATE"
            risk_color = "#FFA000"
            threat_level = "MEDIUM"
            interventions = ["OPERATOR_MONITORING"]
        else:
            risk_band = "LOW"
            risk_color = "#388E3C"
            threat_level = "LOW"
            interventions = ["ROUTINE_LOGGING"]

        # Explanations
        top_contributors = []
        if base_emotion_weight > 0.5:
            top_contributors.append(f"Acoustic emotion classified as {top_emotion}")
        if acoustic_raw.pitch_std > 50:
            top_contributors.append("High pitch variance / voice instability")
        if matched_triggers:
            top_contributors.append(f"Flagged keywords: {', '.join(matched_triggers)}")

        return AudioAnalysisResponse(
            status="success",
            filename=filename,
            duration_seconds=round(duration, 2),
            transcript=transcript if transcript else "[NO SPEECH DETECTED]",
            acoustic_indicators=AcousticIndicators(
                top_emotion=top_emotion,
                pitch_volatility=round(float(acoustic_raw.pitch_std), 2),
                rms_energy=round(float(acoustic_raw.energy_rms), 4),
                confidence=round(emotion_score, 1)
            ),
            nlp_indicators=NLPIndicators(
                detected_emotion=top_emotion,
                threat_level=threat_level,
                flagged_keywords=matched_triggers
            ),
            svi_metrics=SVIMetrics(
                final_svi_score=svi_score,
                risk_band=risk_band,
                risk_color=risk_color,
                override_triggered=override_triggered,
                override_reason=override_reason
            ),
            explainability=Explainability(
                summary=f"Analysis yielded a {risk_band} risk profile with SVI score {svi_score}.",
                top_contributors=top_contributors if top_contributors else ["Baseline acoustic features normal"]
            ),
            recommended_interventions=interventions
        )