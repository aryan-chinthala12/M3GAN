"""
backend/svi_engine.py

SVI Multimodal Fusion Engine for the NHAA (14566) ecosystem.

Modalities:
  - AUDIO: Wav2Vec2 speech-emotion + Silero VAD / librosa acoustic biomarkers
           + multilingual lexical analysis of the Whisper transcript.
  - TEXT:  multilingual lexical analysis of chat / portal / chatbot
           narratives directly (no transcription needed).

Fusion & Safety Design:
  - Numerical SVI Score (0..100): Calibrated monotonic component fusion,
    mathematically bounded within 0..100.
  - Safety Flags: Independent alert triggers for high-severity disclosures
    (e.g., suicidal ideation, sexual violence, death threats) that alert the human
    operator WITHOUT corrupting or overwriting the numerical SVI score.
  - Confidence & Signal Quality: Evaluates analysis reliability based on
    VAD speech ratio, acoustic signal quality, SER confidence, and text length.
"""

import os
import re
import threading
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from faster_whisper import WhisperModel

from backend.audio.audio_processor import AudioProcessor
from backend.core.config import (
    WEIGHT_ML_EMOTION,
    WEIGHT_ACOUSTIC,
    WEIGHT_LEXICAL,
    RISK_RULES,
    THREAT_LEVEL_MAP,
    expansion_factor,
)
from backend.nlp.text_analyzer import analyze_transcript, redact, threat_level_from_band
from backend.core.case_store import CaseStore
from backend.api.schemas import (
    AudioAnalysisResponse,
    TextAnalysisRequest,
    TextAnalysisResponse,
    AcousticIndicators,
    NLPIndicators,
    TextIndicators,
    SVIMetrics,
    AcousticComponent,
    Explainability,
    CaseSummary,
    StatsResponse,
    SafetyFlagItem,
    ConfidenceMetrics,
)


MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


class LegacyEmotionClassifier(nn.Module):
    """Compatibility model for the legacy Wav2Vec2 emotion checkpoint."""

    def __init__(self, config):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = nn.ModuleDict({
            "dense": nn.Linear(config.hidden_size, config.hidden_size),
            "output": nn.Linear(config.hidden_size, config.num_labels),
        })

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            feature_attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            mask = feature_attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (
                (hidden_states * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )
        else:
            pooled = hidden_states.mean(dim=1)

        x = torch.tanh(self.classifier["dense"](pooled))
        return self.classifier["output"](x)


class SVIEngine:
    def __init__(self, use_vad_fallback: bool = False):
        print("[INFO] Initializing SVI Multimodal Engine...")

        self.audio_processor = AudioProcessor(use_vad_fallback=use_vad_fallback)
        self.case_store = CaseStore()

        self._models_loaded = False
        self._model_lock = threading.Lock()
        self.speech_classifier = None
        self.emotion_feature_extractor = None
        self.whisper_model = None

        self.language_map = {
            "English": "en",
            "Hindi": "hi",
            "Telugu": "te",
            "Marathi": "mr",
            "Bengali": "bn",
            "Tamil": "ta",
            "Kannada": "kn",
        }

    # =============================================================
    # MODEL LOADING (lazy, thread-safe)
    # =============================================================

    def _ensure_models(self):
        if self._models_loaded:
            return

        with self._model_lock:
            if self._models_loaded:
                return

            print("[INFO] Loading Wav2Vec 2.0 Speech Emotion Transformer...")
            self.speech_classifier = self._load_emotion_model()

            print("[INFO] Loading Faster-Whisper STT Engine...")
            self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

            self._models_loaded = True
            print("[INFO] All ML models ready.")

    def _load_emotion_model(self):
        print("[INFO] Loading legacy Wav2Vec2 checkpoint...")

        config = Wav2Vec2Config.from_pretrained(MODEL_ID)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        checkpoint_path = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors")

        checkpoint = load_file(checkpoint_path)
        model = LegacyEmotionClassifier(config)

        remapped = {}
        for key, value in checkpoint.items():
            new_key = key
            if key.endswith("wav2vec2.encoder.pos_conv_embed.conv.weight_g"):
                new_key = (
                    "wav2vec2.encoder.pos_conv_embed.conv."
                    "parametrizations.weight.original0"
                )
            elif key.endswith("wav2vec2.encoder.pos_conv_embed.conv.weight_v"):
                new_key = (
                    "wav2vec2.encoder.pos_conv_embed.conv."
                    "parametrizations.weight.original1"
                )
            remapped[new_key] = value

        missing, unexpected = model.load_state_dict(remapped, strict=False)

        print("[INFO] Emotion checkpoint loaded.")
        model.eval()
        self.emotion_feature_extractor = feature_extractor
        return model

    # =============================================================
    # EMOTION CLASSIFICATION (AUDIO)
    # =============================================================

    def _classify_emotion(self, file_path):
        import librosa

        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)

        inputs = self.emotion_feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.speech_classifier(
                input_values=inputs.input_values,
                attention_mask=(
                    inputs.attention_mask
                    if hasattr(inputs, "attention_mask")
                    else None
                ),
            )
            probabilities = torch.softmax(logits, dim=-1)

        scores = probabilities[0]
        labels = [
            "angry", "calm", "disgust", "fearful",
            "happy", "neutral", "sad", "surprised",
        ]

        predictions = [
            {"label": labels[i], "score": float(scores[i])}
            for i in range(len(labels))
        ]
        predictions.sort(key=lambda item: item["score"], reverse=True)
        return predictions

    @staticmethod
    def _emotion_stress_score(predictions):
        stress_weights = {
            "angry": 0.65, "calm": 0.05, "disgust": 0.50, "fearful": 0.90,
            "happy": 0.05, "neutral": 0.20, "sad": 0.75, "surprised": 0.40,
        }
        score = 0.0
        for prediction in predictions:
            label = prediction["label"].lower()
            probability = prediction["score"]
            score += probability * stress_weights.get(label, 0.20)
        return float(np.clip(score, 0.0, 1.0))

    # =============================================================
    # SVI FUSION & SAFETY FLAGS
    # =============================================================

    @staticmethod
    def _component_curve(x: float, weight: float) -> float:
        """Monotonic expansion curve f(x) = x ** (1 - e/2)."""
        e = expansion_factor(weight)
        g = 1.0 - (e / 2.0)
        x = float(np.clip(x, 0.0, 1.0))
        return float(x ** g)

    @staticmethod
    def _build_component(name, score, weight, detail=None):
        contribution = round(score * weight * 100, 2)
        return AcousticComponent(
            name=name,
            score=round(float(score), 4),
            weight=weight,
            contribution=contribution,
            detail=detail,
        )

    def _fuse_with_safety_flags(
        self,
        components: List[Tuple[str, float, float, str]],
        hits: List[dict],
        floor_labels: List[str],
        severity_floor: float,
    ) -> Tuple[float, List[SafetyFlagItem], bool, Optional[str]]:
        """
        Calculates mathematically bounded 0..100 SVI score.
        Generates independent SafetyFlagItem objects without modifying the numerical score.
        """
        total = 0.0
        for name, score, weight, *rest in components:
            curved = self._component_curve(score, weight)
            total += curved * weight

        raw = float(np.clip(total, 0.0, 1.0))
        svi_score = round(float(np.clip(raw * 100.0, 0.0, 100.0)), 2)

        safety_flags = []
        for cat_label in floor_labels:
            matched = [h["term"] for h in hits if h["category"] == cat_label]
            severity = "CRITICAL" if severity_floor >= 70.0 else "HIGH"
            safety_flags.append(
                SafetyFlagItem(
                    category=cat_label,
                    severity=severity,
                    flag="URGENT_HUMAN_REVIEW",
                    message=f"High-severity disclosure detected ({cat_label}). Urgent human review required.",
                    matched_terms=matched,
                )
            )

        override_triggered = bool(len(safety_flags) > 0)
        override_reason = (
            f"Safety flag(s) active for urgent human review: {', '.join(floor_labels)}."
            if safety_flags
            else None
        )

        return svi_score, safety_flags, override_triggered, override_reason

    # =============================================================
    # RISK BAND
    # =============================================================

    @staticmethod
    def _get_risk_band(svi_score):
        if svi_score >= RISK_RULES["CRITICAL"]["min_score"]:
            return "CRITICAL"
        if svi_score >= RISK_RULES["HIGH"]["min_score"]:
            return "HIGH"
        if svi_score >= RISK_RULES["MODERATE"]["min_score"]:
            return "MODERATE"
        return "LOW"

    # =============================================================
    # AUDIO PIPELINE
    # =============================================================

    @staticmethod
    def get_fusion_weights(
        signal_quality: str,
        fusion_mode: str = "DYNAMIC_SCHEME_A",
    ) -> Tuple[float, float, float]:
        """
        Return (weight_acoustic, weight_emotion, weight_lexical) based on signal quality state.
        Supported fusion_modes for benchmarking: 'BASELINE', 'DYNAMIC_SCHEME_A', 'DYNAMIC_SCHEME_B'.
        """
        if fusion_mode == "BASELINE":
            return WEIGHT_ACOUSTIC, WEIGHT_ML_EMOTION, WEIGHT_LEXICAL

        if fusion_mode == "DYNAMIC_SCHEME_B":
            if signal_quality in ("WHISPER", "LOW_VOLUME"):
                return 0.10, 0.30, 0.60
            if signal_quality == "LOW_SNR":
                return 0.20, 0.30, 0.50
            return WEIGHT_ACOUSTIC, WEIGHT_ML_EMOTION, WEIGHT_LEXICAL

        # Default: DYNAMIC_SCHEME_A
        if signal_quality in ("WHISPER", "LOW_VOLUME"):
            return 0.15, 0.20, 0.65
        if signal_quality == "LOW_SNR":
            return 0.15, 0.35, 0.50

        return WEIGHT_ACOUSTIC, WEIGHT_ML_EMOTION, WEIGHT_LEXICAL

    def process_multimodal_audio(
        self,
        file_path: str,
        filename: str,
        language: str = "English",
        consent: bool = False,
        fusion_mode: str = "DYNAMIC_SCHEME_A",
    ) -> AudioAnalysisResponse:
        # 0. LAZY MODEL LOAD
        self._ensure_models()

        # 1. ACOUSTIC FEATURES & VAD
        acoustic_raw = self.audio_processor.extract_acoustic_features(file_path)

        if getattr(acoustic_raw, "error", None):
            raise ValueError(f"Audio validation failed: {acoustic_raw.error}")

        # 2. SPEECH EMOTION
        emotion_predictions = self._classify_emotion(file_path)
        top_emotion = emotion_predictions[0]["label"].upper()
        emotion_confidence = emotion_predictions[0]["score"] * 100
        emotion_stress = self._emotion_stress_score(emotion_predictions)

        # 3. SPEECH TO TEXT
        whisper_language = self.language_map.get(language)
        segments, info = self.whisper_model.transcribe(
            file_path,
            beam_size=5,
            language=whisper_language,
        )
        transcript = " ".join(segment.text for segment in segments).strip()
        duration = float(info.duration) if hasattr(info, "duration") else acoustic_raw.duration

        # 4. PRIVACY: REDACT PII
        redacted_transcript, pii_redactions = redact(transcript)

        # 5. MULTILINGUAL LEXICAL ANALYSIS
        text_result = analyze_transcript(redacted_transcript)
        lexical_score = text_result["lexical_score"]
        severity_floor = text_result["severity_floor"]
        floor_labels = text_result["floor_categories"]

        # 6. COMPONENT SCORES & FUSION (SVI + Safety Flags)
        acoustic_score = getattr(acoustic_raw, "acoustic_score", 0.0)
        quality = getattr(acoustic_raw, "signal_quality", "GOOD")

        w_ac, w_em, w_lex = self.get_fusion_weights(quality, fusion_mode=fusion_mode)

        components = [
            (
                "acoustic",
                acoustic_score,
                w_ac,
                getattr(acoustic_raw, "acoustic_detail", f"pitch_std={acoustic_raw.pitch_std:.2f} st"),
            ),
            (
                "emotion",
                emotion_stress,
                w_em,
                f"Wav2Vec2 top emotion {top_emotion} ({emotion_confidence:.1f}% conf.)",
            ),
            (
                "lexical",
                lexical_score,
                w_lex,
                f"{len(text_result['hits'])} multilingual hit(s) in {len(redacted_transcript.split())} words",
            ),
        ]

        svi_score, safety_flags, override_triggered, override_reason = self._fuse_with_safety_flags(
            components,
            text_result["hits"],
            floor_labels,
            severity_floor,
        )

        risk_band = self._get_risk_band(svi_score)
        risk_rule = RISK_RULES[risk_band]
        threat_level = threat_level_from_band(risk_band)

        # 7. CONFIDENCE & SIGNAL QUALITY METRICS
        voiced_ratio = getattr(acoustic_raw, "voiced_ratio", 0.0)
        pause_ratio = getattr(acoustic_raw, "pause_ratio", 0.0)
        speech_dur = getattr(acoustic_raw, "speech_duration", 0.0)
        pitch_rel = getattr(acoustic_raw, "pitch_reliable", True)

        quality_conf_map = {
            "GOOD": 100.0,
            "DEGRADED": 70.0,
            "LOW_SNR": 60.0,
            "LOW_VOLUME": 55.0,
            "WHISPER": 50.0,
            "UNRELIABLE": 10.0,
        }
        signal_quality_conf = quality_conf_map.get(quality, 60.0)

        word_cnt = len(redacted_transcript.split())
        text_conf = min(100.0, word_cnt * 10.0) if word_cnt > 0 else 0.0

        overall_confidence = round(
            0.40 * signal_quality_conf + 0.35 * emotion_confidence + 0.25 * text_conf, 1
        )
        conf_rating = "HIGH" if overall_confidence >= 75.0 else ("MEDIUM" if overall_confidence >= 45.0 else "LOW")

        confidence_metrics = ConfidenceMetrics(
            overall_confidence=overall_confidence,
            confidence_rating=conf_rating,
            signal_quality=quality,
            speech_duration_seconds=speech_dur,
            voiced_ratio=voiced_ratio,
            pause_ratio=pause_ratio,
            pitch_reliable=pitch_rel,
            snr_state=getattr(acoustic_raw, "snr_state", "UNAVAILABLE"),
            snr_db=getattr(acoustic_raw, "snr_db", None),
            provenance={
                "acoustic_engine": "SileroVAD + librosa pyin",
                "emotion_engine": f"Wav2Vec2 ({MODEL_ID})",
                "asr_engine": "Faster-Whisper (tiny)",
                "nlp_engine": "Multilingual distress lexicon scanner",
            },
        )

        # 8. EXPLAINABILITY
        top_contributors = sorted(
            components,
            key=lambda c: c[1] * c[2],
            reverse=True,
        )
        contributor_labels = []
        for name, score, weight, *rest in top_contributors:
            detail = rest[0] if rest else None
            if score >= 0.35:
                contributor_labels.append(f"{name.capitalize()} signal: {detail or 'elevated'}")

        if text_result["categories"]:
            contributor_labels.append("Distress categories: " + ", ".join(text_result["categories"]))

        if not contributor_labels:
            contributor_labels.append("No dominant elevated signal detected")

        summary = (
            f"Normalized baseline assessment yielded a {risk_band} risk profile with SVI "
            f"{svi_score:.1f}/100 (acoustic {acoustic_score:.2f}, emotion "
            f"{emotion_stress:.2f}, lexical {lexical_score:.2f}). "
            f"Confidence: {overall_confidence:.1f}% ({conf_rating})."
        )

        # 9. PERSIST CASE
        case_id = self.case_store.create_case(
            channel="audio",
            language=language,
            svi_score=svi_score,
            risk_band=risk_band,
            risk_color=risk_rule["color"],
            status="logged",
            text=redacted_transcript,
        )

        return AudioAnalysisResponse(
            status="success",
            case_id=case_id,
            filename=filename,
            channel="audio",
            language=language,
            duration_seconds=round(duration, 2),
            transcript=redacted_transcript if redacted_transcript else "[NO SPEECH DETECTED]",
            acoustic_indicators=AcousticIndicators(
                top_emotion=top_emotion,
                pitch_volatility=round(float(acoustic_raw.pitch_std), 2),
                rms_energy=round(float(acoustic_raw.energy_rms), 4),
                energy_variation=round(float(getattr(acoustic_raw, "energy_variation", 0.0)), 2),
                median_pitch_hz=round(float(getattr(acoustic_raw, "median_pitch", 0.0)), 2),
                confidence=round(emotion_confidence, 1),
                voiced_ratio=voiced_ratio,
                pause_ratio=pause_ratio,
                signal_quality=quality,
                pitch_reliable=pitch_rel,
            ),
            nlp_indicators=NLPIndicators(
                threat_level=threat_level,
                flagged_keywords=text_result["flagged_keywords"],
                distress_categories=text_result["categories"],
                pii_redactions=pii_redactions,
            ),
            text_indicators=TextIndicators(
                word_count=len(redacted_transcript.split()),
                distress_density=lexical_score,
            ),
            svi_metrics=SVIMetrics(
                final_svi_score=svi_score,
                risk_band=risk_band,
                risk_color=risk_rule["color"],
                override_triggered=override_triggered,
                override_reason=override_reason,
                components=[
                    self._build_component(name, score, weight, rest[0] if rest else None)
                    for name, score, weight, *rest in components
                ],
                safety_flags=safety_flags,
            ),
            explainability=Explainability(
                summary=summary,
                top_contributors=contributor_labels,
            ),
            recommended_interventions=risk_rule["actions"],
            consent_recorded=consent,
            confidence_metrics=confidence_metrics,
            safety_flags=safety_flags,
        )

    # =============================================================
    # TEXT PIPELINE
    # =============================================================

    def process_text(self, request: TextAnalysisRequest) -> TextAnalysisResponse:
        raw_text = request.text.strip()

        if not raw_text:
            raise ValueError("Text payload is empty after trimming.")

        # 1. PRIVACY: REDACT PII
        redacted_text, pii_redactions = redact(raw_text)

        # 2. MULTILINGUAL LEXICAL ANALYSIS
        text_result = analyze_transcript(redacted_text)
        lexical_score = text_result["lexical_score"]
        severity_floor = text_result["severity_floor"]
        floor_labels = text_result["floor_categories"]

        # 3. FUSION & SAFETY FLAGS
        components = [
            (
                "lexical",
                lexical_score,
                1.00,
                (
                    f"{len(text_result['hits'])} multilingual lexicon hit(s) "
                    f"in {len(redacted_text.split())} words; categories: "
                    f"{', '.join(text_result['categories']) or 'none'}"
                ),
            ),
        ]

        svi_score, safety_flags, override_triggered, override_reason = self._fuse_with_safety_flags(
            components,
            text_result["hits"],
            floor_labels,
            severity_floor,
        )

        risk_band = self._get_risk_band(svi_score)
        risk_rule = RISK_RULES[risk_band]
        threat_level = threat_level_from_band(risk_band)

        # 4. CONFIDENCE METRICS
        word_cnt = len(redacted_text.split())
        overall_confidence = min(100.0, word_cnt * 10.0) if word_cnt > 0 else 0.0
        conf_rating = "HIGH" if overall_confidence >= 75.0 else ("MEDIUM" if overall_confidence >= 40.0 else "LOW")

        confidence_metrics = ConfidenceMetrics(
            overall_confidence=overall_confidence,
            confidence_rating=conf_rating,
            signal_quality="GOOD",
            speech_duration_seconds=0.0,
            voiced_ratio=0.0,
            pause_ratio=0.0,
            pitch_reliable=False,
            snr_state="UNAVAILABLE",
            snr_db=None,
            provenance={"text_engine": "Multilingual distress lexicon scanner"},
        )

        # 5. EXPLAINABILITY
        contributor_labels = []
        if text_result["categories"]:
            contributor_labels.append("Distress categories: " + ", ".join(text_result["categories"]))
        if text_result["hits"]:
            contributor_labels.append("Flagged terms: " + ", ".join(text_result["flagged_keywords"]))
        if not contributor_labels:
            contributor_labels.append("No elevated distress indicators in narrative")

        summary = (
            f"Normalized baseline text analysis yielded a {risk_band} risk profile with SVI "
            f"{svi_score:.1f}/100 (lexical density {lexical_score:.2f})."
        )

        # 6. PERSIST CASE
        case_id = self.case_store.create_case(
            channel=request.channel,
            language=request.language,
            svi_score=svi_score,
            risk_band=risk_band,
            risk_color=risk_rule["color"],
            status="logged",
            text=redacted_text,
        )

        return TextAnalysisResponse(
            status="success",
            case_id=case_id,
            channel=request.channel,
            language=request.language,
            redacted_text=redacted_text,
            text_indicators=TextIndicators(
                word_count=word_cnt,
                distress_density=lexical_score,
            ),
            nlp_indicators=NLPIndicators(
                threat_level=threat_level,
                flagged_keywords=text_result["flagged_keywords"],
                distress_categories=text_result["categories"],
                pii_redactions=pii_redactions,
            ),
            svi_metrics=SVIMetrics(
                final_svi_score=svi_score,
                risk_band=risk_band,
                risk_color=risk_rule["color"],
                override_triggered=override_triggered,
                override_reason=override_reason,
                components=[
                    self._build_component(name, score, weight, rest[0] if rest else None)
                    for name, score, weight, *rest in components
                ],
                safety_flags=safety_flags,
            ),
            explainability=Explainability(
                summary=summary,
                top_contributors=contributor_labels,
            ),
            recommended_interventions=risk_rule["actions"],
            consent_recorded=True,
            confidence_metrics=confidence_metrics,
            safety_flags=safety_flags,
        )

    def list_cases(self, limit: int = 100) -> list:
        return self.case_store.list_cases(limit)

    def get_stats(self) -> StatsResponse:
        return self.case_store.get_stats()
