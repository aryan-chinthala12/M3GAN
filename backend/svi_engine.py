"""
backend/svi_engine.py

SVI Multimodal Fusion Engine for the NHAA (14566) ecosystem.

Modalities:
  - AUDIO: Wav2Vec2 speech-emotion + librosa acoustic biomarkers
           + multilingual lexical analysis of the Whisper transcript.
  - TEXT:  multilingual lexical analysis of chat / portal / chatbot
           narratives directly (no transcription needed).

Fusion:
  - Quadratic-bounded component curves (finer low-range resolution,
    wider spread at the extremes).
  - Severity floors keep genuine high-severity disclosures grounded
    (e.g. suicidal ideation never scores below ~78 even if softly
    spoken / briefly written).
  - No single keyword can force CRITICAL; floors are floors, not
    overrides, and a human reviewer always sees the reasoning.
"""

import os
import re
import threading
import torch
import torch.nn as nn
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from faster_whisper import WhisperModel

from backend.audio_processor import AudioProcessor
from backend.config import (
    WEIGHT_ML_EMOTION,
    WEIGHT_ACOUSTIC,
    WEIGHT_LEXICAL,
    RISK_RULES,
    THREAT_LEVEL_MAP,
    expansion_factor,
)
from backend.text_analyzer import analyze_transcript, redact, threat_level_from_band
from backend.case_store import CaseStore
from backend.schemas import (
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
    def __init__(self):
        print("[INFO] Initializing SVI Multimodal Engine...")

        self.audio_processor = AudioProcessor()
        self.case_store = CaseStore()

        # Heavy ML models load lazily on first audio request so the
        # server starts instantly and TEXT channels work even when the
        # transformers / whisper checkpoints cannot be downloaded.
        self._models_loaded = False
        self._model_lock = threading.Lock()
        self.speech_classifier = None
        self.emotion_feature_extractor = None
        self.whisper_model = None

        # Whisper output-language map for the UI language selector
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
        model = LegacyEmotionClassifier(config)

        checkpoint_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors",
        )
        checkpoint = load_file(checkpoint_path)

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
        if missing:
            print(f"[WARNING] Missing checkpoint weights: {len(missing)}")
        if unexpected:
            print(f"[WARNING] Unexpected checkpoint weights: {len(unexpected)}")
        if not missing and not unexpected:
            print("[INFO] All emotion-model checkpoint weights matched.")

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
    # ACOUSTIC STRESS SCORE
    # =============================================================

    @staticmethod
    def _acoustic_stress_score(acoustic_raw):
        pitch_component = np.clip(acoustic_raw.pitch_std / 2.0, 0.0, 1.0)
        energy_component = np.clip(acoustic_raw.energy_rms / 0.10, 0.0, 1.0)
        energy_variation_component = np.clip(
            acoustic_raw.energy_variation / 12.0, 0.0, 1.0
        )

        acoustic_score = (
            0.50 * pitch_component
            + 0.30 * energy_component
            + 0.20 * energy_variation_component
        )
        return float(np.clip(acoustic_score, 0.0, 1.0))

    # =============================================================
    # SVI FUSION
    # =============================================================

    @staticmethod
    def _component_curve(x: float, weight: float) -> float:
        """
        Monotonic expansion curve: f(x) = x ** g, g = 1 - e/2 in (0.7, 1).
        f(0)=0, f(1)=1; low-intensity signals are lifted so they are not
        drowned by fusion weights, extremes stay well separated.
        """
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

    def _fuse(
        self,
        components,
        severity_floor: float = 0.0,
        floor_labels=None,
    ):
        """
        Weighted fusion of component scores (0..1 each) into a 0..100 SVI.

        A severity floor applies AFTER fusion: it guarantees a minimum
        score when high-severity categories were disclosed, but never
        pushes a case into CRITICAL by itself.
        """
        total = 0.0
        for name, score, weight, *rest in components:
            curved = self._component_curve(score, weight)
            total += curved * weight

        raw = float(np.clip(total, 0.0, 1.0))
        svi = raw * 100.0

        override_triggered = False
        override_reason = None

        if severity_floor > svi:
            svi = severity_floor
            override_triggered = True
            override_reason = (
                f"High-severity distress disclosure detected; "
                f"score raised to severity floor {severity_floor:.0f}. "
                f"Categories: {', '.join(floor_labels or [])}."
            )

        return round(float(np.clip(svi, 0.0, 100.0)), 2), override_triggered, override_reason

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

    def process_multimodal_audio(
        self,
        file_path: str,
        filename: str,
        language: str = "English",
        consent: bool = False,
    ) -> AudioAnalysisResponse:
        # 0. LAZY MODEL LOAD ---------------------------------------
        self._ensure_models()

        # 1. ACOUSTIC FEATURES ------------------------------------
        acoustic_raw = self.audio_processor.extract_acoustic_features(file_path)

        if getattr(acoustic_raw, "error", None):
            raise ValueError(f"Audio validation failed: {acoustic_raw.error}")

        # 2. SPEECH EMOTION ----------------------------------------
        emotion_predictions = self._classify_emotion(file_path)
        top_emotion = emotion_predictions[0]["label"].upper()
        emotion_confidence = emotion_predictions[0]["score"] * 100
        emotion_stress = self._emotion_stress_score(emotion_predictions)

        # 3. SPEECH TO TEXT ----------------------------------------
        whisper_language = self.language_map.get(language)
        segments, info = self.whisper_model.transcribe(
            file_path,
            beam_size=5,
            language=whisper_language,
        )
        transcript = " ".join(segment.text for segment in segments).strip()
        duration = float(info.duration) if hasattr(info, "duration") else 0.0

        # 4. PRIVACY: redact PII before anything is stored/returned
        redacted_transcript, pii_redactions = redact(transcript)

        # 5. MULTILINGUAL LEXICAL ANALYSIS -------------------------
        text_result = analyze_transcript(redacted_transcript)
        lexical_score = text_result["lexical_score"]
        severity_floor = text_result["severity_floor"]
        floor_labels = text_result["floor_categories"]

        # 6. COMPONENT SCORES + FUSION -----------------------------
        acoustic_score = self._acoustic_stress_score(acoustic_raw)

        components = [
            (
                "acoustic",
                acoustic_score,
                WEIGHT_ACOUSTIC,
                f"pitch_std={acoustic_raw.pitch_std:.2f} st, "
                f"rms={acoustic_raw.energy_rms:.4f}, "
                f"energy_var={acoustic_raw.energy_variation:.1f} dB",
            ),
            (
                "emotion",
                emotion_stress,
                WEIGHT_ML_EMOTION,
                f"Wav2Vec2 top emotion {top_emotion} "
                f"({emotion_confidence:.1f}% conf.)",
            ),
            (
                "lexical",
                lexical_score,
                WEIGHT_LEXICAL,
                (
                    f"{len(text_result['hits'])} multilingual lexicon hit(s) "
                    f"in {len(redacted_transcript.split())} words"
                ),
            ),
        ]

        svi_score, override_triggered, override_reason = self._fuse(
            components,
            severity_floor,
            floor_labels,
        )

        risk_band = self._get_risk_band(svi_score)
        risk_rule = RISK_RULES[risk_band]
        threat_level = threat_level_from_band(risk_band)

        # 7. EXPLAINABILITY ----------------------------------------
        top_contributors = sorted(
            components,
            key=lambda c: c[1] * c[2],
            reverse=True,
        )
        contributor_labels = []
        for name, score, weight, *rest in top_contributors:
            detail = rest[0] if rest else None
            if score >= 0.35:
                contributor_labels.append(
                    f"{name.capitalize()} signal: {detail or 'elevated'}"
                )

        if text_result["categories"]:
            contributor_labels.append(
                "Distress categories: " + ", ".join(text_result["categories"])
            )

        if not contributor_labels:
            contributor_labels.append("No dominant elevated signal detected")

        summary = (
            f"Analysis yielded a {risk_band} risk profile with SVI "
            f"{svi_score} (acoustic {acoustic_score:.2f}, emotion "
            f"{emotion_stress:.2f}, lexical {lexical_score:.2f})."
        )

        # 8. PERSIST CASE ------------------------------------------
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
            ),
            explainability=Explainability(
                summary=summary,
                top_contributors=contributor_labels,
            ),
            recommended_interventions=risk_rule["actions"],
            consent_recorded=consent,
        )

    # =============================================================
    # TEXT PIPELINE (chat / portal / chatbot / IVRS transcript)
    # =============================================================

    def process_text(self, request: TextAnalysisRequest) -> TextAnalysisResponse:
        raw_text = request.text.strip()

        if not raw_text:
            raise ValueError("Text payload is empty after trimming.")

        # 1. PRIVACY: redact PII before analysis/storage
        redacted_text, pii_redactions = redact(raw_text)

        # 2. MULTILINGUAL LEXICAL ANALYSIS --------------------------
        text_result = analyze_transcript(redacted_text)
        lexical_score = text_result["lexical_score"]
        severity_floor = text_result["severity_floor"]
        floor_labels = text_result["floor_categories"]

        # 3. FUSION (text modality: single component + floor) -------
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

        svi_score, override_triggered, override_reason = self._fuse(
            components,
            severity_floor,
            floor_labels,
        )

        risk_band = self._get_risk_band(svi_score)
        risk_rule = RISK_RULES[risk_band]
        threat_level = threat_level_from_band(risk_band)

        # 4. EXPLAINABILITY ----------------------------------------
        contributor_labels = []
        if text_result["categories"]:
            contributor_labels.append(
                "Distress categories: " + ", ".join(text_result["categories"])
            )
        if text_result["hits"]:
            contributor_labels.append(
                "Flagged terms: " + ", ".join(text_result["flagged_keywords"])
            )
        if not contributor_labels:
            contributor_labels.append(
                "No elevated distress indicators in narrative"
            )

        summary = (
            f"Text analysis yielded a {risk_band} risk profile with SVI "
            f"{svi_score} (lexical {lexical_score:.2f})."
        )

        # 5. PERSIST CASE -------------------------------------------
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
                word_count=len(redacted_text.split()),
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
            ),
            explainability=Explainability(
                summary=summary,
                top_contributors=contributor_labels,
            ),
            recommended_interventions=risk_rule["actions"],
            consent_recorded=True,
        )

    # =============================================================
    # CASE QUERY API (used by FastAPI routes)
    # =============================================================

    def list_cases(self, limit: int = 100) -> list:
        return self.case_store.list_cases(limit)

    def get_stats(self) -> StatsResponse:
        return self.case_store.get_stats()
