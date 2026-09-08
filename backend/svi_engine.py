import os
import re
import torch
import torch.nn as nn
import numpy as np

# Disable symlinks warning on Windows
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
    RISK_RULES,
)
from backend.schemas import (
    AudioAnalysisResponse,
    AcousticIndicators,
    NLPIndicators,
    SVIMetrics,
    Explainability,
)


MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


class LegacyEmotionClassifier(nn.Module):
    """
    Compatibility model for the legacy Wav2Vec2 emotion checkpoint.
    """

    def __init__(self, config):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model(config)

        self.classifier = nn.ModuleDict({
            "dense": nn.Linear(
                config.hidden_size,
                config.hidden_size,
            ),
            "output": nn.Linear(
                config.hidden_size,
                config.num_labels,
            ),
        })

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            feature_attention_mask = (
                self._get_feature_attention_mask(
                    attention_mask,
                    hidden_states.shape[1],
                )
            )

            mask = feature_attention_mask.unsqueeze(-1).to(
                hidden_states.dtype
            )

            pooled = (
                (hidden_states * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )
        else:
            pooled = hidden_states.mean(dim=1)

        x = self.classifier["dense"](pooled)
        x = torch.tanh(x)

        logits = self.classifier["output"](x)

        return logits

    def _get_feature_attention_mask(
        self,
        attention_mask,
        feature_length,
    ):
        return self.wav2vec2._get_feature_vector_attention_mask(
            feature_length,
            attention_mask,
        )


class SVIEngine:

    def __init__(self):

        print("[INFO] Initializing SVI Multimodal Engine...")

        # ---------------------------------------------------------
        # AUDIO PROCESSOR
        # ---------------------------------------------------------

        self.audio_processor = AudioProcessor()

        # ---------------------------------------------------------
        # SPEECH EMOTION MODEL
        # ---------------------------------------------------------

        print(
            "[INFO] Loading Wav2Vec 2.0 Speech Emotion Transformer..."
        )

        self.speech_classifier = self._load_emotion_model()

        # ---------------------------------------------------------
        # WHISPER STT
        # ---------------------------------------------------------

        print("[INFO] Loading Faster-Whisper STT Engine...")

        self.whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
        )

        # ---------------------------------------------------------
        # LEXICAL INDICATORS
        # ---------------------------------------------------------

        # These are indicators only.
        # They must NOT automatically force CRITICAL risk.
        self.risk_keywords = [
            "suicide",
            "kill",
            "die",
            "depressed",
            "help",
            "pain",
            "end it",
            "hopeless",
            "harm",
            "bleeding",
            "overdose",
            "alone",
        ]

        # Higher-severity lexical indicators receive more weight.
        self.keyword_weights = {
            "suicide": 1.00,
            "kill": 0.90,
            "overdose": 0.90,
            "harm": 0.70,
            "bleeding": 0.60,
            "end it": 0.80,
            "hopeless": 0.60,
            "depressed": 0.45,
            "pain": 0.30,
            "alone": 0.25,
            "help": 0.15,
            "die": 0.60,
        }

    # =============================================================
    # MODEL LOADING
    # =============================================================

    def _load_emotion_model(self):

        print("[INFO] Loading legacy Wav2Vec2 checkpoint...")

        config = Wav2Vec2Config.from_pretrained(
            MODEL_ID
        )

        feature_extractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(
                MODEL_ID
            )
        )

        model = LegacyEmotionClassifier(config)

        checkpoint_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors",
        )

        checkpoint = load_file(checkpoint_path)

        remapped = {}

        for key, value in checkpoint.items():

            new_key = key

            if key.endswith(
                "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
            ):
                new_key = (
                    "wav2vec2.encoder.pos_conv_embed.conv."
                    "parametrizations.weight.original0"
                )

            elif key.endswith(
                "wav2vec2.encoder.pos_conv_embed.conv.weight_v"
            ):
                new_key = (
                    "wav2vec2.encoder.pos_conv_embed.conv."
                    "parametrizations.weight.original1"
                )

            remapped[new_key] = value

        missing, unexpected = model.load_state_dict(
            remapped,
            strict=False,
        )

        print("[INFO] Emotion checkpoint loaded.")

        if missing:

            print("[WARNING] Missing checkpoint weights:")

            for key in missing:
                print(f"  - {key}")

        if unexpected:

            print("[WARNING] Unexpected checkpoint weights:")

            for key in unexpected:
                print(f"  - {key}")

        if not missing and not unexpected:

            print(
                "[INFO] All emotion-model checkpoint weights matched."
            )

        model.eval()

        self.emotion_feature_extractor = feature_extractor

        return model

    # =============================================================
    # EMOTION CLASSIFICATION
    # =============================================================

    def _classify_emotion(self, file_path):

        import librosa

        audio, sample_rate = librosa.load(
            file_path,
            sr=16000,
            mono=True,
        )

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

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

        scores = probabilities[0]

        labels = [
            "angry",
            "calm",
            "disgust",
            "fearful",
            "happy",
            "neutral",
            "sad",
            "surprised",
        ]

        predictions = [
            {
                "label": labels[i],
                "score": float(scores[i]),
            }
            for i in range(len(labels))
        ]

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return predictions

    # =============================================================
    # EMOTION STRESS SCORE
    # =============================================================

    @staticmethod
    def _emotion_stress_score(predictions):

        stress_weights = {
            "angry": 0.65,
            "calm": 0.05,
            "disgust": 0.50,
            "fearful": 0.90,
            "happy": 0.05,
            "neutral": 0.20,
            "sad": 0.75,
            "surprised": 0.40,
        }

        score = 0.0

        for prediction in predictions:

            label = prediction["label"].lower()
            probability = prediction["score"]

            weight = stress_weights.get(
                label,
                0.20,
            )

            score += probability * weight

        return float(np.clip(score, 0.0, 1.0))

    # =============================================================
    # ACOUSTIC STRESS SCORE
    # =============================================================

    @staticmethod
    def _acoustic_stress_score(acoustic_raw):

        # Step 1 pitch volatility is now measured in
        # semitone standard deviation.
        #
        # These are deliberately conservative normalization
        # ranges. They are NOT clinical thresholds.
        pitch_component = np.clip(
            acoustic_raw.pitch_std / 2.0,
            0.0,
            1.0,
        )

        # RMS is normalized relative to a practical speech range.
        energy_component = np.clip(
            acoustic_raw.energy_rms / 0.10,
            0.0,
            1.0,
        )

        # Energy variation is also incorporated when available.
        energy_variation_component = np.clip(
            acoustic_raw.energy_variation / 12.0,
            0.0,
            1.0,
        )

        acoustic_score = (
            0.50 * pitch_component
            + 0.30 * energy_component
            + 0.20 * energy_variation_component
        )

        return float(
            np.clip(
                acoustic_score,
                0.0,
                1.0,
            )
        )

    # =============================================================
    # LEXICAL STRESS SCORE
    # =============================================================

    def _lexical_analysis(self, transcript):

        text = transcript.lower()

        matched_triggers = []
        weighted_scores = []

        for keyword in self.risk_keywords:

            # Word-boundary matching for single words.
            # Prevents things like "helpful" matching "help".

            if " " not in keyword:

                pattern = rf"\b{re.escape(keyword)}\b"

            else:

                pattern = rf"\b{re.escape(keyword)}\b"

            if re.search(pattern, text):

                matched_triggers.append(keyword)

                weighted_scores.append(
                    self.keyword_weights.get(
                        keyword,
                        0.20,
                    )
                )

        word_count = max(
            len(text.split()),
            1,
        )

        # Keep lexical contribution bounded.
        #
        # Repetition should increase the score gradually,
        # not instantly produce CRITICAL.
        lexical_strength = (
            sum(weighted_scores)
            / np.sqrt(word_count)
        )

        lexical_score = float(
            np.clip(
                lexical_strength,
                0.0,
                1.0,
            )
        )

        return (
            matched_triggers,
            lexical_score,
        )

    # =============================================================
    # RISK BAND
    # =============================================================

    @staticmethod
    def _get_risk_band(svi_score):

        # Use config.py as the single source of thresholds.

        if svi_score >= RISK_RULES["CRITICAL"]["min_score"]:
            return "CRITICAL"

        if svi_score >= RISK_RULES["HIGH"]["min_score"]:
            return "HIGH"

        if svi_score >= RISK_RULES["MODERATE"]["min_score"]:
            return "MODERATE"

        return "LOW"

    # =============================================================
    # MAIN MULTIMODAL PIPELINE
    # =============================================================

    def process_multimodal_audio(
        self,
        file_path: str,
        filename: str,
        language: str = "English",
    ) -> AudioAnalysisResponse:

        # ---------------------------------------------------------
        # 1. ACOUSTIC FEATURES
        # ---------------------------------------------------------

        acoustic_raw = (
            self.audio_processor.extract_acoustic_features(
                file_path
            )
        )

        # ---------------------------------------------------------
        # 2. SPEECH EMOTION
        # ---------------------------------------------------------

        emotion_predictions = (
            self._classify_emotion(file_path)
        )

        top_emotion = (
            emotion_predictions[0]["label"].upper()
        )

        emotion_confidence = (
            emotion_predictions[0]["score"] * 100
        )

        emotion_stress = (
            self._emotion_stress_score(
                emotion_predictions
            )
        )

        # ---------------------------------------------------------
        # 3. SPEECH TO TEXT
        # ---------------------------------------------------------

        language_map = {
            "English": "en",
            "Hindi": "hi",
            "Telugu": "te",
            "Marathi": "mr",
            "Bengali": "bn",
            "Tamil": "ta",
            "Kannada": "kn",
        }

        whisper_language = language_map.get(
            language
        )

        segments, info = self.whisper_model.transcribe(
            file_path,
            beam_size=5,
            language=whisper_language,
        )

        transcript = " ".join(
            segment.text
            for segment in segments
        ).strip()

        duration = (
            float(info.duration)
            if hasattr(info, "duration")
            else 0.0
        )

        # ---------------------------------------------------------
        # 4. LEXICAL ANALYSIS
        # ---------------------------------------------------------

        (
            matched_triggers,
            lexical_score,
        ) = self._lexical_analysis(
            transcript
        )

        # ---------------------------------------------------------
        # 5. ACOUSTIC SCORE
        # ---------------------------------------------------------

        acoustic_score = (
            self._acoustic_stress_score(
                acoustic_raw
            )
        )

        # ---------------------------------------------------------
        # 6. MULTIMODAL SVI
        # ---------------------------------------------------------

        #
        # ML emotion + acoustic DSP are the primary components.
        #
        # Lexical indicators provide an additional bounded signal.
        #
        # IMPORTANT:
        # A keyword alone can no longer force CRITICAL.
        #

        primary_score = (
            WEIGHT_ML_EMOTION * emotion_stress
            + WEIGHT_ACOUSTIC * acoustic_score
        )

        lexical_contribution = (
            0.20 * lexical_score
        )

        raw_svi = (
            0.80 * primary_score
            + lexical_contribution
        )

        svi_score = round(
            float(
                np.clip(
                    raw_svi * 100,
                    0,
                    100,
                )
            ),
            2,
        )

        # ---------------------------------------------------------
        # 7. RISK BAND
        # ---------------------------------------------------------

        risk_band = self._get_risk_band(
            svi_score
        )

        risk_rule = RISK_RULES[risk_band]

        risk_color = risk_rule["color"]

        # Keep threat_level compatible with current frontend.
        threat_level_map = {
            "LOW": "LOW",
            "MODERATE": "MEDIUM",
            "HIGH": "HIGH",
            "CRITICAL": "HIGH",
        }

        threat_level = threat_level_map[
            risk_band
        ]

        # Existing intervention schema preserved.
        interventions = risk_rule["actions"]

        # ---------------------------------------------------------
        # 8. OVERRIDE INFORMATION
        # ---------------------------------------------------------

        # No automatic keyword emergency override.
        #
        # This field remains in the API for frontend compatibility.

        override_triggered = False

        override_reason = None

        # ---------------------------------------------------------
        # 9. EXPLAINABILITY
        # ---------------------------------------------------------

        top_contributors = []

        if emotion_stress >= 0.50:

            top_contributors.append(
                f"Speech emotion signal: {top_emotion}"
            )

        if acoustic_score >= 0.50:

            top_contributors.append(
                "Elevated acoustic variability"
            )

        if lexical_score >= 0.20:

            top_contributors.append(
                "Distress-related lexical indicators detected"
            )

        if not top_contributors:

            top_contributors.append(
                "No dominant elevated signal detected"
            )

        # ---------------------------------------------------------
        # 10. RESPONSE
        # ---------------------------------------------------------

        return AudioAnalysisResponse(

            status="success",

            filename=filename,

            duration_seconds=round(
                duration,
                2,
            ),

            transcript=(
                transcript
                if transcript
                else "[NO SPEECH DETECTED]"
            ),

            acoustic_indicators=AcousticIndicators(

                top_emotion=top_emotion,

                pitch_volatility=round(
                    float(
                        acoustic_raw.pitch_std
                    ),
                    2,
                ),

                rms_energy=round(
                    float(
                        acoustic_raw.energy_rms
                    ),
                    4,
                ),

                confidence=round(
                    emotion_confidence,
                    1,
                ),
            ),

            nlp_indicators=NLPIndicators(
                speech_emotion=top_emotion,
              threat_level=threat_level,
              flagged_keywords=matched_triggers,
            ),

            svi_metrics=SVIMetrics(

                final_svi_score=svi_score,

                risk_band=risk_band,

                risk_color=risk_color,

                override_triggered=override_triggered,

                override_reason=override_reason,
            ),

            explainability=Explainability(

                summary=(
                    f"Analysis yielded a "
                    f"{risk_band} risk profile "
                    f"with SVI score {svi_score}."
                ),

                top_contributors=top_contributors,
            ),

            recommended_interventions=interventions,
        )