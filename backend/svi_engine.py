import os
import re
import torch
import numpy as np

# Disable symlinks for Hugging Face on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline
from faster_whisper import WhisperModel

from backend.audio_processor import AudioProcessor
from backend.config import (
    ACOUSTIC_PITCH_WEIGHT,
    ACOUSTIC_RMS_WEIGHT,
    PITCH_VOLATILITY_NORMALIZATION,
    RISK_RULES,
    RMS_NORMALIZATION,
    WEIGHT_ACOUSTIC,
    WEIGHT_EMOTION,
    WEIGHT_LINGUISTIC,
)
from backend.schemas import (
    AudioAnalysisResponse,
    AcousticIndicators,
    NLPIndicators,
    SVIMetrics,
    Explainability
)


class SVIEngine:
    """
    Multimodal Speech Vulnerability Index engine.

    SVI is an engineering-based distress indicator from 0-100.
    It is NOT a clinical diagnosis or validated psychological scale.

    Components:
        45% - Speech emotion
        30% - Acoustic characteristics
        25% - Linguistic distress indicators
    """

    # =========================================================
    # SVI WEIGHTS
    # =========================================================

    EMOTION_WEIGHT = WEIGHT_EMOTION
    ACOUSTIC_WEIGHT = WEIGHT_ACOUSTIC
    LINGUISTIC_WEIGHT = WEIGHT_LINGUISTIC

    # =========================================================
    # EMOTION → DISTRESS MAPPING
    # =========================================================

    EMOTION_DISTRESS_WEIGHTS = {
        "SAD": 0.75,
        "SADNESS": 0.75,

        "FEAR": 0.90,
        "FEARFUL": 0.90,

        "ANGRY": 0.65,
        "ANGER": 0.65,

        "DISGUST": 0.50,

        "SURPRISED": 0.30,
        "SURPRISE": 0.30,

        "NEUTRAL": 0.05,

        "HAPPY": 0.05,
        "HAPPINESS": 0.05,

        "CALM": 0.05,
    }

    # =========================================================
    # LINGUISTIC KEYWORDS
    # =========================================================

    # Ordinary distress indicators.
    # These increase SVI but do NOT automatically make the
    # assessment CRITICAL.
    DISTRESS_KEYWORDS = {
        "depressed": 0.45,
        "help": 0.20,
        "pain": 0.25,
        "alone": 0.20,
        "hopeless": 0.60,
        "hurt": 0.35,
        "harm": 0.45,
        "bleeding": 0.60,
        "die": 0.70,
    }

    # High-risk safety indicators.
    # These trigger the safety override.
    CRITICAL_KEYWORDS = {
        "suicide": 1.00,
        "suicidal": 1.00,
        "kill myself": 1.00,
        "kill me": 1.00,
        "end my life": 1.00,
        "end it all": 0.95,
        "overdose": 0.95,
        "self harm": 0.90,
        "self-harm": 0.90,
    }

    def __init__(self):
        print("[INFO] Initializing SVI Multimodal Engine...")

        # -----------------------------------------------------
        # Audio Preprocessing
        # -----------------------------------------------------

        self.audio_processor = AudioProcessor()

        # -----------------------------------------------------
        # Speech Emotion Recognition
        # -----------------------------------------------------

        print("[INFO] Loading Wav2Vec 2.0 Speech Emotion Transformer...")

        self.speech_classifier = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        self.emotion_class_count = len(
            getattr(self.speech_classifier.model.config, "id2label", {})
        )

        # -----------------------------------------------------
        # Faster-Whisper Speech-to-Text
        # -----------------------------------------------------

        print("[INFO] Loading Faster-Whisper STT Engine...")

        self.whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )

    # =========================================================
    # EMOTION SCORE
    # =========================================================

    def _calculate_emotion_distress(self, predictions):
        """
        Convert the complete emotion probability distribution
        into a normalized distress score between 0 and 1.

        Example:

            fear     0.70
            neutral  0.20
            sad      0.10

        becomes approximately:

            0.70*0.90 + 0.20*0.05 + 0.10*0.75

        rather than simply treating the top emotion confidence
        as a stress percentage.
        """

        score = 0.0

        for prediction in predictions:
            label = prediction["label"].strip().upper()
            probability = float(prediction["score"])

            distress_weight = self.EMOTION_DISTRESS_WEIGHTS.get(
                label,
                0.10
            )

            score += probability * distress_weight

        return float(
            np.clip(score, 0.0, 1.0)
        )

    # =========================================================
    # ACOUSTIC SCORE
    # =========================================================

    def _calculate_acoustic_stress(self, acoustic_raw):
        """
        Convert acoustic features into a normalized 0-1 score.

        Pitch volatility is the strongest acoustic component.

        RMS energy is deliberately given less importance because
        microphone distance and recording volume can dramatically
        change absolute RMS values.
        """

        pitch_volatility = float(
            getattr(acoustic_raw, "pitch_std", 0.0) or 0.0
        )

        rms_energy = float(
            getattr(acoustic_raw, "energy_rms", 0.0) or 0.0
        )

        # -----------------------------------------------------
        # Pitch volatility
        # -----------------------------------------------------

        # Pitch volatility is already expressed in semitones relative to the
        # speaker's median F0 by AudioProcessor.
        pitch_component = np.clip(
            pitch_volatility / PITCH_VOLATILITY_NORMALIZATION,
            0.0,
            1.0
        )

        # -----------------------------------------------------
        # RMS energy
        # -----------------------------------------------------

        # Keep this contribution deliberately small.
        energy_component = np.clip(
            rms_energy / RMS_NORMALIZATION,
            0.0,
            1.0
        )

        # -----------------------------------------------------
        # Final acoustic score
        # -----------------------------------------------------

        acoustic_score = (
            ACOUSTIC_PITCH_WEIGHT * pitch_component
            + ACOUSTIC_RMS_WEIGHT * energy_component
        )

        return float(
            np.clip(
                acoustic_score,
                0.0,
                1.0
            )
        )

    # =========================================================
    # KEYWORD MATCHING
    # =========================================================

    @staticmethod
    def _is_negated_or_contextual_reference(text, match, keyword):
        """Reject transparent negations and prevention/discussion references.

        This is intentionally limited rule-based protection, not clinical NLP.
        """
        sentence_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        ) + 1
        sentence_end_candidates = [
            position for position in (
                text.find(".", match.end()),
                text.find("!", match.end()),
                text.find("?", match.end()),
                text.find("\n", match.end()),
            ) if position != -1
        ]
        sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(text)
        sentence = text[sentence_start:sentence_end]
        escaped_keyword = re.escape(keyword)

        negation_patterns = (
            rf"\b(?:not|never|without)\b(?:\W+\w+){{0,4}}\W+{escaped_keyword}\b",
            rf"\b(?:do not|don't|does not|doesn't|did not|didn't)\b"
            rf"(?:\W+\w+){{0,5}}\W+{escaped_keyword}\b",
        )
        if any(re.search(pattern, sentence) for pattern in negation_patterns):
            return True

        is_prevention_reference = re.search(
            r"\b(?:suicide|suicidal|self[-\s]harm)\s+"
            r"(?:prevention|awareness|education)\b",
            sentence,
        )
        is_discussion = re.search(
            r"\b(?:discuss(?:ed|ing)?|talk(?:ed|ing)?\s+about|"
            r"learn(?:ed|ing)?\s+about)\b",
            sentence,
        )
        return bool(is_prevention_reference and is_discussion)

    def _find_keyword_matches(self, transcript):
        """
        Find distress and critical keywords using word-aware
        matching rather than simple substring matching.

        This prevents cases such as:

            "skilled"

        accidentally matching:

            "kill"
        """

        text = (transcript or "").lower()

        distress_matches = []
        critical_matches = []

        # -----------------------------------------------------
        # Ordinary distress keywords
        # -----------------------------------------------------

        for keyword, weight in self.DISTRESS_KEYWORDS.items():

            pattern = rf"\b{re.escape(keyword)}\b"

            if re.search(pattern, text):
                distress_matches.append(
                    (keyword, weight)
                )

        # -----------------------------------------------------
        # Critical phrases
        # -----------------------------------------------------

        for keyword, weight in self.CRITICAL_KEYWORDS.items():

            # Phrases such as "kill myself" need normal
            # whitespace matching.
            pattern = rf"\b{re.escape(keyword)}\b"

            for match in re.finditer(pattern, text):
                if not self._is_negated_or_contextual_reference(
                    text, match, keyword
                ):
                    critical_matches.append((keyword, weight))
                    break

        return distress_matches, critical_matches

    # =========================================================
    # LINGUISTIC SCORE
    # =========================================================

    def _calculate_linguistic_score(
        self,
        transcript,
        distress_matches,
        critical_matches
    ):
        """
        Convert linguistic indicators into a normalized 0-1 score.

        Critical phrases receive a very high score.

        Ordinary distress keywords contribute according to:
            - keyword severity
            - number of distinct indicators
            - transcript length
        """

        text = (transcript or "").strip()

        if not text:
            return 0.0

        word_count = max(
            len(text.split()),
            1
        )

        # -----------------------------------------------------
        # Critical indicators
        # -----------------------------------------------------

        if critical_matches:

            strongest_critical = max(
                weight
                for _, weight in critical_matches
            )

            # Critical language should strongly influence SVI.
            # The safety override is handled separately.
            critical_score = (
                0.80
                + 0.20 * strongest_critical
            )

            return float(
                np.clip(
                    critical_score,
                    0.0,
                    1.0
                )
            )

        # -----------------------------------------------------
        # Ordinary distress indicators
        # -----------------------------------------------------

        if not distress_matches:
            return 0.0

        weighted_sum = sum(
            weight
            for _, weight in distress_matches
        )

        # More indicators increase severity, but the score is
        # bounded so that a long transcript does not automatically
        # become critical.
        indicator_component = np.clip(
            weighted_sum / 1.5,
            0.0,
            1.0
        )

        # Short statements containing distress language deserve
        # slightly more weight than very long transcripts.
        length_factor = np.clip(
            20.0 / word_count,
            0.35,
            1.0
        )

        lexical_score = (
            0.75 * indicator_component
            + 0.25 * length_factor
        )

        return float(
            np.clip(
                lexical_score,
                0.0,
                1.0
            )
        )

    # =========================================================
    # RISK BAND
    # =========================================================

    def _determine_risk_band(self, svi_score, override_triggered):
        """
        Determine risk level.

        Critical safety language overrides the numerical SVI.
        """

        if override_triggered or svi_score >= RISK_RULES["CRITICAL"]["min_score"]:
            return (
                "CRITICAL",
                RISK_RULES["CRITICAL"]["color"],
                "HIGH",
                [
                    "IMMEDIATE_HUMAN_DISPATCH",
                    "ALERT_SAFETY_TEAM"
                ]
            )

        elif svi_score >= RISK_RULES["HIGH"]["min_score"]:
            return (
                "HIGH",
                RISK_RULES["HIGH"]["color"],
                "HIGH",
                [
                    "ESCALATE_TO_SENIOR_SUPERVISOR",
                    "PRIORITIZE_COUNSELOR_FOLLOWUP"
                ]
            )

        elif svi_score >= RISK_RULES["MODERATE"]["min_score"]:
            return (
                "MODERATE",
                RISK_RULES["MODERATE"]["color"],
                "MEDIUM",
                [
                    "OPERATOR_MONITORING",
                    "COUNSELOR_FOLLOWUP"
                ]
            )

        else:
            return (
                "LOW",
                RISK_RULES["LOW"]["color"],
                "LOW",
                [
                    "ROUTINE_LOGGING"
                ]
            )

    # =========================================================
    # MAIN MULTIMODAL PROCESSING
    # =========================================================

    def process_multimodal_audio(
        self,
        file_path: str,
        filename: str
    ) -> AudioAnalysisResponse:

        """
        Complete multimodal assessment pipeline:

            Audio
              ↓
            Acoustic features
              ↓
            Wav2Vec2 emotion
              ↓
            Whisper transcript
              ↓
            Linguistic analysis
              ↓
            Multimodal SVI
              ↓
            Risk band
        """

        # =====================================================
        # 1. ACOUSTIC SIGNAL PROCESSING
        # =====================================================

        acoustic_raw = (
            self.audio_processor
            .extract_acoustic_features(file_path)
        )

        # =====================================================
        # 2. SPEECH EMOTION RECOGNITION
        # =====================================================

        emotion_predictions = self.speech_classifier(
            file_path,
            top_k=self.emotion_class_count or None,
        )

        if not emotion_predictions:
            emotion_predictions = [
                {
                    "label": "NEUTRAL",
                    "score": 1.0
                }
            ]

        print("\n[EMOTION DEBUG]")
        for prediction in emotion_predictions:
            print(
                f"  {prediction['label'].upper():12} "
                f": {float(prediction['score']) * 100:.2f}%"
            )

        top_emotion = (
            emotion_predictions[0]["label"]
            .strip()
            .upper()
        )

        # This is model confidence, NOT stress.
        emotion_confidence = float(
            emotion_predictions[0]["score"]
        ) * 100.0

        emotion_distress = (
            self._calculate_emotion_distress(
                emotion_predictions
            )
        )

        # =====================================================
        # 3. WHISPER SPEECH-TO-TEXT
        # =====================================================

        segments, info = self.whisper_model.transcribe(
            file_path,
            beam_size=5
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

        # =====================================================
        # 4. LINGUISTIC ANALYSIS
        # =====================================================

        distress_matches, critical_matches = (
            self._find_keyword_matches(
                transcript
            )
        )

        linguistic_score = (
            self._calculate_linguistic_score(
                transcript,
                distress_matches,
                critical_matches
            )
        )

        # All keywords shown to frontend.
        matched_triggers = [
            keyword
            for keyword, _ in distress_matches
        ] + [
            keyword
            for keyword, _ in critical_matches
        ]

        # Remove duplicates while preserving order.
        matched_triggers = list(
            dict.fromkeys(matched_triggers)
        )

        # =====================================================
        # 5. ACOUSTIC DISTRESS SCORE
        # =====================================================

        acoustic_stress = (
            self._calculate_acoustic_stress(
                acoustic_raw
            )
        )

        # =====================================================
        # 6. MULTIMODAL SVI
        # =====================================================

        # IMPORTANT:
        #
        # Emotion       = 45%
        # Acoustic      = 30%
        # Linguistic    = 25%
        #
        # The weights sum to exactly 1.0.

        raw_svi = (
            self.EMOTION_WEIGHT * emotion_distress
            + self.ACOUSTIC_WEIGHT * acoustic_stress
            + self.LINGUISTIC_WEIGHT * linguistic_score
        )

        svi_score = round(
            float(
                np.clip(
                    raw_svi * 100.0,
                    0.0,
                    100.0
                )
            ),
            2
        )

        # =====================================================
        # 7. SAFETY OVERRIDE
        # =====================================================

        override_triggered = (
            len(critical_matches) > 0
        )

        if override_triggered:

            override_reason = (
                "High-risk safety language detected "
                "in transcript"
            )

        else:

            override_reason = None

        # =====================================================
        # 8. RISK BANDING
        # =====================================================

        (
            risk_band,
            risk_color,
            threat_level,
            interventions
        ) = self._determine_risk_band(
            svi_score,
            override_triggered
        )

        # =====================================================
        # 9. EXPLAINABILITY
        # =====================================================

        emotion_contribution = self.EMOTION_WEIGHT * emotion_distress * 100.0
        acoustic_contribution = self.ACOUSTIC_WEIGHT * acoustic_stress * 100.0
        linguistic_contribution = self.LINGUISTIC_WEIGHT * linguistic_score * 100.0
        top_contributors = [
            f"Emotion distress contribution: {emotion_contribution:.1f} SVI points ({top_emotion})",
            f"Acoustic contribution: {acoustic_contribution:.1f} SVI points",
            f"Linguistic contribution: {linguistic_contribution:.1f} SVI points",
        ]
        if override_triggered:
            top_contributors.append(
                "Safety override: explicit high-risk self-harm language detected"
            )

        # =====================================================
        # 10. DEBUG OUTPUT
        # =====================================================

        print(
            "\n"
            "[SVI DEBUG]\n"
            f"  Top emotion          : {top_emotion}\n"
            f"  Emotion confidence   : {emotion_confidence:.2f}%\n"
            f"  Emotion distress     : {emotion_distress:.3f}\n"
            f"  Acoustic score       : {acoustic_stress:.3f}\n"
            f"  Linguistic score     : {linguistic_score:.3f}\n"
            f"  Critical override    : {override_triggered}\n"
            f"  Final SVI            : {svi_score:.2f}\n"
            f"  Risk band            : {risk_band}\n"
        )

        # =====================================================
        # 11. API RESPONSE
        # =====================================================

        return AudioAnalysisResponse(

            status="success",

            filename=filename,

            duration_seconds=round(
                duration,
                2
            ),

            transcript=(
                transcript
                if transcript
                else "[NO SPEECH DETECTED]"
            ),

            acoustic_indicators=AcousticIndicators(

                top_emotion=top_emotion,

                pitch_volatility=round(
                    float(acoustic_raw.pitch_std),
                    2
                ),

                rms_energy=round(
                    float(acoustic_raw.energy_rms),
                    4
                ),

                # This remains emotion MODEL CONFIDENCE,
                # not SVI or stress percentage.
                confidence=round(
                    emotion_confidence,
                    1
                )
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

                summary=(
                    f"Multimodal screening yielded a {risk_band} risk profile "
                    f"with SVI score {svi_score}; this is not a clinical diagnosis."
                ),

                top_contributors=top_contributors
            ),

            recommended_interventions=interventions
        )
