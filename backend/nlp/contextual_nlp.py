"""
backend/contextual_nlp.py

Pluggable Contextual NLP Analysis Module for M3GAN SVI Engine.

Responsibilities:
  - Provides contextual transformer-based text emotion and distress analysis.
  - Pluggable architecture: can load transformer model or fall back cleanly to keyword lexicon.
  - Hybrid fusion helper combining surface keyword severity floors with transformer semantic context.
  - Measure inference latency and confidence metrics.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default cached model identifier
DEFAULT_NLP_MODEL = "j-hartmann/emotion-english-distilroberta-base"


class ContextualNLPAnalyzer:
    _instance = None
    _classifier = None
    _is_initialized = False

    def __init__(self, model_name: str = DEFAULT_NLP_MODEL, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_NLP_MODEL, enabled: bool = True):
        if cls._instance is None:
            cls._instance = ContextualNLPAnalyzer(model_name=model_name, enabled=enabled)
        return cls._instance

    def initialize_model(self) -> bool:
        """Lazy load transformer pipeline safely."""
        if self._is_initialized and self._classifier is not None:
            return True

        if not self.enabled:
            logger.info("Contextual NLP is disabled by configuration.")
            return False

        try:
            from transformers import pipeline
            t0 = time.perf_counter()
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=-1  # CPU inference
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._is_initialized = True
            logger.info(f"Contextual NLP model '{self.model_name}' loaded in {elapsed:.1f} ms.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Contextual NLP model '{self.model_name}': {e}. Falling back to lexicon.")
            self._classifier = None
            self._is_initialized = False
            return False

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using contextual transformer model.

        Returns:
            {
                "available": bool,
                "contextual_distress_score": float (0.0..1.0),
                "primary_emotion": str,
                "emotion_scores": Dict[str, float],
                "model_name": str,
                "latency_ms": float,
            }
        """
        text = (text or "").strip()
        if not text:
            return {
                "available": False,
                "contextual_distress_score": 0.0,
                "primary_emotion": "neutral",
                "emotion_scores": {},
                "model_name": self.model_name,
                "latency_ms": 0.0,
            }

        t0 = time.perf_counter()
        if not self._is_initialized:
            success = self.initialize_model()
            if not success or self._classifier is None:
                return {
                    "available": False,
                    "contextual_distress_score": 0.0,
                    "primary_emotion": "unknown",
                    "emotion_scores": {},
                    "model_name": self.model_name,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        try:
            raw_output = self._classifier(text)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # Format top_k=None list of dicts [{'label': 'fear', 'score': 0.95}, ...]
            scores_dict = {}
            if raw_output and isinstance(raw_output[0], list):
                for item in raw_output[0]:
                    scores_dict[item["label"].lower()] = float(item["score"])
            elif raw_output and isinstance(raw_output, list):
                for item in raw_output:
                    scores_dict[item["label"].lower()] = float(item["score"])

            # Compute distress weight:
            # High distress emotions: fear (1.0), sadness (0.85), anger (0.70), disgust (0.50)
            # Low distress emotions: joy (0.0), neutral (0.0), surprise (0.20)
            fear = scores_dict.get("fear", 0.0)
            sadness = scores_dict.get("sadness", 0.0)
            anger = scores_dict.get("anger", 0.0)
            disgust = scores_dict.get("disgust", 0.0)

            contextual_distress = min(1.0, (fear * 1.0) + (sadness * 0.85) + (anger * 0.70) + (disgust * 0.50))
            primary_emotion = max(scores_dict, key=scores_dict.get) if scores_dict else "neutral"

            return {
                "available": True,
                "contextual_distress_score": round(float(contextual_distress), 4),
                "primary_emotion": primary_emotion,
                "emotion_scores": {k: round(v, 4) for k, v in scores_dict.items()},
                "model_name": self.model_name,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error(f"Error during Contextual NLP inference: {e}")
            return {
                "available": False,
                "contextual_distress_score": 0.0,
                "primary_emotion": "error",
                "emotion_scores": {},
                "model_name": self.model_name,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }


def fuse_hybrid_narrative(lexical_result: dict, contextual_result: dict) -> dict:
    """
    Combines keyword lexicon score with contextual NLP distress score.
    - If contextual model is unavailable: returns 100% lexical score.
    - If contextual model is available: weighted hybrid (70% Lexicon + 30% Contextual NLP),
      while guaranteeing severity floors from genuine disclosures are NEVER suppressed.
    - Applies False Positive Control: damps contextual emotion score when zero lexicon hits
      occur on general administrative/civil text.
    """
    lex_score = lexical_result.get("lexical_score", 0.0)
    sev_floor = lexical_result.get("severity_floor", 0.0)
    hits = lexical_result.get("hits", [])

    if not contextual_result.get("available", False):
        return {
            "narrative_score": lex_score,
            "severity_floor": sev_floor,
            "fusion_mode": "LEXICON_ONLY",
            "contextual_emotion": "UNAVAILABLE",
        }

    ctx_score = contextual_result.get("contextual_distress_score", 0.0)

    # False Positive Control: If zero keyword hits and zero severity floor,
    # damp contextual emotion score by 60% so general anxiety/frustration doesn't cause false alarm.
    if len(hits) == 0 and sev_floor == 0.0:
        damped_ctx_score = ctx_score * 0.40
    else:
        damped_ctx_score = ctx_score

    # Hybrid score: 0.70 * Lexical + 0.30 * Damped Contextual NLP
    hybrid_score = min(1.0, (0.70 * lex_score) + (0.30 * damped_ctx_score))

    return {
        "narrative_score": round(hybrid_score, 4),
        "severity_floor": sev_floor,
        "fusion_mode": "HYBRID_LEXICON_CONTEXTUAL",
        "contextual_emotion": contextual_result.get("primary_emotion", "neutral"),
        "contextual_score": ctx_score,
        "damped_contextual_score": round(damped_ctx_score, 4),
    }
