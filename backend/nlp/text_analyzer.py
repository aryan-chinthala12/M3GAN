"""
backend/text_analyzer.py

Shared narrative analysis used by BOTH the audio pipeline (post-STT)
and the direct text pipeline (chat / portal / chatbot / IVRS transcript).

Responsibilities:
  - PII redaction (privacy by default)
  - Multilingual distress-lexicon scanning (via backend.lexicon)
  - Bounded lexical score computation
  - Severity-floor lookup for grounded high-severity disclosures
"""

import re
from typing import List, Tuple

from backend.core.config import LEXICAL_SATURATION_CAP
from backend.nlp.lexicon import scan_text, max_category_floor
from backend.nlp.contextual_nlp import ContextualNLPAnalyzer, fuse_hybrid_narrative
from backend.nlp.language_detector import LanguageDetector
from backend.nlp.hinglish_normalizer import HinglishNormalizer


def redact(text: str) -> Tuple[str, int]:
    """PII redaction wrapper. Returns (redacted_text, redaction_count)."""
    from backend.nlp.lexicon import redact_pii
    return redact_pii(text)


def analyze_transcript(text: str, enable_contextual: bool = True) -> dict:
    """
    Run the full multilingual narrative analysis on raw text.
    Pipeline: PII Redaction -> Language/Script Detection -> Hinglish Normalization -> Lexicon -> Contextual NLP -> Hybrid Fusion
    """
    text = (text or "").strip()

    if not text:
        return {
            "hits": [],
            "flagged_keywords": [],
            "categories": [],
            "lexical_score": 0.0,
            "severity_floor": 0.0,
            "floor_categories": [],
            "contextual": {"available": False, "contextual_distress_score": 0.0, "primary_emotion": "neutral"},
            "narrative_score": 0.0,
            "language": "UNKNOWN",
            "script": "UNKNOWN",
            "original_text": "",
            "normalized_text": "",
            "normalization_applied": False,
        }

    # 1. PII Redaction (Privacy by default)
    redacted_text, red_count = redact(text)

    # 2. Language/Script Detection & Hinglish Normalization
    norm_info = HinglishNormalizer.normalize_text(redacted_text)
    target_text = norm_info["normalized_transcript"]
    lang = norm_info["language"]
    script = norm_info["script"]

    # 3. Multilingual Lexicon Scanning
    hits = scan_text(target_text)
    word_count = max(len(target_text.split()), 1)
    weighted_sum = sum(hit["weight"] for hit in hits)

    lexical_score = min(
        weighted_sum / max(word_count ** 0.35, 1.0),
        LEXICAL_SATURATION_CAP,
    )

    severity_floor, floor_categories = max_category_floor(hits)

    # 4. Contextual Transformer NLP Analysis
    contextual_res = {"available": False, "contextual_distress_score": 0.0, "primary_emotion": "neutral"}
    if enable_contextual:
        analyzer = ContextualNLPAnalyzer.get_instance(enabled=True)
        contextual_res = analyzer.analyze_text(target_text)

    # 5. Hybrid Fusion
    lex_res = {
        "hits": hits,
        "lexical_score": round(float(lexical_score), 4),
        "severity_floor": float(severity_floor),
    }
    hybrid_res = fuse_hybrid_narrative(lex_res, contextual_res)

    return {
        "hits": hits,
        "flagged_keywords": [hit["term"] for hit in hits],
        "categories": sorted({hit["category"] for hit in hits}),
        "lexical_score": round(float(lexical_score), 4),
        "severity_floor": float(severity_floor),
        "floor_categories": floor_categories,
        "contextual": contextual_res,
        "narrative_score": hybrid_res["narrative_score"],
        "language": lang,
        "script": script,
        "original_text": text,
        "normalized_text": target_text,
        "normalization_applied": norm_info["normalization_applied"],
    }


def threat_level_from_band(risk_band: str) -> str:
    """Map risk band to the legacy threat-level string the frontend expects."""
    return {
        "LOW": "LOW",
        "MODERATE": "MEDIUM",
        "HIGH": "HIGH",
        "CRITICAL": "HIGH",
    }.get(risk_band, "LOW")

