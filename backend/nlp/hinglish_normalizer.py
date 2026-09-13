"""
backend/hinglish_normalizer.py

Phase 5B Romanized Hindi & Hinglish Phonetic Normalization Module for M3GAN.

Responsibilities:
  - Normalizes common phonetic spelling variations in Romanized Hindi, Hinglish, Telugu, Tamil, and Bengali.
  - Standardizes surface forms for distress terms so multilingual lexicon and contextual NLP match accurately.
  - Preserves original transcript without in-place mutation.
"""

import re
from typing import Dict, Any
from backend.nlp.language_detector import LanguageDetector


# Phonetic Standardization Replacements (Case-Insensitive Regex)
HINGLISH_PHONETIC_RULES = [
    # Suicidal ideation variants
    (r"\b(aatmahatya|aatmahatye|atmahatya|aatma\s*hatya)\b", "atmahatya"),
    (r"\b(khudkhushi|khud-khushi|khud_khushi|khudkusi)\b", "khudkushi"),
    (r"\b(marana\s*chaha?ti|marna\s*chahati|marna\s*chahata|marna\s*chahti)\b", "marna chahta"),
    (r"\b(jaan\s*de\s*dunga|jaan\s*de\s*dungi|jaan\s*dena)\b", "jaan de dunga"),

    # Threat & Violence variants
    (r"\b(dhamkee|dhamaki|dhamkiya|dhamkiyan)\b", "dhamki"),
    (r"\b(maar\s*dalenge|mar\s*dalenge|maardenge|mar\s*denge)\b", "mar dalenge"),
    (r"\b(maar\s*diya|mar\s*diya|maar\s*peet|marpeet)\b", "mar diya"),
    (r"\b(darr|dar|dara|darrr)\b", "dar"),

    # Sexual violence variants
    (r"\b(balatkaar|balatkaara|blatkar|raap)\b", "balatkar"),
    (r"\b(dushkarm|dushkarma)\b", "dushkarm"),

    # General distress variants
    (r"\b(bohot|bohat|bhat)\b", "bahut"),
    (r"\b(madadha|madath|maddad)\b", "madad"),
    (r"\b(shikaayata|shikayat)\b", "shikayat"),

    # Telugu/Tamil phonetic standardization
    (r"\b(champethanu|chathipota|champesaru)\b", "champesaru"),
    (r"\b(adichanga|adithanga|adichangha)\b", "adichanga"),
]


class HinglishNormalizer:
    """Phonetic normalizer for Romanized Indian text."""

    @staticmethod
    def normalize_text(text: str) -> Dict[str, Any]:
        """
        Normalize Romanized Hindi / Hinglish distress terms.

        Returns:
            {
                "original_transcript": str,
                "normalized_transcript": str,
                "language": str,
                "script": str,
                "normalization_applied": bool,
                "method": str,
            }
        """
        raw_text = (text or "").strip()
        if not raw_text:
            return {
                "original_transcript": "",
                "normalized_transcript": "",
                "language": "UNKNOWN",
                "script": "UNKNOWN",
                "normalization_applied": False,
                "method": "none",
            }

        det = LanguageDetector.detect_language(raw_text)
        lang = det["language"]
        script = det["script"]

        normalized = raw_text
        applied = False

        # Only apply Romanized phonetic normalization if text contains Latin script
        if "LATIN" in script or det["is_romanized"]:
            for pattern, replacement in HINGLISH_PHONETIC_RULES:
                new_text, count = re.subn(pattern, replacement, normalized, flags=re.IGNORECASE)
                if count > 0:
                    normalized = new_text
                    applied = True

        return {
            "original_transcript": raw_text,
            "normalized_transcript": normalized,
            "language": lang,
            "script": script,
            "normalization_applied": applied,
            "method": "phonetic_rules" if applied else "passthrough",
        }
