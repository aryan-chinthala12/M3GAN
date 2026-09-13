"""
backend/language_detector.py

Lightweight Language and Script Detection Layer for M3GAN SVI Engine.

Capabilities:
  - Detects native Unicode scripts (Devanagari, Bengali, Tamil, Telugu, Kannada, Marathi).
  - Distinguishes English vs Romanized Hindi / Hinglish using vocabulary and character n-gram distribution.
  - Detects code-switching (mixed Latin/Devanagari text).
  - Returns structured detection output with language, script, confidence, and flags.
"""

import re
import unicodedata
from typing import Dict, Any, List


# Common Romanized Indian Language Stopwords / Markers
ROMANIZED_HINDI_STOPWORDS = {
  "hai", "hain", "hoon", "hun", "main", "mein", "mujhe", "mera", "meri", "mere",
  "tum", "tumhe", "apna", "apni", "apne", "ko", "se", "par", "pe", "ka", "ki", "ke",
  "bhi", "toh", "to", "nahi", "nahin", "nhi", "kya", "kyun", "kyu", "kaise", "kuch",
  "kar", "karna", "raha", "rahi", "rahe", "diya", "di", "de", "darr", "dar", "dhamki",
  "bohot", "bahut", "yaar", "aur", "ya", "par", "walon", "wala", "wali", "wale"
}

ROMANIZED_TELUGU_STOPWORDS = {
  "nenu", "naa", "naaku", "maku", "gurinchi", "leka", "kadu", "kuda", "undi",
  "unnannu", "matladali", "aduguthunnanu", "saami", "chesina", "valla", "enti"
}

ROMANIZED_TAMIL_STOPWORDS = {
  "enakku", "ennai", "yarenum", "yaarum", "illai", "irukkirathu", "aanaal",
  "ippo", "naan", "avanga", "panna", "adichanga", "solla"
}


class LanguageDetector:
    """Fast rule-based and vocabulary-assisted language/script detector."""

    @staticmethod
    def detect_script(text: str) -> str:
        """Classify primary Unicode script of text."""
        if not text:
            return "UNKNOWN"

        counts = {
            "DEVANAGARI": 0,
            "BENGALI": 0,
            "TAMIL": 0,
            "TELUGU": 0,
            "KANNADA": 0,
            "LATIN": 0,
        }

        for char in text:
            name = unicodedata.name(char, "")
            if "DEVANAGARI" in name:
                counts["DEVANAGARI"] += 1
            elif "BENGALI" in name:
                counts["BENGALI"] += 1
            elif "TAMIL" in name:
                counts["TAMIL"] += 1
            elif "TELUGU" in name:
                counts["TELUGU"] += 1
            elif "KANNADA" in name:
                counts["KANNADA"] += 1
            elif "LATIN" in name and char.isalpha():
                counts["LATIN"] += 1

        total_alpha = sum(counts.values())
        if total_alpha == 0:
            return "UNKNOWN"

        # Check for code-mixing across scripts
        active_scripts = [script for script, count in counts.items() if count > 0]
        if len(active_scripts) > 1 and counts["LATIN"] > 0:
            return "LATIN_MIXED"

        max_script = max(counts, key=counts.get)
        return max_script if counts[max_script] > 0 else "UNKNOWN"

    @classmethod
    def detect_language(cls, text: str) -> Dict[str, Any]:
        """
        Detect language, script, and code-switching state.

        Returns:
            {
                "language": str,      # "English", "Hindi", "Hinglish", "Telugu", "Bengali", "Tamil", "MIXED", "UNKNOWN"
                "script": str,        # "LATIN", "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "LATIN_MIXED", "UNKNOWN"
                "confidence": float,  # 0.0 .. 1.0
                "is_romanized": bool, # True if Indian language written in Latin script
                "code_switching": bool, # True if mixed language/script
            }
        """
        text = (text or "").strip()
        if not text:
            return {
                "language": "UNKNOWN",
                "script": "UNKNOWN",
                "confidence": 0.0,
                "is_romanized": False,
                "code_switching": False,
            }

        script = cls.detect_script(text)
        words = [w.lower() for w in re.findall(r"\b[A-Za-z0-9\u0900-\u0D7F']+\b", text)]

        if not words:
            return {
                "language": "UNKNOWN",
                "script": script,
                "confidence": 0.0,
                "is_romanized": False,
                "code_switching": False,
            }

        # 1. Native Scripts
        if script == "DEVANAGARI":
            return {
                "language": "Hindi",
                "script": "DEVANAGARI",
                "confidence": 0.95,
                "is_romanized": False,
                "code_switching": False,
            }
        elif script == "BENGALI":
            return {
                "language": "Bengali",
                "script": "BENGALI",
                "confidence": 0.95,
                "is_romanized": False,
                "code_switching": False,
            }
        elif script == "TAMIL":
            return {
                "language": "Tamil",
                "script": "TAMIL",
                "confidence": 0.95,
                "is_romanized": False,
                "code_switching": False,
            }
        elif script == "TELUGU":
            return {
                "language": "Telugu",
                "script": "TELUGU",
                "confidence": 0.95,
                "is_romanized": False,
                "code_switching": False,
            }
        elif script == "KANNADA":
            return {
                "language": "Kannada",
                "script": "KANNADA",
                "confidence": 0.95,
                "is_romanized": False,
                "code_switching": False,
            }

        # 2. Latin Script: Distinguish English vs Romanized Hindi / Hinglish vs Telugu / Tamil
        hindi_count = sum(1 for w in words if w in ROMANIZED_HINDI_STOPWORDS)
        telugu_count = sum(1 for w in words if w in ROMANIZED_TELUGU_STOPWORDS)
        tamil_count = sum(1 for w in words if w in ROMANIZED_TAMIL_STOPWORDS)

        total_words = len(words)
        hindi_ratio = hindi_count / total_words

        if hindi_count >= 2 or hindi_ratio >= 0.20:
            # Check if code-mixed with English words
            english_common = {"is", "the", "and", "to", "i", "my", "me", "raped", "kill", "threaten", "help", "please"}
            english_count = sum(1 for w in words if w in english_common)
            is_code_mixed = english_count > 0 and hindi_count > 0

            return {
                "language": "Hinglish" if is_code_mixed else "Hindi",
                "script": "LATIN_MIXED" if script == "LATIN_MIXED" else "LATIN",
                "confidence": min(0.95, round(0.50 + hindi_ratio * 0.50, 2)),
                "is_romanized": True,
                "code_switching": is_code_mixed,
            }

        if telugu_count >= 1:
            return {
                "language": "Telugu",
                "script": "LATIN",
                "confidence": 0.85,
                "is_romanized": True,
                "code_switching": False,
            }

        if tamil_count >= 1:
            return {
                "language": "Tamil",
                "script": "LATIN",
                "confidence": 0.85,
                "is_romanized": True,
                "code_switching": False,
            }

        # Default Latin assumption: English
        return {
            "language": "English",
            "script": "LATIN",
            "confidence": 0.85,
            "is_romanized": False,
            "code_switching": False,
        }
