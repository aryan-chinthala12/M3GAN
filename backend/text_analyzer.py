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

from backend.config import LEXICAL_SATURATION_CAP
from backend.lexicon import scan_text, max_category_floor


def redact(text: str) -> Tuple[str, int]:
    """PII redaction wrapper. Returns (redacted_text, redaction_count)."""
    from backend.lexicon import redact_pii
    return redact_pii(text)


def analyze_transcript(text: str) -> dict:
    """
    Run the full multilingual narrative analysis on raw text.

    Returns:
        {
            "hits": [...],               # matched lexicon entries
            "flagged_keywords": [...],   # surface terms for the UI
            "categories": [...],         # human-readable category labels
            "lexical_score": float,      # bounded 0..1
            "severity_floor": float,     # 0..100, may be 0.0
            "floor_categories": [...],   # categories that imposed a floor
        }
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
        }

    hits = scan_text(text)

    word_count = max(len(text.split()), 1)

    # Bounded lexical strength: sum of weighted hits, damped by message
    # length so long transcripts do not automatically max out the score.
    # sqrt damping mirrors established distress-density heuristics.
    weighted_sum = sum(hit["weight"] for hit in hits)

    lexical_score = min(
        weighted_sum / max(word_count ** 0.35, 1.0),
        LEXICAL_SATURATION_CAP,
    )

    # Severity floor: genuine high-severity disclosures (suicidal
    # ideation, sexual violence, death threats, self-harm) must keep
    # the SVI grounded even if the caller speaks softly or briefly.
    severity_floor, floor_categories = max_category_floor(hits)

    return {
        "hits": hits,
        "flagged_keywords": [hit["term"] for hit in hits],
        "categories": sorted({hit["category"] for hit in hits}),
        "lexical_score": round(float(lexical_score), 4),
        "severity_floor": float(severity_floor),
        "floor_categories": floor_categories,
    }


def threat_level_from_band(risk_band: str) -> str:
    """Map risk band to the legacy threat-level string the frontend expects."""
    return {
        "LOW": "LOW",
        "MODERATE": "MEDIUM",
        "HIGH": "HIGH",
        "CRITICAL": "HIGH",
    }.get(risk_band, "LOW")
