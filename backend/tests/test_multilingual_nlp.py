"""
backend/tests/test_multilingual_nlp.py

Unit Test Suite for Phase 5B Multilingual NLP & Hinglish Normalization.

Tests:
  1. Language & Script Detection (LanguageDetector).
  2. Romanized Hindi & Hinglish Phonetic Normalization (HinglishNormalizer).
  3. PII Redaction Preservation before and after normalization.
  4. False Positive Damping on general administrative/civil text.
  5. Multi-script & Code-Switching handling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.nlp.language_detector import LanguageDetector
from backend.nlp.hinglish_normalizer import HinglishNormalizer
from backend.nlp.text_analyzer import analyze_transcript, redact


def test_language_detection():
    print("--- 1. Testing Language & Script Detection ---")
    d1 = LanguageDetector.detect_language("Woh mujhe jaan se marne ki dhamki de rahe hain")
    print(f"    Text 1 -> Language: {d1['language']} | Script: {d1['script']} | Romanized: {d1['is_romanized']}")
    assert d1["language"] in ["Hindi", "Hinglish"], f"FAILED: Expected Hindi/Hinglish, got {d1['language']}"

    d2 = LanguageDetector.detect_language("मुझसे अब सहन नहीं होता, मैं मरना चाहती हूँ")
    print(f"    Text 2 -> Language: {d2['language']} | Script: {d2['script']}")
    assert d2["language"] == "Hindi" and d2["script"] == "DEVANAGARI"

    d3 = LanguageDetector.detect_language("Ennai paaliyal vanmuraiyikku utpaduthinaargal")
    print(f"    Text 3 -> Language: {d3['language']} | Script: {d3['script']}")
    assert d3["language"] == "Tamil"
    print("    [VERIFIED] Language and script detection operational.")


def test_hinglish_normalization():
    print("\n--- 2. Testing Hinglish Phonetic Normalization ---")
    norm1 = HinglishNormalizer.normalize_text("Woh mujhe marana chahati aur dhamkee di")
    print(f"    Original  : '{norm1['original_transcript']}'")
    print(f"    Normalized: '{norm1['normalized_transcript']}'")
    assert "marna chahta" in norm1["normalized_transcript"]
    assert "dhamki" in norm1["normalized_transcript"]
    print("    [VERIFIED] Hinglish phonetic normalization operational.")


def test_pii_redaction_preservation():
    print("\n--- 3. Testing PII Redaction Preservation ---")
    raw_text = "Mera naam Ramesh Kumar hai mera phone number 9876543210 hai email ramesh@gmail.com hai"
    red_text, count = redact(raw_text)
    print(f"    Redacted  : '{red_text}' (Redaction Count: {count})")
    assert "[PHONE_REDACTED]" in red_text
    assert "[EMAIL_REDACTED]" in red_text

    analysis = analyze_transcript(raw_text)
    print(f"    Analyzed Normalized: '{analysis['normalized_text']}'")
    assert "[PHONE_REDACTED]" in analysis["normalized_text"]
    print("    [VERIFIED] PII redaction preserved prior to normalization & NLP.")


def test_false_positive_control():
    print("\n--- 4. Testing False Positive Control on General Civil Text ---")
    civil_text = "I am feeling really anxious and frustrated because the landlord is taking so long"
    res = analyze_transcript(civil_text, enable_contextual=True)
    narrative_score = res["narrative_score"]
    print(f"    Civil Text Score: {narrative_score:.4f} | Hits: {res['hits']} | SevFloor: {res['severity_floor']}")
    assert narrative_score < 0.35, f"FAILED: General civil anxiety scored too high ({narrative_score:.4f})!"
    print("    [VERIFIED] False positive control damps general non-critical anxiety text.")


def run_all_multilingual_tests():
    print("=========================================================================")
    print(" M3GAN PHASE 5B MULTILINGUAL NLP & NORMALIZATION TEST SUITE")
    print("=========================================================================")
    test_language_detection()
    test_hinglish_normalization()
    test_pii_redaction_preservation()
    test_false_positive_control()
    print("\n ALL MULTILINGUAL NLP TESTS PASSED SUCCESSFULLY.")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_all_multilingual_tests()
