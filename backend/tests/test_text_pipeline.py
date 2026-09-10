"""
Quick regression checks for the SVI text pipeline.
Run: .venv/Scripts/python.exe -X utf8 backend/tests/test_text_pipeline.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.lexicon import _COMPILED, scan_text, redact_pii
from backend.text_analyzer import analyze_transcript


def show(label, text):
    result = analyze_transcript(text)
    print(f"--- {label}")
    print(f"    text   : {text}")
    print(f"    hits   : {result['flagged_keywords']}")
    print(f"    score  : {result['lexical_score']}  floor: {result['severity_floor']}")
    print(f"    cats   : {result['categories']}")
    return result


def main():
    # 1. Debug: does the Devanagari phrase pattern compile/match?
    text = "मैं मरना चाहती हूँ"
    for cat, pat in _COMPILED:
        if "marana" in pat.pattern or "\u092e\u0930\u0928\u093e" in pat.pattern:
            print(f"DBG pattern: {pat.pattern!r} -> match: {pat.findall(text)}")

    show("English severe", "They raped me and now they threaten to kill me, I want to die")
    show("Romanized Hindi fear", "Woh log mujhe dhamki de rahe hain, main bahut darr gayi hoon, koi madad nahi kar raha")
    show("Devanagari suicidal", "मुझसे अब सहन नहीं होता, मैं मरना चाहती हूँ")
    show("Bengali fear", "আমি ভয় পাচ্ছি, কোনো সাহায্য নেই")
    show("Tamil agglutinative", "எனக்கு பயமாக இருக்கிறது, என்னை யாரும் உதவி செய்யவில்லை")
    show("Tamil sexual violence", "என்னை பாலியல் வன்முறைக்கு உட்படுத்தினார்கள்")
    show("Benign", "I need help filing a complaint about the dispute")

    red, n = redact_pii("My name is Ramesh and my number is 9876543210, mail me at ramesh@gmail.com")
    print(f"--- PII redaction ({n} redactions): {red}")


if __name__ == "__main__":
    main()
