"""
backend/tests/test_svi_foundation.py

Unit & Regression Test Suite for M3GAN Phase 1 SVI Foundation.
Run: .venv/Scripts/python.exe -X utf8 backend/tests/test_svi_foundation.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.audio.acoustic_normalizer import AcousticNormalizer, SignalQualityState
from backend.audio.vad_processor import SileroVADProcessor
from backend.core.svi_engine import SVIEngine
from backend.api.schemas import TextAnalysisRequest, AudioAnalysisResponse, TextAnalysisResponse


def test_signal_quality_classification():
    print("--- 1. Testing Signal Quality Classification ---")

    # Fixture A: Pure Silence (3s)
    q_silence = AcousticNormalizer.evaluate_signal_quality(
        duration=3.0, speech_duration=0.0, voiced_ratio=0.0, voiced_frames=0, rms_energy=0.00005
    )
    print(f"    Silence Fixture: Quality={q_silence.quality}, PitchReliable={q_silence.pitch_reliable}, Reason='{q_silence.reason}'")
    assert q_silence.quality == "UNRELIABLE"
    assert q_silence.pitch_reliable is False
    assert q_silence.snr_state == "UNAVAILABLE"
    assert q_silence.snr_db is None

    # Fixture B: Low speech duration (< 0.3s speech)
    q_low = AcousticNormalizer.evaluate_signal_quality(
        duration=2.5, speech_duration=0.2, voiced_ratio=0.12, voiced_frames=2, rms_energy=0.02
    )
    print(f"    Low Speech Fixture: Quality={q_low.quality}, PitchReliable={q_low.pitch_reliable}")
    assert q_low.quality == "UNRELIABLE"
    assert q_low.pitch_reliable is False

    # Fixture C: Good audio
    q_good = AcousticNormalizer.evaluate_signal_quality(
        duration=4.0, speech_duration=3.2, voiced_ratio=0.80, voiced_frames=25, rms_energy=0.03
    )
    print(f"    Good Audio Fixture: Quality={q_good.quality}, PitchReliable={q_good.pitch_reliable}")
    assert q_good.quality == "GOOD"
    assert q_good.pitch_reliable is True


def test_pitch_reliability_independence():
    print("--- 2. Testing Pitch Reliability Independence ---")

    # When pitch is unreliable (<5 voiced frames), normalized pitch must return None
    norm_pitch = AcousticNormalizer.normalize_pitch_volatility(
        pitch_semitone_std=1.85, pitch_reliable=False
    )
    assert norm_pitch is None, "Unreliable pitch must normalize to None, not a fake float"

    # When pitch is reliable, standard semitone normalization applies
    norm_pitch_valid = AcousticNormalizer.normalize_pitch_volatility(
        pitch_semitone_std=1.75, pitch_reliable=True
    )
    assert norm_pitch_valid == round(1.75 / 3.5, 4), f"Unexpected normalized pitch: {norm_pitch_valid}"
    print(f"    Unreliable pitch normalized -> {norm_pitch} (None)")
    print(f"    Reliable pitch 1.75 st normalized -> {norm_pitch_valid}")


def test_safety_flag_independence_from_svi_score():
    print("--- 3. Testing Safety Flag Independence from SVI Score ---")
    engine = SVIEngine(use_vad_fallback=True)

    # Severe distress disclosure text (triggers Suicidal Ideation & Sexual Violence safety flags)
    req = TextAnalysisRequest(
        text="They raped me and now they threaten to kill me, I want to die",
        channel="chat",
        language="English",
    )

    res = engine.process_text(req)

    print(f"    Final SVI Score : {res.svi_metrics.final_svi_score}")
    print(f"    Risk Band       : {res.svi_metrics.risk_band}")
    print(f"    Safety Flags    : {[f.category for f in res.svi_metrics.safety_flags]}")

    assert len(res.svi_metrics.safety_flags) > 0, "High-severity disclosures must generate safety flags"

    # Mathematically verify that safety flags did NOT force svi_score to severity_floor (78.0)
    # The pure lexical score for this text should be calibrated from component weights
    lexical_score = res.svi_metrics.components[0].score
    expected_svi = round(engine._component_curve(lexical_score, 1.0) * 100.0, 2)

    assert res.svi_metrics.final_svi_score == expected_svi, (
        f"Safety flag altered SVI score! Got {res.svi_metrics.final_svi_score}, expected pure fusion {expected_svi}"
    )
    print(f"    [VERIFIED] Numerical SVI score ({res.svi_metrics.final_svi_score}) is pure fusion and NOT overwritten by safety flags!")


def test_confidence_metrics_and_serialization():
    print("--- 4. Testing Confidence Metrics & API Serialization ---")
    engine = SVIEngine(use_vad_fallback=True)

    req = TextAnalysisRequest(
        text="Mujhe roz darr lag raha hai, koi madad nahi kar raha",
        channel="chat",
        language="Hindi",
    )

    res = engine.process_text(req)

    print(f"    Overall Confidence: {res.confidence_metrics.overall_confidence}% ({res.confidence_metrics.confidence_rating})")
    print(f"    Signal Quality    : {res.confidence_metrics.signal_quality}")

    assert res.confidence_metrics is not None
    assert 0.0 <= res.confidence_metrics.overall_confidence <= 100.0
    assert res.confidence_metrics.confidence_rating in ["HIGH", "MEDIUM", "LOW"]

    # Verify Pydantic JSON serialization backwards compatibility
    json_dict = res.model_dump()
    assert "status" in json_dict
    assert "case_id" in json_dict
    assert "svi_metrics" in json_dict
    assert "safety_flags" in json_dict
    print("    [VERIFIED] Response model serializes to valid backwards-compatible API JSON payload!")


def main():
    print("[TEST SUITE] Starting SVI Foundation & Reliability Tests...")
    test_signal_quality_classification()
    test_pitch_reliability_independence()
    test_safety_flag_independence_from_svi_score()
    test_confidence_metrics_and_serialization()
    print("\n[TEST SUITE] All SVI foundation tests PASSED successfully!\n")


if __name__ == "__main__":
    main()
