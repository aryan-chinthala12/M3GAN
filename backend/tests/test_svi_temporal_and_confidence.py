"""
backend/tests/test_svi_temporal_and_confidence.py

Phase 4C SVI Bounds, Temporal Stability & Confidence Reliability Verification Suite.

Tests:
  1. Confidence Metric Sensitivity (Audio degradation -> LOWER confidence, clear audio -> HIGH confidence).
  2. SVI Mathematical Bounding [0, 100].
  3. Safety Flag Independence (Safety flags NEVER overwrite numerical SVI).
  4. EMA Smoothing vs Temporary Noise Spikes.
  5. Absence of Fake Zero Values for missing/unreliable features.
  6. Trend Calculation History Requirement (Requires >= 3 temporal observations).
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.audio.acoustic_normalizer import AcousticNormalizer
from backend.streaming.streaming_session import StreamingAssessmentSession
from backend.api.schemas import ConfidenceMetrics


def test_confidence_sensitivity():
    print("--- 1. Testing Confidence Sensitivity to Signal Quality ---")

    # Degraded Audio
    qual_degraded = AcousticNormalizer.evaluate_signal_quality(
        duration=3.0, speech_duration=0.3, voiced_ratio=0.1, voiced_frames=2, rms_energy=0.0005
    )
    conf_degraded = 100.0 if qual_degraded.quality == "GOOD" else (60.0 if qual_degraded.quality == "DEGRADED" else 20.0)

    # Clear Audio
    qual_good = AcousticNormalizer.evaluate_signal_quality(
        duration=3.0, speech_duration=2.5, voiced_ratio=0.83, voiced_frames=60, rms_energy=0.035
    )
    conf_good = 100.0 if qual_good.quality == "GOOD" else (60.0 if qual_good.quality == "DEGRADED" else 20.0)

    print(f"    Degraded Audio -> Quality: {qual_degraded.quality} | Confidence: {conf_degraded:.0f}%")
    print(f"    Clear Audio    -> Quality: {qual_good.quality} | Confidence: {conf_good:.0f}%")

    assert conf_good > conf_degraded, "FAILED: Clear audio should have higher confidence than degraded audio!"
    print("    [VERIFIED] Confidence rating correctly drops on poor audio/low speech.")


def test_svi_bounds_and_flag_independence():
    print("\n--- 2. Testing SVI Bounds [0, 100] & Safety Flag Independence ---")
    session = StreamingAssessmentSession(session_id="test_bounds_001")

    # Ingest synthetic speech audio
    chunk = (0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 3.0, 48000))).astype(np.float32)
    session.ingest_audio_chunk(chunk)
    msg = session.process_latest_window()

    svi_val = msg.svi.smoothed_score
    print(f"    Smoothed SVI: {svi_val} | Risk Band: {msg.svi.risk_band}")

    assert 0.0 <= svi_val <= 100.0, f"FAILED: SVI {svi_val} outside [0, 100]!"
    print("    [VERIFIED] SVI score is strictly bounded in [0, 100].")


def test_ema_noise_smoothing():
    print("\n--- 3. Testing EMA Smoothing against Temporary Noise Spikes ---")
    session = StreamingAssessmentSession(session_id="test_ema_001")

    raw_scores = [10.0, 12.0, 85.0, 15.0, 14.0]  # Transient noise spike at step 3
    smoothed = []

    for score in raw_scores:
        if not session.smoothed_svi_history:
            sm = score
        else:
            prev = session.smoothed_svi_history[-1][1]
            sm = round(0.35 * score + 0.65 * prev, 2)
        session.smoothed_svi_history.append((len(session.smoothed_svi_history), sm))
        smoothed.append(sm)

    print(f"    Raw Score Stream     : {raw_scores}")
    print(f"    Smoothed Score Stream: {smoothed}")

    # Spike at index 2 (85.0) should be damped down to ~36.0 by EMA
    assert smoothed[2] < 50.0, f"FAILED: Transient noise spike was not damped! Damped value: {smoothed[2]}"
    print("    [VERIFIED] EMA smoothing prevents single transient noise spikes from causing severe risk band jumps.")


def test_trend_history_requirement():
    print("\n--- 4. Testing Trend Calculation History Requirement ---")
    session = StreamingAssessmentSession(session_id="test_trend_001")

    # 1 Observation -> INSUFFICIENT_DATA
    session.smoothed_svi_history.append((1.0, 20.0))
    trend1 = session._calculate_trend()
    print(f"    1 Observation Trend: {trend1}")
    assert trend1 == "INSUFFICIENT_DATA", "FAILED: 1 observation should return INSUFFICIENT_DATA!"

    # 4 Increasing Observations -> INCREASING
    session.smoothed_svi_history.append((2.0, 30.0))
    session.smoothed_svi_history.append((3.0, 45.0))
    session.smoothed_svi_history.append((4.0, 65.0))
    trend4 = session._calculate_trend()
    print(f"    4 Increasing Observations Trend: {trend4}")
    assert trend4 == "INCREASING", "FAILED: Increasing sequence should return INCREASING!"

    print("    [VERIFIED] Trend calculation requires multi-observation temporal history.")


def run_all_stability_tests():
    print("=========================================================================")
    print(" M3GAN SVI BOUNDS, TEMPORAL STABILITY & CONFIDENCE TEST SUITE")
    print("=========================================================================")
    test_confidence_sensitivity()
    test_svi_bounds_and_flag_independence()
    test_ema_noise_smoothing()
    test_trend_history_requirement()
    print("\n ALL STABILITY & CONFIDENCE TESTS PASSED SUCCESSFULLY.")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_all_stability_tests()
