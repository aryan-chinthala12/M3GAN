"""
backend/tests/test_vad_processor.py

Silero VAD Latency Benchmark & Unit Test Suite.
Run: .venv/Scripts/python.exe -X utf8 backend/tests/test_vad_processor.py
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.audio.vad_processor import SileroVADProcessor, VADResult


def generate_synthetic_audio(duration_sec=3.0, sample_rate=16000, speech_ratio=0.5):
    """Generate synthetic audio with alternating tone (speech) and pure silence."""
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Add 220Hz sine wave tone for speech portions
    speech_samples = int(total_samples * speech_ratio)
    t = np.linspace(0, speech_ratio * duration_sec, speech_samples, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # Insert speech tone in middle
    start = (total_samples - speech_samples) // 2
    audio[start : start + speech_samples] = tone

    return audio


def test_vad_latency():
    print("--- 1. Testing Silero VAD Latency Benchmark ---")
    processor = SileroVADProcessor(force_fallback=False)

    # Test durations: 1s, 3s, 5s, 10s
    durations = [1.0, 3.0, 5.0, 10.0]
    latencies = []

    for dur in durations:
        audio = generate_synthetic_audio(duration_sec=dur, speech_ratio=0.6)
        res = processor.process_audio(audio)
        latencies.append(res.execution_time_ms)
        print(
            f"    Audio Duration: {dur}s | Engine: {res.vad_engine} | "
            f"Latency: {res.execution_time_ms:.2f} ms | Voiced Ratio: {res.voiced_ratio:.2f} | "
            f"Sufficient Speech: {res.has_sufficient_speech}"
        )

    avg_latency = sum(latencies) / len(latencies)
    print(f"[RESULT] Average VAD Execution Latency: {avg_latency:.2f} ms")
    assert avg_latency < 500.0, "VAD latency exceeds 500ms threshold!"


def test_silence_detection():
    print("--- 2. Testing Pure Silence Detection ---")
    processor = SileroVADProcessor()
    silent_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s silence

    res = processor.process_audio(silent_audio)
    print(
        f"    Silence Result: speech_duration={res.speech_duration:.2f}s, "
        f"voiced_ratio={res.voiced_ratio:.2f}, pause_count={res.pause_count}, "
        f"has_sufficient_speech={res.has_sufficient_speech}"
    )

    assert res.speech_duration == 0.0, "Silence should yield 0.0s speech duration"
    assert res.has_sufficient_speech is False, "Silence must not be marked as sufficient speech"


def main():
    print("[TEST SUITE] Starting Silero VAD Processor Tests...")
    test_vad_latency()
    test_silence_detection()
    print("[TEST SUITE] All VAD processor tests PASSED successfully.\n")


if __name__ == "__main__":
    main()
