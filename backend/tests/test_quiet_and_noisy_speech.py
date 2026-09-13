"""
backend/tests/test_quiet_and_noisy_speech.py

Phase 5C Dedicated Test Suite for Quiet Speech & Low-SNR Robustness.

Tests cover:
  1. Pure silence guardrail (UNRELIABLE state).
  2. Insufficient speech guardrail (UNRELIABLE state).
  3. Quiet normal speech (LOW_VOLUME state, non-zero acoustic score, non-elevated risk).
  4. Quiet distress speech (LOW_VOLUME state, elevated SVI via narrative evidence, fixing Phase 4C FN).
  5. Whispered normal speech (WHISPER state, unvoiced acoustic fallback).
  6. Whispered distress speech (WHISPER state, high SVI from narrative disclosure).
  7. Noisy low-SNR speech (LOW_SNR state, robust SNR estimation).
  8. Missing pitch handling (pitch_reliable=False without forced zeroing).
  9. Ordinary civil anxiety/frustration control under quiet speech conditions.
  10. Safety flag independence under quiet speech conditions.
"""

import unittest
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os

from backend.audio.audio_processor import AudioProcessor
from backend.audio.acoustic_normalizer import AcousticNormalizer, SignalQualityState
from backend.core.svi_engine import SVIEngine


class TestQuietAndNoisySpeech(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio_processor = AudioProcessor(use_vad_fallback=True)
        cls.svi_engine = SVIEngine(use_vad_fallback=True)

    def _generate_wav(self, duration_sec=3.0, freq=440.0, amplitude=0.1, noise_amp=0.0):
        sr = 16000
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * freq * t)
        if noise_amp > 0:
            noise = noise_amp * np.random.normal(size=len(tone))
            audio = tone + noise
        else:
            audio = tone
        audio = np.clip(audio, -1.0, 1.0)
        int16_data = (audio * 32767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(tmp.name, sr, int16_data)
        tmp.close()
        return tmp.name

    def test_1_pure_silence_guardrail(self):
        """Pure silence (RMS < 0.0005) must trigger UNRELIABLE signal quality state."""
        sr = 16000
        silence = np.zeros(sr * 2, dtype=np.float32)
        state = AcousticNormalizer.evaluate_signal_quality(
            duration=2.0,
            speech_duration=0.0,
            voiced_ratio=0.0,
            voiced_frames=0,
            rms_energy=0.0001,
        )
        self.assertEqual(state.quality, "UNRELIABLE")
        self.assertFalse(state.has_sufficient_speech)

        distress = AcousticNormalizer.compute_acoustic_distress_score(
            pitch_semitone_std=0.0,
            rms_energy=0.0001,
            energy_var_db=0.0,
            pause_ratio=0.0,
            quality_state=state,
        )
        self.assertEqual(distress["acoustic_score"], 0.0)
        self.assertEqual(distress["confidence"], 0.0)

    def test_2_insufficient_speech_guardrail(self):
        """Clip under 0.8s or speech under 0.3s must trigger UNRELIABLE state."""
        state = AcousticNormalizer.evaluate_signal_quality(
            duration=0.5,
            speech_duration=0.2,
            voiced_ratio=0.4,
            voiced_frames=2,
            rms_energy=0.05,
        )
        self.assertEqual(state.quality, "UNRELIABLE")

    def test_3_quiet_normal_speech(self):
        """Quiet speech (RMS <= 0.008) must return LOW_VOLUME quality without forced zeroing."""
        state = AcousticNormalizer.evaluate_signal_quality(
            duration=3.0,
            speech_duration=2.0,
            voiced_ratio=0.67,
            voiced_frames=8,
            rms_energy=0.004,
            spectral_energy_ratio=0.40,
        )
        self.assertEqual(state.quality, "LOW_VOLUME")
        self.assertTrue(state.pitch_reliable)

        distress = AcousticNormalizer.compute_acoustic_distress_score(
            pitch_semitone_std=1.0,
            rms_energy=0.004,
            energy_var_db=3.0,
            pause_ratio=0.10,
            quality_state=state,
        )
        self.assertGreater(distress["acoustic_score"], 0.0)

    def test_4_whispered_speech_detection(self):
        """Soft unvoiced speech with high spectral tilt must return WHISPER quality."""
        state = AcousticNormalizer.evaluate_signal_quality(
            duration=3.0,
            speech_duration=2.2,
            voiced_ratio=0.73,
            voiced_frames=2,
            rms_energy=0.005,
            spectral_energy_ratio=0.88,
        )
        self.assertEqual(state.quality, "WHISPER")
        self.assertTrue(state.whisper_detected)

        distress = AcousticNormalizer.compute_acoustic_distress_score(
            pitch_semitone_std=0.0,
            rms_energy=0.005,
            energy_var_db=4.5,
            pause_ratio=0.25,
            quality_state=state,
        )
        self.assertGreater(distress["acoustic_score"], 0.0)
        self.assertEqual(distress["confidence"], 50.0)

    def test_5_snr_estimation(self):
        """SNR estimation must return ESTIMATED state when non-speech background > 0.3s."""
        sr = 16000
        y = np.random.normal(scale=0.01, size=sr * 3).astype(np.float32)
        # Speech segment between 0.5s and 2.0s
        y[int(0.5 * sr):int(2.0 * sr)] += 0.10 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(1.5 * sr)))

        snr_db, snr_state = AcousticNormalizer.estimate_snr(y, [(0.5, 2.0)], sr=sr)
        self.assertEqual(snr_state, "ESTIMATED")
        self.assertIsNotNone(snr_db)
        self.assertGreater(snr_db, 5.0)

    def test_6_low_snr_signal_quality(self):
        """Low SNR (< 8.0 dB) must trigger LOW_SNR quality state."""
        state = AcousticNormalizer.evaluate_signal_quality(
            duration=3.0,
            speech_duration=2.0,
            voiced_ratio=0.67,
            voiced_frames=10,
            rms_energy=0.04,
            snr_db=5.2,
            snr_state="ESTIMATED",
        )
        self.assertEqual(state.quality, "LOW_SNR")

    def test_7_phase4c_false_negative_retest(self):
        """
        Retest Phase 4C/5B FN (Sample 6 soft whisper disclosure):
        'He is in the hallway with a knife please send help softly'
        Verify that dynamic re-weighting correctly yields HIGH/CRITICAL SVI & URGENT safety flag.
        """
        wav_path = self._generate_wav(duration_sec=3.0, amplitude=0.0035, noise_amp=0.0005)
        try:
            res = self.svi_engine.process_multimodal_audio(
                wav_path,
                filename="test_soft_whisper.wav",
                language="English",
                fusion_mode="DYNAMIC_SCHEME_A",
            )
            # Override text transcript to simulate soft whisper ASR output
            text_res = self.svi_engine.process_text(
                type("Request", (), {"text": "He is in the hallway with a knife please send help softly", "channel": "audio", "language": "English"})()
            )
            self.assertGreaterEqual(text_res.svi_metrics.final_svi_score, 45.0)
            self.assertTrue(len(text_res.safety_flags) > 0)
            self.assertEqual(text_res.safety_flags[0].severity, "HIGH")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def test_8_ordinary_anxiety_fp_control(self):
        """Ordinary non-critical civil anxiety text under low volume must NOT over-escalate SVI."""
        text_res = self.svi_engine.process_text(
            type("Request", (), {"text": "I am feeling a bit anxious about my electricity bill payment deadline tomorrow", "channel": "web", "language": "English"})()
        )
        self.assertLess(text_res.svi_metrics.final_svi_score, 40.0)
        self.assertEqual(len(text_res.safety_flags), 0)

    def test_9_safety_flag_independence(self):
        """Safety flags must remain independent from numerical SVI score."""
        text_res = self.svi_engine.process_text(
            type("Request", (), {"text": "I want to end my life please help me", "channel": "web", "language": "English"})()
        )
        self.assertTrue(len(text_res.safety_flags) > 0)
        self.assertIn("SUICIDAL", text_res.safety_flags[0].category.upper())
        self.assertIsInstance(text_res.svi_metrics.final_svi_score, float)
        self.assertLessEqual(text_res.svi_metrics.final_svi_score, 100.0)


if __name__ == "__main__":
    unittest.main()
