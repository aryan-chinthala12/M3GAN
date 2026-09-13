"""
backend/audio_processor.py

Audio signal processing & acoustic biomarker extraction for the M3GAN SVI engine.

Integrates:
  - Silero VAD (voice activity detection, speech/silence segmentation, pause metrics).
  - Acoustic Normalizer (Signal Quality classification, pitch reliability check).
  - Robust F0 semitone volatility & RMS energy variation.
"""

import librosa
import numpy as np
from types import SimpleNamespace
from typing import Dict, Any, Tuple

from backend.audio.vad_processor import SileroVADProcessor
from backend.audio.acoustic_normalizer import AcousticNormalizer, SignalQualityState


class AudioProcessor:
    def __init__(self, use_vad_fallback: bool = False):
        self.vad_processor = SileroVADProcessor(force_fallback=use_vad_fallback)

    @staticmethod
    def validate_and_load(file_path: str, target_sr: int = 16000) -> Tuple[Any, float, Any]:
        """Load audio and validate duration and signal energy guardrails."""
        try:
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)

            duration = float(librosa.get_duration(y=y, sr=sr))

            if duration < 0.8:
                return None, duration, f"Audio too short ({duration:.2f}s < 0.8s requirement)"

            rms = float(np.sqrt(np.mean(np.square(y))))

            if rms < 0.001:
                return None, duration, "Silent or near-zero energy audio input"

            return y, duration, None

        except Exception as e:
            return None, 0.0, f"Error loading audio file: {str(e)}"

    def extract_voice_biomarkers(self, y: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Extract robust acoustic biomarkers backed by Silero VAD and Acoustic Normalizer.
        """
        total_duration = float(len(y) / sr)

        # ---------------------------------------------------------
        # 1. Voice Activity Detection (VAD) & Pause Analysis
        # ---------------------------------------------------------
        vad_result = self.vad_processor.process_audio(y, sample_rate=sr)

        # ---------------------------------------------------------
        # 2. Fundamental Frequency (F0) on Voiced Segments Only
        # ---------------------------------------------------------
        try:
            # Mask audio to voiced speech if VAD detected valid segments
            if vad_result.speech_segments:
                voiced_samples = []
                for start, end in vad_result.speech_segments:
                    i0 = int(start * sr)
                    i1 = min(int(end * sr), len(y))
                    if i1 > i0:
                        voiced_samples.append(y[i0:i1])
                target_audio = np.concatenate(voiced_samples) if voiced_samples else y
            else:
                target_audio = y

            f0, voiced_flag, voiced_prob = librosa.pyin(
                target_audio,
                fmin=70,
                fmax=400.0,
                sr=sr,
                frame_length=2048,
                hop_length=256,
            )

            # Filter valid voiced frames
            valid_f0 = f0[~np.isnan(f0) & (voiced_prob >= 0.4)]

        except Exception:
            valid_f0 = np.array([])

        voiced_frames_count = int(len(valid_f0))

        # Calculate overall RMS energy over full signal
        rms_series = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]
        rms_energy = float(np.mean(rms_series))

        if len(rms_series) >= 5 and rms_energy > 0.0001:
            rms_db = librosa.amplitude_to_db(rms_series, ref=np.max)
            energy_variation = float(np.std(rms_db))
        else:
            energy_variation = 0.0

        # Long pause count (pauses exceeding 1.5s between speech segments)
        long_pause_count = 0
        last_seg_end = 0.0
        for seg_start, seg_end in vad_result.speech_segments:
            if (seg_start - last_seg_end) >= 1.5:
                long_pause_count += 1
            last_seg_end = seg_end
        if (total_duration - last_seg_end) >= 1.5:
            long_pause_count += 1

        # Spectral energy ratio (High Freq 1000-4000Hz vs Low Freq 100-1000Hz)
        try:
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            low_mask = (freqs >= 100) & (freqs <= 1000)
            high_mask = (freqs > 1000) & (freqs <= 4000)

            low_energy = float(np.sum(S[low_mask, :]))
            high_energy = float(np.sum(S[high_mask, :]))
            spectral_energy_ratio = float(high_energy / max(low_energy, 1e-6))
        except Exception:
            spectral_energy_ratio = 1.0

        # Estimate SNR from VAD speech and non-speech background segments
        snr_db, snr_state = AcousticNormalizer.estimate_snr(y, vad_result.speech_segments, sr=sr)

        # ---------------------------------------------------------
        # 3. Signal Quality & Pitch Reliability Assessment
        # ---------------------------------------------------------
        quality_state = AcousticNormalizer.evaluate_signal_quality(
            duration=total_duration,
            speech_duration=vad_result.speech_duration,
            voiced_ratio=vad_result.voiced_ratio,
            voiced_frames=voiced_frames_count,
            rms_energy=rms_energy,
            spectral_energy_ratio=spectral_energy_ratio,
            snr_db=snr_db,
            snr_state=snr_state,
        )

        # ---------------------------------------------------------
        # 4. Semitone Pitch Volatility & Rich Biomarkers
        # ---------------------------------------------------------
        if quality_state.pitch_reliable and voiced_frames_count >= 5:
            median_f0 = float(np.median(valid_f0))
            pitch_semitones = 12.0 * np.log2(valid_f0 / median_f0)
            pitch_volatility = float(np.std(pitch_semitones))
            pitch_semitone_range = float(np.max(pitch_semitones) - np.min(pitch_semitones))
            median_pitch = median_f0

            # F0 Jitter Proxy: relative frame-to-frame F0 variation
            if len(valid_f0) >= 2:
                f0_diffs = np.abs(np.diff(valid_f0))
                f0_jitter_proxy = float(np.mean(f0_diffs) / max(median_f0, 1.0))
            else:
                f0_jitter_proxy = 0.0
        else:
            pitch_volatility = 0.0  # Kept as float for legacy backward-compat
            pitch_semitone_range = 0.0
            f0_jitter_proxy = 0.0
            median_pitch = 0.0

        # Compute normalized acoustic distress breakdown
        acoustic_distress = AcousticNormalizer.compute_acoustic_distress_score(
            pitch_semitone_std=pitch_volatility,
            rms_energy=rms_energy,
            energy_var_db=energy_variation,
            pause_ratio=vad_result.pause_ratio,
            quality_state=quality_state,
        )

        return {
            "pitch_volatility": round(pitch_volatility, 4),
            "pitch_semitone_range": round(pitch_semitone_range, 4),
            "f0_jitter_proxy": round(f0_jitter_proxy, 4),
            "median_pitch": round(median_pitch, 2),
            "rms_energy": round(rms_energy, 6),
            "energy_variation": round(energy_variation, 4),
            "spectral_energy_ratio": round(spectral_energy_ratio, 4),
            "voiced_frames": voiced_frames_count,
            # Extended Phase 1 & Phase 4B metrics
            "speech_duration": vad_result.speech_duration,
            "voiced_ratio": vad_result.voiced_ratio,
            "pause_count": vad_result.pause_count,
            "long_pause_count": long_pause_count,
            "total_pause_duration": vad_result.total_pause_duration,
            "mean_pause_duration": vad_result.mean_pause_duration,
            "pause_ratio": vad_result.pause_ratio,
            "signal_quality": quality_state.quality,
            "pitch_reliable": quality_state.pitch_reliable,
            "snr_state": quality_state.snr_state,
            "snr_db": quality_state.snr_db,
            "quality_reason": quality_state.reason,
            "acoustic_score": acoustic_distress["acoustic_score"],
            "acoustic_detail": acoustic_distress["detail"],
            "vad_engine": vad_result.vad_engine,
            "vad_latency_ms": vad_result.execution_time_ms,
        }

    def extract_acoustic_features(self, file_path: str) -> SimpleNamespace:
        """Bridge method consumed by svi_engine.py."""
        y, duration, error = self.validate_and_load(file_path)

        if error or y is None:
            return SimpleNamespace(
                pitch_std=0.0,
                energy_rms=0.0,
                duration=duration,
                error=error,
                median_pitch=0.0,
                energy_variation=0.0,
                voiced_frames=0,
                voiced_ratio=0.0,
                pause_ratio=0.0,
                signal_quality="UNRELIABLE",
                pitch_reliable=False,
                snr_state="UNAVAILABLE",
                snr_db=None,
                acoustic_score=0.0,
                acoustic_detail=f"Audio error: {error}",
            )

        biomarkers = self.extract_voice_biomarkers(y)
        print("[DEBUG] Acoustic biomarkers:", biomarkers)

        return SimpleNamespace(
            pitch_std=biomarkers["pitch_volatility"],
            energy_rms=biomarkers["rms_energy"],
            duration=duration,
            error=None,
            median_pitch=biomarkers["median_pitch"],
            energy_variation=biomarkers["energy_variation"],
            voiced_frames=biomarkers["voiced_frames"],
            voiced_ratio=biomarkers["voiced_ratio"],
            pause_ratio=biomarkers["pause_ratio"],
            signal_quality=biomarkers["signal_quality"],
            pitch_reliable=biomarkers["pitch_reliable"],
            snr_state=biomarkers["snr_state"],
            snr_db=biomarkers["snr_db"],
            speech_duration=biomarkers["speech_duration"],
            acoustic_score=biomarkers["acoustic_score"],
            acoustic_detail=biomarkers["acoustic_detail"],
        )