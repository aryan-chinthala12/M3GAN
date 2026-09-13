"""
backend/acoustic_normalizer.py

Signal Quality Classifier & Acoustic Feature Normalizer.

Responsibilities:
  - Classifies audio input into explicit Signal Quality states:
    'GOOD', 'DEGRADED', 'UNRELIABLE'.
  - Assesses pitch reliability independently from pitch volatility.
  - Handles missing/insufficient data without fabricating false zeros or fake SNRs.
  - Normalizes biomarkers (semitone pitch volatility, relative RMS energy, pause ratio)
    into bounded 0..1 distress components.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class SignalQualityState:
    quality: str                  # "GOOD", "DEGRADED", "LOW_VOLUME", "WHISPER", "LOW_SNR", "UNRELIABLE"
    pitch_reliable: bool          # True if >= 5 valid voiced frames and non-silent
    snr_db: Optional[float]       # Signal-to-Noise Ratio in dB or None if unavailable
    snr_state: str                # "ESTIMATED", "UNAVAILABLE"
    has_sufficient_speech: bool   # True if speech >= 0.4s
    reason: str                   # Explanation of quality decision
    whisper_detected: bool = False
    spectral_energy_ratio: float = 1.0


class AcousticNormalizer:
    """Evaluates signal quality, checks biomarker reliability, and normalizes acoustic stress features."""

    @staticmethod
    def estimate_snr(
        y: np.ndarray,
        speech_segments: list,
        sr: int = 16000,
    ) -> Tuple[Optional[float], str]:
        """
        Estimate Signal-to-Noise Ratio (SNR dB) from VAD speech and non-speech background segments.
        Requires >= 0.3s of background silence to produce a reliable estimate.
        """
        if not speech_segments or len(y) < sr * 0.5:
            return None, "UNAVAILABLE"

        total_samples = len(y)
        speech_mask = np.zeros(total_samples, dtype=bool)

        for start, end in speech_segments:
            i0 = max(0, int(start * sr))
            i1 = min(total_samples, int(end * sr))
            speech_mask[i0:i1] = True

        non_speech_samples = y[~speech_mask]

        if len(non_speech_samples) < int(sr * 0.3):
            return None, "UNAVAILABLE"

        speech_samples = y[speech_mask]
        if len(speech_samples) < int(sr * 0.2):
            return None, "UNAVAILABLE"

        rms_speech = float(np.sqrt(np.mean(np.square(speech_samples))))
        rms_noise = float(np.sqrt(np.mean(np.square(non_speech_samples))))

        if rms_noise < 1e-6 or rms_speech < 1e-6:
            return None, "UNAVAILABLE"

        snr_db = round(float(20.0 * np.log2(rms_speech / rms_noise) * (np.log(2) / np.log(10))), 2)
        # Standard SNR calculation: 20 * log10(rms_speech / rms_noise)
        snr_db = round(float(20.0 * np.log10(rms_speech / rms_noise)), 2)

        return snr_db, "ESTIMATED"

    @staticmethod
    def evaluate_signal_quality(
        duration: float,
        speech_duration: float,
        voiced_ratio: float,
        voiced_frames: int,
        rms_energy: float,
        spectral_energy_ratio: float = 1.0,
        snr_db: Optional[float] = None,
        snr_state: str = "UNAVAILABLE",
    ) -> SignalQualityState:
        """
        Evaluate overall audio signal quality into explicit non-clinical states:
        - UNRELIABLE: audio < 0.8s, or speech < 0.3s, or silent (rms < 0.0005).
        - WHISPER: speech >= 0.4s, low voiced_frames (< 5), low RMS (<= 0.015), or elevated spectral ratio (>= 0.85).
        - LOW_VOLUME: speech >= 0.4s, rms <= 0.008, valid speech activity.
        - LOW_SNR: speech >= 0.5s, snr_db < 8.0 dB (when SNR is estimated).
        - DEGRADED: speech between 0.4s and 1.2s or voiced_frames between 3 and 7.
        - GOOD: speech >= 1.2s, duration >= 2.0s, voiced_frames >= 8, nominal volume.
        """
        # 1. Pure silence / missing speech guardrail
        if duration < 0.8 or speech_duration < 0.3 or rms_energy < 0.0005:
            reasons = []
            if duration < 0.8:
                reasons.append(f"Duration too short ({duration:.2f}s < 0.8s)")
            if speech_duration < 0.3:
                reasons.append(f"Insufficient voiced speech ({speech_duration:.2f}s < 0.3s)")
            if rms_energy < 0.0005:
                reasons.append("Silent or near-zero energy input")

            return SignalQualityState(
                quality="UNRELIABLE",
                pitch_reliable=False,
                snr_db=snr_db,
                snr_state=snr_state,
                has_sufficient_speech=False,
                reason="; ".join(reasons) or "Unreliable audio signal",
                whisper_detected=False,
                spectral_energy_ratio=spectral_energy_ratio,
            )

        pitch_reliable = bool(voiced_frames >= 5)

        # 2. Multi-feature Whisper Detection (VAD active + low f0 count + low RMS / high spectral ratio)
        is_whisper = bool(
            speech_duration >= 0.4
            and (
                (voiced_frames < 5 and rms_energy <= 0.015 and spectral_energy_ratio >= 0.70)
                or (rms_energy <= 0.006 and spectral_energy_ratio >= 0.85)
            )
        )

        if is_whisper:
            return SignalQualityState(
                quality="WHISPER",
                pitch_reliable=False,
                snr_db=snr_db,
                snr_state=snr_state,
                has_sufficient_speech=True,
                reason="Soft / whispered speech detected; features extracted with unvoiced acoustic fallback.",
                whisper_detected=True,
                spectral_energy_ratio=spectral_energy_ratio,
            )

        # 3. Low Volume Speech Detection
        if rms_energy <= 0.008:
            return SignalQualityState(
                quality="LOW_VOLUME",
                pitch_reliable=pitch_reliable,
                snr_db=snr_db,
                snr_state=snr_state,
                has_sufficient_speech=True,
                reason="Quiet / low-amplitude speech detected; signal quality marked LOW_VOLUME.",
                whisper_detected=False,
                spectral_energy_ratio=spectral_energy_ratio,
            )

        # 4. Low SNR Detection
        if snr_state == "ESTIMATED" and snr_db is not None and snr_db < 8.0:
            return SignalQualityState(
                quality="LOW_SNR",
                pitch_reliable=pitch_reliable,
                snr_db=snr_db,
                snr_state=snr_state,
                has_sufficient_speech=True,
                reason=f"High background noise detected (SNR={snr_db:.1f} dB < 8.0 dB threshold).",
                whisper_detected=False,
                spectral_energy_ratio=spectral_energy_ratio,
            )

        # 5. Degraded Quality
        if duration < 2.0 or speech_duration < 1.2 or voiced_frames < 8:
            return SignalQualityState(
                quality="DEGRADED",
                pitch_reliable=pitch_reliable,
                snr_db=snr_db,
                snr_state=snr_state,
                has_sufficient_speech=bool(speech_duration >= 0.8),
                reason="Short or low-speech sample; features extracted with reduced confidence.",
                whisper_detected=False,
                spectral_energy_ratio=spectral_energy_ratio,
            )

        # 6. Good Quality Baseline
        return SignalQualityState(
            quality="GOOD",
            pitch_reliable=pitch_reliable,
            snr_db=snr_db,
            snr_state=snr_state,
            has_sufficient_speech=True,
            reason="Signal quality sufficient for reliable acoustic feature extraction.",
            whisper_detected=False,
            spectral_energy_ratio=spectral_energy_ratio,
        )

    @staticmethod
    def normalize_pitch_volatility(
        pitch_semitone_std: float,
        pitch_reliable: bool,
    ) -> Optional[float]:
        """
        Convert pitch semitone std into a normalized 0..1 acoustic distress component.
        If pitch is unreliable (e.g. unvoiced/silent/insufficient frames), returns None.
        Standard calibration mapping: 0.0 st -> 0.0, >= 3.5 st -> 1.0.
        """
        if not pitch_reliable:
            return None

        val = float(pitch_semitone_std)
        normalized = float(np.clip(val / 3.5, 0.0, 1.0))
        return round(normalized, 4)

    @staticmethod
    def normalize_rms_energy(rms_energy: float) -> float:
        """
        Convert mean RMS energy to 0..1 component.
        0.10 RMS represents nominal full-scale voice peak.
        """
        val = float(rms_energy)
        return round(float(np.clip(val / 0.10, 0.0, 1.0)), 4)

    @staticmethod
    def normalize_energy_variation(energy_var_db: float) -> float:
        """
        Convert energy variation in dB to 0..1 component.
        12.0 dB standard deviation represents maximum energy dynamic range.
        """
        val = float(energy_var_db)
        return round(float(np.clip(val / 12.0, 0.0, 1.0)), 4)

    @staticmethod
    def normalize_pause_ratio(pause_ratio: float) -> float:
        """
        Convert pause ratio (0..1) to acoustic distress component.
        High pause ratios (> 0.40) indicate speech disruption / hesitations.
        """
        val = float(pause_ratio)
        return round(float(np.clip(val / 0.45, 0.0, 1.0)), 4)

    @staticmethod
    def normalize_spectral_ratio(spectral_ratio: float) -> float:
        """
        Convert high-frequency spectral energy ratio (1-4 kHz vs < 1 kHz) to 0..1 component.
        """
        val = float(spectral_ratio)
        return round(float(np.clip(val / 2.0, 0.0, 1.0)), 4)

    @classmethod
    def compute_acoustic_distress_score(
        cls,
        pitch_semitone_std: float,
        rms_energy: float,
        energy_var_db: float,
        pause_ratio: float,
        quality_state: SignalQualityState,
    ) -> Dict[str, Any]:
        """
        Compute robust 0..1 acoustic distress score handling missing/unreliable pitch gracefully.
        Supports WHISPER and LOW_VOLUME fallback models so quiet speech is NOT zeroed.
        """
        if quality_state.quality == "UNRELIABLE":
            return {
                "acoustic_score": 0.0,
                "confidence": 0.0,
                "pitch_component": None,
                "energy_component": cls.normalize_rms_energy(rms_energy),
                "energy_var_component": cls.normalize_energy_variation(energy_var_db),
                "pause_component": cls.normalize_pause_ratio(pause_ratio),
                "detail": f"Unreliable audio quality ({quality_state.reason}); acoustic score zeroed.",
            }

        energy_comp = cls.normalize_rms_energy(rms_energy)
        energy_var_comp = cls.normalize_energy_variation(energy_var_db)
        pause_comp = cls.normalize_pause_ratio(pause_ratio)
        spectral_comp = cls.normalize_spectral_ratio(quality_state.spectral_energy_ratio)
        pitch_comp = cls.normalize_pitch_volatility(
            pitch_semitone_std, quality_state.pitch_reliable
        )

        if pitch_comp is not None:
            # Full pitch-aware acoustic model
            score = (
                0.40 * pitch_comp
                + 0.25 * energy_comp
                + 0.15 * energy_var_comp
                + 0.20 * pause_comp
            )
            confidence = 100.0 if quality_state.quality == "GOOD" else (
                60.0 if quality_state.quality == "LOW_SNR" else 70.0
            )
            detail_msg = (
                f"Pitch volatility={pitch_semitone_std:.2f} st (comp={pitch_comp:.2f}), "
                f"RMS={rms_energy:.4f}, EnergyVar={energy_var_db:.1f} dB, "
                f"PauseRatio={pause_ratio:.2f}"
            )
        elif quality_state.quality in ("WHISPER", "LOW_VOLUME"):
            # Quiet / whispered speech fallback model (non-pitch features)
            score = 0.35 * energy_var_comp + 0.35 * pause_comp + 0.30 * spectral_comp
            confidence = 50.0
            detail_msg = (
                f"Unvoiced quiet/whisper speech ({quality_state.reason}); "
                f"applied unvoiced fallback (EnergyVar={energy_var_db:.1f} dB, PauseRatio={pause_ratio:.2f}, "
                f"SpectralRatio={quality_state.spectral_energy_ratio:.2f})."
            )
        else:
            # Standard unvoiced fallback model
            score = 0.40 * energy_comp + 0.25 * energy_var_comp + 0.35 * pause_comp
            confidence = 50.0 if quality_state.quality == "GOOD" else 30.0
            detail_msg = (
                f"Pitch feature unavailable ({quality_state.reason}); "
                f"substituted energy & pause components (RMS={rms_energy:.4f}, PauseRatio={pause_ratio:.2f})"
            )

        return {
            "acoustic_score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "confidence": confidence,
            "pitch_component": pitch_comp,
            "energy_component": energy_comp,
            "energy_var_component": energy_var_comp,
            "pause_component": pause_comp,
            "detail": detail_msg,
        }

