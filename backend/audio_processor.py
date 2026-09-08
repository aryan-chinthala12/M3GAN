import librosa
import numpy as np
from types import SimpleNamespace


class AudioProcessor:

    @staticmethod
    def validate_and_load(file_path: str, target_sr: int = 16000):
        """Load audio and validate duration and signal energy guardrails."""
        try:
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)

            duration = float(librosa.get_duration(y=y, sr=sr))

            if duration < 0.8:
                return None, 0.0, "Audio too short (< 0.8s)"

            rms = float(np.sqrt(np.mean(np.square(y))))

            if rms < 0.001:
                return None, duration, "Silent or empty audio input"

            return y, duration, None

        except Exception as e:
            return None, 0.0, f"Error processing audio file: {str(e)}"

    @staticmethod
    def extract_voice_biomarkers(y, sr: int = 16000) -> dict:
        """
        Extract robust acoustic biomarkers.

        Pitch volatility is calculated from voiced F0 estimates
        rather than raw piptrack candidates.
        """

        # ---------------------------------------------------------
        # 1. Fundamental frequency (F0)
        # ---------------------------------------------------------

        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=70,
                fmax=400.0,
                sr=sr,
                frame_length=2048,
                hop_length=256
            )

            # Keep only reliable voiced frames
            valid_f0 = f0[
                ~np.isnan(f0)
                & (voiced_prob >= 0.4)
            ]

        except Exception:
            valid_f0 = np.array([])

        # ---------------------------------------------------------
        # 2. Robust pitch variation
        # ---------------------------------------------------------

        if len(valid_f0) >= 5:

            # Median F0 is more robust than mean F0
            median_f0 = float(np.median(valid_f0))

            # Convert F0 variation to semitone variation.
            # This makes the measurement relative rather than
            # dependent on absolute voice pitch.
            pitch_semitones = 12.0 * np.log2(
                valid_f0 / median_f0
            )

            pitch_volatility = float(
                np.std(pitch_semitones)
            )

            median_pitch = median_f0

        else:
            pitch_volatility = 0.0
            median_pitch = 0.0

        # ---------------------------------------------------------
        # 3. RMS energy
        # ---------------------------------------------------------

        rms = librosa.feature.rms(
            y=y,
            frame_length=2048,
            hop_length=256
        )[0]

        rms_energy = float(np.mean(rms))

        # Relative energy variation is more useful than raw
        # microphone volume for later SVI normalization.
        if len(rms) >= 5 and np.mean(rms) > 0:

            rms_db = librosa.amplitude_to_db(
                rms,
                ref=np.max
            )

            energy_variation = float(np.std(rms_db))

        else:
            energy_variation = 0.0

        # ---------------------------------------------------------
        # 4. Return biomarkers
        # ---------------------------------------------------------

        return {
            "pitch_volatility": round(pitch_volatility, 4),
            "median_pitch": round(median_pitch, 2),
            "rms_energy": round(rms_energy, 6),
            "energy_variation": round(energy_variation, 4),
            "voiced_frames": int(len(valid_f0))
        }

    def extract_acoustic_features(self, file_path: str):
        """Bridge method required by svi_engine.py."""

        y, duration, error = self.validate_and_load(file_path)

        if error or y is None:
            return SimpleNamespace(
                pitch_std=0.0,
                energy_rms=0.0,
                duration=duration,
                error=error
            )

        biomarkers = self.extract_voice_biomarkers(y)
        print("[DEBUG] Acoustic biomarkers:", biomarkers)

        return SimpleNamespace(
            pitch_std=biomarkers["pitch_volatility"],
            energy_rms=biomarkers["rms_energy"],
            duration=duration,
            error=None,

            # New values available to SVI later
            median_pitch=biomarkers["median_pitch"],
            energy_variation=biomarkers["energy_variation"],
            voiced_frames=biomarkers["voiced_frames"]
        )