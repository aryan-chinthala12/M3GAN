import librosa
import numpy as np
from types import SimpleNamespace


class AudioProcessor:

    @staticmethod
    def validate_and_load(file_path: str, target_sr: int = 16000):
        """Loads audio and validates duration and signal energy guardrails."""

        try:
            y, sr = librosa.load(
                file_path,
                sr=target_sr,
                mono=True
            )

            duration = float(
                librosa.get_duration(
                    y=y,
                    sr=sr
                )
            )

            # -------------------------------------------------
            # Duration validation
            # -------------------------------------------------

            if duration < 0.8:
                return None, 0.0, "Audio too short (< 0.8s)"

            # -------------------------------------------------
            # Silence validation
            # -------------------------------------------------

            rms = float(
                np.mean(
                    librosa.feature.rms(y=y)
                )
            )

            if rms < 0.001:
                return None, duration, "Silent or empty audio input"

            return y, duration, None

        except Exception as e:

            return (
                None,
                0.0,
                f"Error processing audio file: {str(e)}"
            )

    # =========================================================
    # VOICE BIOMARKERS
    # =========================================================

    @staticmethod
    def extract_voice_biomarkers(
        y,
        sr: int = 16000
    ) -> dict:

        """
        Extract acoustic features for SVI.

        Pitch volatility is calculated using fundamental
        frequency (F0) and expressed as standard deviation
        in semitones relative to the speaker's median pitch.

        This is much more robust than taking the standard
        deviation of raw Hz values from librosa.piptrack().
        """

        # =====================================================
        # 1. FUNDAMENTAL FREQUENCY (F0)
        # =====================================================

        try:

            f0 = librosa.yin(
                y,
                fmin=70.0,
                fmax=400.0,
                sr=sr,
                frame_length=2048,
                hop_length=256
            )

            # Remove invalid / unvoiced frames.
            f0 = f0[
                np.isfinite(f0)
                & (f0 > 70.0)
                & (f0 < 400.0)
            ]

        except Exception:

            f0 = np.array([])

        # =====================================================
        # 2. PITCH VOLATILITY
        # =====================================================

        if len(f0) >= 3:

            # Use the speaker's median pitch as the reference.
            median_f0 = float(
                np.median(f0)
            )

            # Convert pitch to relative semitone values.
            #
            # This measures pitch variation relative to the
            # speaker rather than absolute Hz differences.

            pitch_semitones = (
                12.0
                * np.log2(
                    f0 / median_f0
                )
            )

            pitch_volatility = float(
                np.std(pitch_semitones)
            )

        else:

            pitch_volatility = 0.0

        # =====================================================
        # 3. RMS ENERGY
        # =====================================================

        rms_energy = float(
            np.mean(
                librosa.feature.rms(
                    y=y
                )
            )
        )

        # =====================================================
        # 4. RETURN BIOMARKERS
        # =====================================================

        return {
            "pitch_volatility": round(
                pitch_volatility,
                2
            ),

            "rms_energy": round(
                rms_energy,
                4
            )
        }

    # =========================================================
    # ENGINE BRIDGE
    # =========================================================

    def extract_acoustic_features(
        self,
        file_path: str
    ):

        """
        Bridge method required by svi_engine.py.
        """

        y, duration, error = (
            self.validate_and_load(
                file_path
            )
        )

        # -----------------------------------------------------
        # Invalid / silent audio
        # -----------------------------------------------------

        if error or y is None:

            return SimpleNamespace(
                pitch_std=0.0,
                energy_rms=0.0,
                duration=duration,
                error=error
            )

        # -----------------------------------------------------
        # Extract biomarkers
        # -----------------------------------------------------

        biomarkers = (
            self.extract_voice_biomarkers(
                y
            )
        )

        # -----------------------------------------------------
        # Return object compatible with svi_engine.py
        # -----------------------------------------------------

        return SimpleNamespace(

            # Despite the existing field name "pitch_std",
            # this is now pitch volatility in SEMITONES.
            pitch_std=biomarkers[
                "pitch_volatility"
            ],

            energy_rms=biomarkers[
                "rms_energy"
            ],

            duration=duration,

            error=None
        )