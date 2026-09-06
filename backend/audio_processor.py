import librosa
import numpy as np
from types import SimpleNamespace

class AudioProcessor:
    @staticmethod
    def validate_and_load(file_path: str, target_sr: int = 16000):
        """Loads audio and validates duration and signal energy guardrails."""
        try:
            y, sr = librosa.load(file_path, sr=target_sr)
            duration = float(librosa.get_duration(y=y, sr=sr))
            
            if duration < 0.8:
                return None, 0.0, "Audio too short (< 0.8s)"
                
            rms = float(np.mean(librosa.feature.rms(y=y)))
            if rms < 0.001:
                return None, duration, "Silent or empty audio input"
                
            return y, duration, None
        except Exception as e:
            return None, 0.0, f"Error processing audio file: {str(e)}"

    @staticmethod
    def extract_voice_biomarkers(y, sr: int = 16000) -> dict:
        """Extracts pitch volatility and RMS energy for acoustic SVI weighting."""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        pitch_volatility = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        
        return {
            "pitch_volatility": round(pitch_volatility, 2),
            "rms_energy": round(rms_energy, 4)
        }

    def extract_acoustic_features(self, file_path: str):
        """Bridge method required by svi_engine.py."""
        y, duration, error = self.validate_and_load(file_path)
        
        if error or y is None:
            # Fallback for short/silent audio to prevent engine crash
            return SimpleNamespace(pitch_std=0.0, energy_rms=0.0, duration=duration, error=error)
            
        biomarkers = self.extract_voice_biomarkers(y)
        
        # Returns object accessible via dot notation (.pitch_std and .energy_rms)
        return SimpleNamespace(
            pitch_std=biomarkers["pitch_volatility"],
            energy_rms=biomarkers["rms_energy"],
            duration=duration,
            error=None
        )