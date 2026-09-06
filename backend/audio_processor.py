import librosa
import numpy as np

def extract_acoustic_features(file_path: str) -> dict:
    # 1. Load Audio File (Resampled to 16kHz Mono)
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    
    if duration < 0.5:
        raise ValueError("Audio duration too short for analysis (min 0.5s required).")

    # 2. Extract Pitch (F0) using Probabilistic YIN (pyin)
    # Range set to human vocal bounds: 65 Hz to 2093 Hz
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2093, sr=sr)
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else []
    
    pitch_variance = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0

    # 3. Extract Volume / Energy Volatility (RMS)
    rms = librosa.feature.rms(y=y)[0]
    rms_volatility = float(np.std(rms))

    # 4. Extract Pause / Silence Ratio
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    if len(non_silent_intervals) > 0:
        speech_duration = sum([(end - start) for start, end in non_silent_intervals]) / sr
    else:
        speech_duration = 0.0
        
    speech_duration = float(speech_duration)
    pause_duration = max(0.0, duration - speech_duration)
    pause_ratio = float(pause_duration / duration) if duration > 0 else 0.0

    return {
        "duration_seconds": round(duration, 2),
        "pitch_variance": round(pitch_variance, 2),
        "rms_volatility": round(rms_volatility, 4),
        "pause_ratio": round(pause_ratio, 2)
    }