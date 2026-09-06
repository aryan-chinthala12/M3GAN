import pickle
import numpy as np
import librosa
import os

MODEL_PATH = os.path.join("backend", "emotion_model.pkl")

# Load ML Model if available
ml_model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)

def extract_ml_features(file_path):
    """Extract features matching training feature vector shape."""
    y, sr = librosa.load(file_path, sr=16000, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel]).reshape(1, -1)

def calculate_ml_svi(file_path: str, acoustic_metrics: dict) -> dict:
    global ml_model
    # Reload model if created after server startup
    if ml_model is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            ml_model = pickle.load(f)

    if ml_model is None:
        raise RuntimeError("Model 'emotion_model.pkl' not found! Run train_model.py first.")

    # 1. Get ML Emotion Probabilities
    features = extract_ml_features(file_path)
    probabilities = ml_model.predict_proba(features)[0]
    classes = ml_model.classes_
    
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    
    p_fear = prob_dict.get('fearful', 0.0)
    p_angry = prob_dict.get('angry', 0.0)
    p_sad = prob_dict.get('sad', 0.0)

    # 2. Compute Hybrid SVI (60% ML Emotion Inference + 40% Pitch/Tremor Acoustics)
    ml_distress_score = (p_fear * 0.50) + (p_angry * 0.35) + (p_sad * 0.15)
    acoustic_score = min(1.0, acoustic_metrics["pitch_variance"] / 80.0)
    
    raw_svi = ((0.60 * ml_distress_score) + (0.40 * acoustic_score)) * 100.0
    svi_score = round(min(100.0, max(0.0, raw_svi)), 1)

    # 3. Categorize Risk Levels & Recommend Actions
    if svi_score >= 75.0:
        risk_level = "CRITICAL"
        color = "#D50000"
        actions = [
            "Immediate Priority Transfer to Senior Trauma Counselor",
            "Auto-dispatch Emergency Alert to District Police Desk",
            "Enable line-tracing protocol and emergency logging"
        ]
    elif svi_score >= 50.0:
        risk_level = "HIGH"
        color = "#FF6D00"
        actions = [
            "Escalate call to Senior Duty Supervisor",
            "Flag profile for Immediate Psychological First Aid (PFA)",
            "Queue parallel dispatch for local legal aid officer"
        ]
    elif svi_score >= 25.0:
        risk_level = "MODERATE"
        color = "#FFD600"
        actions = [
            "Route to standard counseling intake queue",
            "Schedule follow-up welfare check within 24 hours"
        ]
    else:
        risk_level = "LOW"
        color = "#00C853"
        actions = [
            "Standard grievance intake and administrative logging"
        ]

    predicted_emotion = "unknown"
    if prob_dict:
        predicted_emotion = max(prob_dict.keys(), key=lambda k: float(prob_dict[k]))
    return {
        "svi_score": svi_score,
        "risk_level": risk_level,
        "risk_color": color,
        "predicted_emotion": predicted_emotion.upper(),
        "emotion_probabilities": {k: round(v * 100, 1) for k, v in prob_dict.items()},
        "recommended_actions": actions
    }