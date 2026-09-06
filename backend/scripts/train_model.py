import os
import glob
import pickle
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Optional

DATASET_PATH = "dataset"
MODEL_SAVE_PATH = os.path.join("backend", "emotion_model.pkl")

# Emotion mapping across all 4 datasets
EMOTION_MAP = {
    # RAVDESS codes
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust",
    # TESS & CREMA-D codes
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
    "angry": "angry", "disgust": "disgust", "fear": "fearful",
    "happy": "happy", "neutral": "neutral", "ps": "happy", "sad": "sad",
    # SAVEE codes
    "a": "angry", "d": "disgust", "f": "fearful",
    "h": "happy", "neutral": "neutral", "sa": "sad", "su": "happy"
}

def extract_features(file_path: str):
    """Extract acoustic features matching audio_processor.py schema (180 features)."""
    try:
        y, sr = librosa.load(file_path, duration=3.0, offset=0.5, sr=22050)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        return np.hstack([mfcc, chroma, mel, rms, zcr])
    except Exception:
        return None

def parse_ravdess(base_path: str):
    features, labels = [], []
    ravdess_path = os.path.join(base_path, "RAVDESS")
    if not os.path.exists(ravdess_path):
        ravdess_path = base_path

    for actor_dir in glob.glob(os.path.join(ravdess_path, "Actor_*")):
        for file_path in glob.glob(os.path.join(actor_dir, "*.wav")):
            parts = os.path.basename(file_path).split("-")
            if len(parts) >= 3 and parts[2] in EMOTION_MAP:
                feat = extract_features(file_path)
                if feat is not None:
                    features.append(feat)
                    labels.append(EMOTION_MAP[parts[2]])
    return features, labels

def parse_tess(base_path: str):
    features, labels = [], []
    tess_path = os.path.join(base_path, "TESS")
    if not os.path.exists(tess_path):
        return features, labels

    for folder in os.listdir(tess_path):
        folder_path = os.path.join(tess_path, folder)
        if os.path.isdir(folder_path):
            folder_lower = folder.lower()
            for key in ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]:
                if key in folder_lower:
                    detected_emotion = EMOTION_MAP[key]
                    for file_path in glob.glob(os.path.join(folder_path, "*.wav")):
                        feat = extract_features(file_path)
                        if feat is not None:
                            features.append(feat)
                            labels.append(detected_emotion)
                    break
    return features, labels

def parse_cremad(base_path: str):
    features, labels = [], []
    crema_path = os.path.join(base_path, "CREMA-D")
    if not os.path.exists(crema_path):
        crema_path = os.path.join(base_path, "AudioWAV")
        if not os.path.exists(crema_path):
            return features, labels

    for file_path in glob.glob(os.path.join(crema_path, "*.wav")):
        parts = os.path.basename(file_path).split("_")
        if len(parts) >= 3 and parts[2] in EMOTION_MAP:
            feat = extract_features(file_path)
            if feat is not None:
                features.append(feat)
                labels.append(EMOTION_MAP[parts[2]])
    return features, labels

def parse_savee(base_path: str):
    features, labels = [], []
    savee_path = os.path.join(base_path, "SAVEE")
    if not os.path.exists(savee_path):
        savee_path = os.path.join(base_path, "AudioData") # common subfolder name
        if not os.path.exists(savee_path):
            return features, labels

    # SAVEE filename convention: DC_a01.wav, JE_f01.wav, DC_sa01.wav
    for root, _, files in os.walk(savee_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                filename = os.path.basename(file).split(".")[0] # e.g., DC_a01
                code_part = filename.split("_")[-1] # e.g., a01 or sa01
                
                # Extract alphabetical emotion code
                code = "".join([c for c in code_part if not c.isdigit()])
                
                if code in EMOTION_MAP:
                    feat = extract_features(file_path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(EMOTION_MAP[code])
    return features, labels

def train():
    print("Extracting features across all 4 datasets (RAVDESS + TESS + CREMA-D + SAVEE)...")
    
    x_rav, y_rav = parse_ravdess(DATASET_PATH)
    x_tess, y_tess = parse_tess(DATASET_PATH)
    x_crm, y_crm = parse_cremad(DATASET_PATH)
    x_sav, y_sav = parse_savee(DATASET_PATH)
    
    X = np.array(x_rav + x_tess + x_crm + x_sav)
    y = np.array(y_rav + y_tess + y_crm + y_sav)
    
    print(f"Dataset Summary -> RAVDESS: {len(y_rav)} | TESS: {len(y_tess)} | CREMA-D: {len(y_crm)} | SAVEE: {len(y_sav)} | Total: {len(y)}")
    
    if len(y) == 0:
        print("Error: No audio files parsed. Check dataset folder names in 'dataset/'.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest Classifier on Combined Multimodal Dataset...")
    model = RandomForestClassifier(n_estimators=250, max_depth=22, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Trained Successfully! Combined Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved production-grade model artifact to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    train()