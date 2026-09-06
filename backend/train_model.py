import os
import glob
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# RAVDESS Emotion Mapping according to file naming conventions:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Focus on key distress vs neutral emotions for SIH
TARGET_EMOTIONS = ['neutral', 'calm', 'sad', 'angry', 'fearful']

def extract_features(file_path):
    """Extract MFCC, Chroma, and Mel-Spectrogram features from audio file."""
    y, sr = librosa.load(file_path, sr=16000, duration=3, offset=0.5)
    
    # 1. MFCC (40 features)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    
    # 2. Chroma (12 features)
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    
    # 3. Mel-Spectrogram (128 features)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    return np.hstack([mfcc, chroma, mel])

def load_data(dataset_path):
    x, y = [], []
    print("Extracting features from RAVDESS dataset... This will take 2-4 minutes.")
    
    search_path = os.path.join(dataset_path, "Actor_*", "*.wav")
    files = glob.glob(search_path)
    
    if len(files) == 0:
        # Fallback search if files are directly in dataset folder
        files = glob.glob(os.path.join(dataset_path, "*.wav"))

    for file in files:
        file_name = os.path.basename(file)
        parts = file_name.split("-")
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in EMOTIONS:
                emotion = EMOTIONS[emotion_code]
                if emotion in TARGET_EMOTIONS:
                    feature = extract_features(file)
                    x.append(feature)
                    y.append(emotion)
                
    return np.array(x), np.array(y)

if __name__ == "__main__":
    DATASET_PATH = "dataset"
    
    if not os.path.exists(DATASET_PATH):
        print("ERROR: Download RAVDESS dataset from Kaggle and place Actor_* folders in 'dataset/' directory!")
        exit(1)
        
    X, y = load_data(DATASET_PATH)
    print(f"Dataset Loaded! Total Valid Samples: {len(X)}")
    
    if len(X) == 0:
        print("ERROR: No valid audio files found. Check your dataset directory structure.")
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Emotion Classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Trained Successfully! Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))
    
    os.makedirs("backend", exist_ok=True)
    model_save_path = os.path.join("backend", "emotion_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained model to '{model_save_path}'.")