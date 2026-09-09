# M3GAN 🎙️

### AI-Assisted Multimodal Stress & Trauma Assessment System

M3GAN is an AI-assisted multimodal assessment system designed to help helpline operators identify distress and potentially high-risk situations from voice interactions.

The system combines **speech emotion analysis, acoustic voice biomarkers, speech-to-text transcription, and linguistic distress indicators** to calculate a **Stress Vulnerability Index (SVI)** and provide a separate **Situation Severity** classification for human review.

M3GAN is being developed in the context of the **Smart India Hackathon 2026** problem statement:

> **SIH26093 — AI-Based Real-Time Stress and Trauma Assessment Module for Victims/Complainants Accessing NHAA (14566) and Integrated Portal**

The system is designed as a **decision-support tool**. It does not replace trained helpline operators, clinical professionals, or authorized emergency responders.

---

## 📌 Problem Statement

Victims and complainants contacting emergency or support helplines may experience severe emotional distress, fear, shock, panic, or trauma.

During a high-priority call, manually assessing the caller's condition can be difficult because important signals may appear simultaneously in:

- The caller's voice
- Speech patterns
- Emotional expression
- Acoustic characteristics
- Spoken content

M3GAN addresses this challenge by performing multimodal analysis of incoming audio and presenting interpretable indicators to an authorized human operator.

Instead of relying on a single signal, M3GAN combines multiple modalities to provide a broader assessment of the caller's current distress state.

---

## 🎯 Objectives

M3GAN aims to:

- Analyze incoming voice recordings using multiple AI and signal-processing techniques.
- Extract acoustic indicators such as pitch volatility and vocal energy.
- Perform Speech Emotion Recognition (SER).
- Convert speech into text using Speech-to-Text (STT).
- Detect distress-related linguistic indicators.
- Calculate a normalized **Stress Vulnerability Index (SVI)** from 0–100.
- Separately classify the operational **Situation Severity**.
- Detect explicit high-risk safety indicators through a deterministic safety override.
- Provide explainable assessment factors.
- Recommend appropriate next-step actions for human operators.
- Support both **live microphone recording** and **uploaded audio files**.

---

# ⚡ Key Features

## 🎙️ Live Voice Assessment

The frontend can capture audio directly from the user's microphone through the browser.

The captured recording is sent to the FastAPI backend for multimodal analysis.

---

## 📁 Audio File Assessment

M3GAN also supports assessment from previously recorded audio files.

Supported formats include:

- WAV
- MP3
- M4A
- WebM
- OGG
- FLAC

Live recordings and uploaded recordings use the same backend analysis pipeline.

---

## 🧠 Speech Emotion Recognition

M3GAN uses a Wav2Vec 2.0-based Speech Emotion Recognition model to estimate the emotional characteristics of the speaker.

The system considers the **emotion probability distribution** rather than treating the confidence of only the top emotion as the complete distress score.

---

## 🎚️ Acoustic Voice Analysis

The audio processing pipeline extracts voice biomarkers including:

- Fundamental frequency (F0)
- Pitch volatility
- Median pitch
- RMS energy
- Energy variation
- Number of voiced frames

Pitch volatility is calculated using robust voiced F0 estimation and converted into a relative semitone-based measure, reducing dependence on the speaker's absolute pitch.

---

## 📝 Speech-to-Text

The system uses **Faster-Whisper** to convert incoming speech into text.

The resulting transcript is then used for linguistic analysis and explainability.

---

## 🔎 Linguistic Distress Indicators

The transcript is analyzed for distress-related linguistic indicators.

Ordinary distress-related language contributes to the linguistic component of the SVI.

Explicit high-risk safety indicators are handled separately through the safety-override mechanism.

---

# 📊 Stress Vulnerability Index (SVI)

The **Stress Vulnerability Index (SVI)** is a numerical score from:

```text
0 → 100
```

The current multimodal SVI combines three components:

```text
SVI =
    45% Emotion Distress
  + 30% Acoustic Stress
  + 25% Linguistic Distress
```

### 1. Emotion Distress — 45%

Derived from the model's complete emotion probability distribution.

This provides a more stable representation of emotional distress than simply using the highest-confidence emotion.

### 2. Acoustic Stress — 30%

Derived from acoustic voice characteristics including:

- Pitch volatility
- RMS energy
- Vocal energy variation

### 3. Linguistic Distress — 25%

Derived from distress-related indicators detected in the transcript.

---

# 🚨 SVI Score vs Situation Severity

M3GAN intentionally separates the **numerical SVI Score** from the operational **Situation Severity**.

### SVI Score

The SVI is a continuous numerical indicator from 0 to 100.

It represents the combined multimodal assessment.

### Situation Severity

Situation Severity represents the operational classification used to prioritize human attention.

Current severity bands are:

| SVI Score | Situation Severity |
|---:|---|
| `< 25` | 🟢 **LOW** |
| `25 – 49.99` | 🟡 **MODERATE** |
| `50 – 74.99` | 🟠 **HIGH** |
| `≥ 75` | 🔴 **CRITICAL** |

However, Situation Severity is **not determined solely by the numerical SVI score**.

---

# 🛡️ Safety Override

Certain explicit high-risk statements require special handling.

M3GAN therefore maintains a separate **safety override mechanism**.

When an explicit high-risk safety indicator is detected, the system can classify the situation as:

```text
CRITICAL
```

even when the numerical SVI score is below the normal CRITICAL threshold.

This separation prevents an explicit safety concern from being hidden by a relatively low numerical multimodal score.

### Example

A caller may receive:

```text
SVI Score: 67.58 / 100

Situation Severity: CRITICAL
```

The numerical score and operational severity are intentionally different.

The safety override does **not** modify the numerical SVI score.

---

# 👤 Human-in-the-Loop Design

M3GAN is designed as an **AI-assisted decision-support system**.

The AI provides:

- Multimodal indicators
- SVI score
- Situation severity
- Explainability
- Recommended interventions
- Safety alerts

The final decision remains with an **authorized human operator**.

M3GAN should not be treated as:

- A clinical diagnostic system
- A replacement for trained counselors
- An autonomous emergency decision-maker
- A substitute for professional assessment

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │     Audio Input     │
                         │                     │
                         │  Live Microphone    │
                         │        OR           │
                         │   Uploaded Audio    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Audio Processing   │
                         │                     │
                         │  • Resampling       │
                         │  • Validation       │
                         │  • F0 Extraction    │
                         │  • RMS Analysis     │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ Acoustic       │ │ Speech Emotion │ │ Speech-to-Text │
        │ Analysis       │ │ Recognition    │ │                │
        │                │ │                │ │ Faster-Whisper │
        │ F0 / RMS /     │ │ Wav2Vec 2.0    │ │                │
        │ Energy         │ │                │ │ Transcript     │
        └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                │                  │                  │
                │                  │                  ▼
                │                  │         ┌────────────────┐
                │                  │         │   Linguistic   │
                │                  │         │    Analysis    │
                │                  │         └───────┬────────┘
                │                  │                 │
                └──────────────────┼─────────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │    SVI Fusion       │
                         │      Engine         │
                         │                     │
                         │ Emotion     45%     │
                         │ Acoustic    30%     │
                         │ Linguistic  25%     │
                         └──────────┬──────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
                      ▼                           ▼
             ┌─────────────────┐        ┌─────────────────┐
             │   SVI Score     │        │ Safety Override │
             │     0–100       │        │                 │
             └────────┬────────┘        └────────┬────────┘
                      │                          │
                      └────────────┬─────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Situation Severity  │
                         │                     │
                         │ LOW / MODERATE /   │
                         │ HIGH / CRITICAL     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Explainability &    │
                         │ Recommended Actions │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Human Operator     │
                         │      Review         │
                         └─────────────────────┘
```

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 |
| Build Tool | Vite |
| UI Styling | Tailwind CSS |
| UI Icons | Lucide React |
| Backend | Python |
| API Framework | FastAPI |
| Server | Uvicorn |
| Audio Processing | Librosa |
| Numerical Processing | NumPy / SciPy |
| Speech Emotion Recognition | Wav2Vec 2.0 |
| Speech-to-Text | Faster-Whisper |
| Data Validation | Pydantic |
| Model Runtime | PyTorch |
| Model Hosting | Hugging Face Hub |
| Version Control | Git / GitHub |

---

# 📁 Project Structure

```text
M3GAN/
│
├── backend/
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   ├── svi_engine.py
│   │
│   └── scripts/
│       └── train_model.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── api/
│   │   │   └── assessment.js
│   │   │
│   │   ├── components/
│   │   ├── data/
│   │   ├── pages/
│