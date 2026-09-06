# M3GAN 🎙️

AI-powered real-time multimodal stress and trauma assessment engine.

**M3GAN** is a production-grade multimodal processing engine designed for real-time distress detection and automated emotional triage on emergency support helplines, including integration pathways for the **National Helpline Against Atrocities (NHAA - 14566)** under the **Department of Social Justice and Empowerment (DoSJE)**, Government of India.

The system extracts real-time vocal biomarkers—pitch volatility and volume energy—and fuses them with Speech Emotion Recognition (SER) and Speech-to-Text (STT) lexical threat detection to compute a dynamic **Stress Vulnerability Index (SVI)** (`0–100`) and trigger emergency response routing.

---

## 📌 Problem Statement & Context

Victims of atrocities and severe distress contacting emergency helplines often suffer from hyperventilation, emotional shock, or acute trauma. Traditional manual triage or text-only chatbots introduce critical delays during high-priority emergency windows.

**M3GAN** solves this by delivering **instant multimodal triage** at both the acoustic signal level (*how a voice sounds*) and the lexical content level (*what is spoken*). This allows helpline operators and emergency responders to immediately identify high-risk callers, bypass manual routing, and prioritize life-saving intervention.

---

## ⚡ Core Capabilities

* **Multimodal Data Fusion:** Fuses Wav2Vec 2.0 Speech Emotion Recognition (SER) with Faster-Whisper Speech-to-Text (STT) transcription.
* **Acoustic Biomarker Signal Processing:** Uses `librosa` to analyze signal energy (RMS) and fundamental pitch variance (`pitch_std`) to detect voice strain and tremors.
* **Lexical Threat Override:** Automatically escalates callers to **CRITICAL** risk when active distress keywords (*suicide*, *overdose*, *harm*, *pain*) are detected in transcripts.
* **Stress Vulnerability Index (SVI):** A mathematical scoring model mapping acoustic strain, emotional weights, and lexical risk to a normalized `0–100` scale.
* **Automated Risk Categorization:** Assigns callers into **LOW**, **MODERATE**, and **CRITICAL** risk tiers accompanied by UI color indicators (`#D32F2F` for Critical).
* **Actionable Protocol Triggers:** Dynamically generates recommended routing actions ranging from standard logging to `IMMEDIATE_HUMAN_DISPATCH`.
* **Microservice Architecture:** Modular FastAPI backend connected to an interactive Streamlit agent dashboard.

---

## 🏗️ Architecture & Processing Pipeline

```text
                     [ Incoming Audio File (.wav / .mp3) ]
                                       │
                                       ▼
                       ┌───────────────┴───────────────┐
                       │   Audio Processing Engine     │
                       └───────────────┬───────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
  ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
  │ Audio Processor   │      │ Wav2Vec 2.0 (SER) │      │  Faster-Whisper   │
  │ Pitch Std / RMS   │      │ Emotion & Conf.   │      │  Transcription    │
  └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
            │                          │                          │
            │                          │                ┌─────────┴─────────┐
            │                          │                │ Keyword Scanner   │
            │                          │                │ Threat Keywords   │
            │                          │                └─────────┬─────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────┴───────────────┐
                       │       SVI Fusion Engine       │
                       │   Score (0-100) + Overrides   │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────┴───────────────┐
                       │ FastAPI Backend (/analyze)    │
                       │ Returns AudioAnalysisResponse │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────┴───────────────┐
                       │   Streamlit Agent Dashboard   │
                       └───────────────────────────────┘
```
---

## 🛠️ Tech Stack

| Layer | Technology | Version / Details |
| :--- | :--- | :--- |
| **Backend** | FastAPI `0.110.0+` | Asynchronous REST API server & router |
| **Audio Processing** | Librosa / SciPy `0.10.1+` | Pitch tracking (`piptrack`), RMS energy, DSP routines |
| **ML Engine** | Wav2Vec 2.0 | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| **STT Engine** | Faster-Whisper `1.0.0+` | CTranslate2-accelerated Speech-to-Text (`tiny` model) |
| **Frontend** | Real-time agent monitoring interface nd something _____|
| **Validation** | Pydantic v2 `2.6.0+` | Strict JSON schema definitions & request parsing |

---

## 🚀 API Endpoints

The backend exposes REST endpoints at `http://127.0.0.1:8000`. Interactive documentation is available via `/docs`.

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/analyze-audio` | Primary multimodal analysis (`.wav`/`.mp3` upload → STT + SER + SVI) |
| `POST` | `/api/v1/interventions/respond` | Record agent acknowledgment and log escalation response |
| `GET` | `/health` | Health check (Engine status + model loading states) |

### Request & Response Details

* **`POST /api/v1/analyze-audio`**
  * **Payload:** `multipart/form-data` with `file`
  * **Response:** Returns `duration_seconds`, `transcript`, `acoustic_indicators`, `nlp_indicators`, `svi_metrics`, and `recommended_interventions`.

* **`POST /api/v1/interventions/respond`**
  * **Payload:** `{"call_id": "string", "operator_id": "string", "action_taken": "string", "notes": "string"}`

---

## 📊 SVI Scoring Model & Risk Bands

The **Stress Vulnerability Index (SVI)** maps raw signal processing values, machine learning outputs, and keyword matches into a unified score scaled from `0` to `100`.

| Score Range | Risk Band | Color Code | Protocol Trigger |
| :--- | :--- | :--- | :--- |
| `0 – 39` | **LOW** | `#2E7D32` (Green) | Routine logging, standard response queue |
| `40 – 69` | **MODERATE** | `#ED6C02` (Orange) | Senior agent review, priority queue placement |
| `70 – 100` | **CRITICAL** | `#D32F2F` (Red) | `IMMEDIATE_HUMAN_DISPATCH`, Safety Team alert |

> **Note:** Any detection of critical lexical keywords (e.g., *suicide*, *harm*) triggers an immediate **CRITICAL** override regardless of acoustic scores.
---

## 📁 Project Directory Structure

```text
M3GAN/
├── backend/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Feature weights, thresholds, and risk parameters
│   ├── audio_processor.py    # Librosa signal processing and biomarker routines
│   ├── schemas.py            # Pydantic schema definitions
│   ├── svi_engine.py         # SVI scoring algorithms, SER, Whisper STT & threat rules
│   └── main.py               # FastAPI router and server initialization
├── frontend/
│   └── xyz           
├── scripts/
│   └── train_model.py        # Offline dataset training & fine-tuning script
├── sample_data/              # Sample audio files for testing
├── requirements.txt          # Python dependency manifest
├── .gitignore                # System and python cache exclusions
└── README.md                 # System documentation