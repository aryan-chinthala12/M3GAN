# M3GAN — AI-Based Real-Time Stress & Vulnerability Assessment Module 🎙️

**Smart India Hackathon (SIH 2024 / Problem Statement 26093)**  
*National Helpline Against Atrocities (NHAA 14566) — Department of Social Justice & Empowerment, Government of India.*

M3GAN is a multimodal decision-support engine designed to perform real-time psychological stress, panic, and vulnerability assessment across emergency helpline digital communications (voice streams, external audio feeds, text reports, and IVRS transcripts). The system computes a continuous **Stress Vulnerability Index (SVI 0–100)** with tiered risk triage (LOW, MEDIUM, HIGH, CRITICAL), independent safety flags, and actionable intervention guidance for human operators.

---

> [!IMPORTANT]
> **RESEARCH & DECISION-SUPPORT DISCLAIMER**  
> M3GAN is an engineering research prototype and decision-support tool designed strictly to assist trained human operators during emergency triage. It is **NOT** a clinical diagnostic system, medical evaluation device, or autonomous emergency dispatch authority. All SVI scores, risk bands, and recommendations require human-in-the-loop review.

---

## ⚡ Key Capabilities

* **Multimodal Feature Fusion**: Combines acoustic signal volatility (F0 pitch, RMS energy, spectral tilt, ZCR), Wav2Vec 2.0 speech emotion recognition (SER), Faster-Whisper ASR transcription, and multilingual threat/distress NLP.
* **Real-Time WebSocket Streaming**: Ingests continuous 16 kHz PCM audio frames over WebSockets and delivers live `assessment_update` payloads to the React operator console.
* **Dual Audio Channel Ingestion**: Supports browser microphone streaming directly from the web client and external hardware audio ingestion (USB headsets, line-in interfaces, soundcards, and virtual audio cables).
* **Indic & Code-Mixed NLP**: Transliterates Romanized Hinglish and analyzes threat markers, distress intensity, and contextual negation across English and regional Indian languages (Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati).
* **Signal Quality & Whisper Handling**: Integrates Silero VAD and rolling acoustic feature normalization to separate signal degradation (low SNR, quiet speech, whisper) from true psychological distress.
* **Idempotent Session Lifecycle**: Preserves session state and finalized summaries (`POST /stop` $\rightarrow$ SQLite case storage $\rightarrow$ `GET /summary`).

---

## 🏗️ System Architecture

### Pipeline Flow

```text
 ┌──────────────────────────────┐          ┌──────────────────────────────┐
 │     Browser Microphone       │          │   External Audio Hardware    │
 │ (16kHz Mono PCM WebSockets)  │          │ (Mic / Headset / Line-In /   │
 └──────────────┬───────────────┘          │  VB-CABLE Virtual Audio Feed)│
                │                          └──────────────┬───────────────┘
                │                                         │
                └───────────────────┬─────────────────────┘
                                    ▼
                      audio_hardware_ingest.py / API
                                    │
                                    ▼
                       StreamingAssessmentSession
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │                                                               │
    ▼                                                               ▼
Silero VAD & Signal Quality                             Speech & Text Intelligence
(RMS, F0, Spectral Tilt, ZCR,                           ├──► Wav2Vec 2.0 SER (emotion_model.pkl fallback)
 Noise Floor & SNR Classification)                       ├──► Faster-Whisper ASR Transcription
                                                        └──► Multilingual / Hinglish NLP Scanner
    │                                                               │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
                      Multimodal Dynamic SVI Engine
             (SVI 0–100 Score, Confidence Metric & Trend Direction)
                                    │
                                    ├──► Independent Safety Flag Evaluator
                                    │
                                    ▼
              Thread-Safe WebSocket Broadcaster (/ws/stream)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   React Operator Console                         SQLite Case Store
(Live Gauges, Volatility & Alerts)            (Persistent case_store.db Registry)
```

### Layer & Component Breakdown

1. **Audio Ingestion Layer**
   - **Hardware Audio Ingestion (`backend/audio/audio_hardware_ingest.py`)**: PyAudio multi-device soundcard capture engine supporting USB headsets, hardware microphones, line-in inputs, and virtual software audio routing (e.g. VB-CABLE for routing external call playback).  
     *Note: Standard USB charging cables connected to mobile phones do NOT directly transmit cellular call audio to host soundcards; external line-in interface hardware or virtual audio cable routing is required.*
   - **Browser Microphone Ingestion (`backend/api/websocket_handler.py`)**: Captures raw 16 kHz 16-bit mono PCM audio frames directly from browser clients over WebSockets.

2. **Streaming & Session Layer**
   - **StreamingSessionManager (`backend/streaming/streaming_session.py`)**: Authoritative in-memory state manager maintaining active `StreamingAssessmentSession` instances, sliding audio window buffers, acoustic history, and rolling SVI trend timelines.
   - **Thread-Safe WebSocket Broadcaster (`backend/api/websocket_handler.py`)**: Schedules real-time `assessment_update` message sends onto the active asyncio event loop when hardware ingestion callbacks fire from background threads.

3. **Machine Learning & NLP Analysis Layer**
   - **Silero VAD (`backend/audio/vad_processor.py`)**: PyTorch voice activity detection model filtering silent frames and non-speech background noise.
   - **Signal Quality & Acoustic Feature Extractor (`backend/audio/acoustic_normalizer.py` & `backend/audio/audio_processor.py`)**: Computes fundamental frequency (F0 pitch volatility), RMS energy, spectral tilt, zero-crossing rate (ZCR), rolling noise floor, and SNR classification (distinguishing low SNR/whisper from psychological panic).
   - **Speech Emotion Recognition (SER) (`backend/audio/audio_processor.py`)**: Wav2Vec 2.0 transformer model (with `backend/models/emotion_model.pkl` fallback) predicting acoustic emotion state probabilities.
   - **Automated Speech Recognition (ASR) (`backend/audio/audio_processor.py`)**: Faster-Whisper transformer engine generating continuous speech-to-text transcripts.
   - **Multilingual & Hinglish NLP (`backend/nlp/`)**: Transliterates Romanized Hinglish script (`hinglish_normalizer.py`), detects language family (`language_detector.py`), scans distress/threat lexicons (`lexicon.py`), and evaluates contextual intensity and negation (`contextual_nlp.py`, `text_analyzer.py`).

4. **SVI Fusion Engine**
   - **Dynamic SVI Engine (`backend/core/svi_engine.py`)**: Fuses acoustic volatility, speech emotion probabilities, ASR transcripts, and NLP threat scores into a unified **Stress Vulnerability Index (SVI 0–100)** score, confidence metric, and trend direction.

5. **Independent Safety Flags**
   - **Safety Flag Evaluator (`backend/core/svi_engine.py`)**: Evaluates autonomous boolean safety flags (`HIGH_THREAT_KEYWORDS`, `EXPLICIT_SELF_HARM`, `VOICE_PANIC_DETECTED`, `SIGNAL_DEGRADED`) independently of numerical SVI weights, ensuring high-risk emergency markers trigger immediate visual warnings.

6. **API & WebSocket Layer**
   - **FastAPI REST API (`backend/main.py`)**: REST routes for session management (`POST /session/start`), hardware controls (`POST /audio/hardware/start`, `POST /audio/hardware/stop`), session finalization (`POST /session/{id}/stop`), summary fetch (`GET /session/{id}/summary`), and text analysis (`POST /evaluate/text`).
   - **WebSocket Gateway (`backend/api/websocket_handler.py`)**: WebSocket streaming server (`/ws/stream`) managing connection lifecycles, PCM frame reception, and live broadcast dispatching.

7. **Frontend Operator Console**
   - **React 19 + Vite Dashboard (`frontend/`)**: Real-time decision-support interface displaying live SVI gauges, risk triage badges (LOW, MEDIUM, HIGH, CRITICAL), acoustic volatility charts, transcript feeds, and safety alerts for human operators.  
     *Note: M3GAN is designed strictly as a decision-support system for human operators—it does NOT make autonomous clinical diagnoses or trigger automated emergency dispatch.*

8. **Persistence Layer**
   - **SQLite Case Store (`backend/core/case_store.py`)**: Embedded SQLite database (`case_store.db`) providing persistent storage for finalized case summaries, session parameters, and transcript audit logs.

---

## 💻 Technology Stack

* **Backend Server**: Python 3.10+, FastAPI 0.140+, Uvicorn, WebSockets
* **Machine Learning & DSP**: PyTorch 2.14+, Transformers 4.44+, Faster-Whisper 1.2+, Librosa 0.11+, SciPy 1.17+
* **Voice Activity & Audio Ingestion**: Silero VAD, PyAudio 0.2.14, Soundfile
* **Database & Persistence**: SQLite 3 (`case_store.py`)
* **Frontend Console**: React 19, Vite 8, Tailwind CSS 4, Lucide React

---

## 📁 Project & File Architecture

```text
M3GAN/
├── backend/                         # FastAPI Python backend application root
│   ├── main.py                      # FastAPI entrypoint, REST API endpoints & lifecycle initialization
│   ├── api/                         # Web & API interface package
│   │   ├── schemas.py               # Pydantic request/response models & WebSocket payload schemas
│   │   └── websocket_handler.py     # WebSocket endpoint (/ws/stream) & thread-safe event loop broadcaster
│   ├── audio/                       # Audio ingestion & Digital Signal Processing (DSP)
│   │   ├── acoustic_normalizer.py   # Rolling acoustic feature normalizer & whisper/SNR classifier
│   │   ├── audio_hardware_ingest.py # Multi-device soundcard/line-in capture engine (PyAudio & VB-CABLE support)
│   │   ├── audio_processor.py       # DSP pipeline, F0 tracking, Wav2Vec2 SER & Faster-Whisper ASR
│   │   └── vad_processor.py         # Silero VAD PyTorch wrapper for voice presence detection
│   ├── core/                        # Core engine & configuration logic
│   │   ├── case_store.py            # Embedded SQLite persistent case repository (case_store.db)
│   │   ├── config.py                # SVI dynamic weighting weights, risk thresholds & intervention rules
│   │   └── svi_engine.py            # Multimodal SVI dynamic fusion engine & safety flag evaluator
│   ├── models/                      # ML model assets & serialized checkpoints
│   │   └── emotion_model.pkl        # Serialized SVM speech emotion classifier asset
│   ├── nlp/                         # Multilingual & Indic Natural Language Processing
│   │   ├── contextual_nlp.py        # Contextual threat intensity scanner & negation processor
│   │   ├── hinglish_normalizer.py   # Romanized Hindi/Indic script transliterator & normalizer
│   │   ├── language_detector.py     # Script-aware language identification engine
│   │   ├── lexicon.py               # Multilingual threat, distress & emergency keyword dictionaries
│   │   └── text_analyzer.py         # Written narrative analysis pipeline
│   ├── streaming/                   # Real-time state management
│   │   └── streaming_session.py     # Authoritative StreamingSessionManager & sliding audio windows
│   ├── scripts/                     # Benchmarking, training & profiling utilities
│   │   ├── ablation_benchmark.py
│   │   ├── benchmark_phase5c.py
│   │   ├── resource_profiler.py
│   │   └── train_model.py
│   └── tests/                       # Automated test suite & verification scripts
│       ├── test_end_to_end_integration.py # 15-test full integration suite
│       ├── verify_hardware_broadcaster.py  # Hardware audio WebSocket broadcast verifier
│       └── verify_manual_lifecycle.py      # Session lifecycle & finalization verifier
├── frontend/                        # React 19 + Vite + Tailwind CSS Operator Console
│   ├── src/
│   │   ├── api/                     # REST & WebSocket API client integration
│   │   ├── components/              # SVI gauge, trend timeline, triage bar & safety flags UI
│   │   └── pages/                   # Live Assessment, Dashboard & Cases pages
│   ├── package.json
│   └── vite.config.js
├── sample_data/                     # Benchmark text corpora & test evaluation samples
├── .gitignore                       # Clean Python, Node & local asset git ignore
├── requirements.txt                 # Backend Python package dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Setup & Local Development Guide

### Prerequisites
* Windows 10/11 (or Linux/macOS)
* Python 3.10 or newer
* Node.js 18+ and npm

### 1. Backend Environment Setup

```bash
# Clone repository and navigate to root
cd M3GAN

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Launch Backend Server

```bash
# Set environment variables for PyTorch / OpenMP compatibility (PowerShell)
$env:PYTHONPATH="."
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"

# Launch FastAPI backend on port 8000
python -m uvicorn backend.main:app --reload --port 8000
```
* Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

### 3. Launch Frontend Operator Console

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
* Access Operator Console: `http://localhost:5173`

---

## 🔌 Hardware & External Audio Ingestion Setup

To capture external phone calls, softphone streams, or hardware microphones:

1. **USB Headset / Microphone**: Connect device and select it from the **Hardware Input Interface** dropdown in the Live Assessment console.
2. **Line Input / Soundcard**: Connect 3.5mm line-out from an external phone interface into the soundcard line-in.
3. **Virtual Audio Cable (VB-CABLE)**:
   * Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/).
   * Direct softphone or call playback output to `CABLE Input (VB-Audio Virtual Cable)`.
   * In M3GAN Live Assessment, select `CABLE Output (VB-Audio Virtual Cable)` as the input source.

---

## 🧪 Running Verification & Test Suites

Run the automated test suite from the repository root:

```bash
# Set PYTHONPATH to project root
$env:PYTHONPATH="."

# 1. Full 15-test End-to-End Integration Suite
python -m unittest backend/tests/test_end_to_end_integration.py

# 2. Hardware Audio WebSocket Broadcaster Test
python backend/tests/verify_hardware_broadcaster.py

# 3. Manual Session Lifecycle Test (Start -> Stop -> Summary)
python backend/tests/verify_manual_lifecycle.py

# 4. Frontend Production Build Verification
npm --prefix frontend run build
```

---

## 📊 Empirical Validation Results

Controlled engineering validation results from the current codebase:

| Verification Target | Command / Test | Result | Performance / Status |
| :--- | :--- | :--- | :--- |
| **End-to-End Integration** | `test_end_to_end_integration.py` | **PASSED (15/15)** | All 15 tests executed cleanly in 94.8s. |
| **Hardware Broadcaster** | `verify_hardware_broadcaster.py` | **PASSED (100%)** | Verified background thread WebSocket update dispatching. |
| **Session Lifecycle** | `verify_manual_lifecycle.py` | **PASSED (100%)** | Verified idempotent finalization (`POST /stop` $\rightarrow$ 200 OK). |
| **Frontend Production Build** | `npm run build` | **PASSED** | Compiled cleanly in 2.45s (`dist/assets/index-Bw29hla5.js`). |
| **Browser Mic Live Stream** | Demo Validation Mode | **VALIDATED** | Real-time SVI updates received via WebSocket (~601ms warm latency). |
| **Hardware Audio Ingestion** | Demo Validation Mode | **VALIDATED** | Thread-safe WebSocket update broadcast verified (~1.0ms broadcast latency). |

---

## ⚠️ Known Limitations & Current Non-Goals

* **Decision Support Only**: M3GAN does not automate emergency service dispatch or legal decision-making.
* **Controlled Validation Scope**: Validation results represent controlled prototype benchmarks, not formal clinical trials.
* **PyAudio Driver Dependency**: Hardware audio ingestion requires host OS audio driver support and permissions.
* **Experimental Dynamic Weighting**: Quiet-speech and low-SNR dynamic weighting configurations remain experimental baseline candidates.

---

## 🛣️ Development Roadmap

- [ ] **Telephony Gateway Bridge**: SIP/RTP protocol adapter for direct PBX helpline integration.
- [ ] **Regional Accent Fine-Tuning**: Indian accent fine-tuning for speech emotion models.
- [ ] **Edge Execution Acceleration**: ONNX Runtime and TensorRT model export for low-power edge deployment.
- [ ] **Multi-Operator Session Handover**: Real-time collaborative triage queue management for supervisor teams.
