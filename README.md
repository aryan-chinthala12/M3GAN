# M3GAN — AI Stress & Trauma Assessment Module for NHAA (14566) 🎙️

AI-powered real-time multimodal stress and trauma assessment engine for the
**National Helpline Against Atrocities (NHAA — 14566)**, Department of Social
Justice & Empowerment, Government of India.

The system assesses psychological stress, trauma, fear, anxiety and
vulnerability of victims/complainants across **every NHAA digital channel** —
voice calls, the Integrated Portal, chatbot, IVRS transcripts and any approved
digital interface — and produces a **Stress Vulnerability Index (SVI 0–100)**
with Low / Moderate / High / Critical risk triage and automated intervention
recommendations.

---

## ⚡ What it does

| Requirement (Problem Statement) | Implementation |
| :--- | :--- |
| Analyse voice, speech patterns, pauses, pitch variation | `librosa` DSP: semitone pitch volatility, RMS energy, energy variation, median F0 (`audio_processor.py`) |
| NLP / Emotion AI on the narrative | Wav2Vec 2.0 speech-emotion SER + weighted multilingual distress lexicon (11 categories) |
| Multilingual (major Indian languages + dialects) | Whisper STT (en/hi/bn/ta/te/mr/kn) + lexicon in 7 languages **plus Romanized / code-mixed speech**, with script-aware matching for Indic Unicode |
| Stress Vulnerability Index on a predefined scale | SVI 0–100 with explainable per-component contributions |
| Low / Moderate / High / Critical categories | Threshold bands at 25 / 50 / 75 with protocol actions per band |
| Detect severe trauma, suicidal ideation, intimidation, isolation | Severity-weighted categories incl. Suicidal Ideation, Self-Harm, Sexual Violence, Violence/Death Threats, Fear/Intimidation, Social Boycott/Isolation, Grief, Caste-Atrocity context |
| Automatic recommendations (counselling, legal aid, police, protection) | Per-band intervention routing in `config.py` (senior counsellor, district police desk, PFA, legal-aid officer, 24-h welfare check) |
| Privacy, informed consent, confidentiality, ethical AI | **PII redaction before storage** (phones, Aadhaar, email, self-disclosed names), mandatory informed-consent gate in the UI, human-oversight banner, human-in-the-loop escalation endpoint |

### Grounded severity floors (ethical safeguard)

A genuine high-severity disclosure keeps the SVI grounded even when the caller
speaks softly or briefly — e.g. suicidal ideation can never score below **78**.
Floors are *floors, not overrides*: no keyword can fabricate a CRITICAL score,
and every application of a floor is shown to the human reviewer with reasons.

### Explainability

Every assessment returns per-component scores, weights and contributions
(`svi_metrics.components`), flagged multilingual terms, distress categories,
and a plain-language summary — the dashboard renders all of it.

---

## 🏗️ Architecture

```text
   Voice call ──┐                          Chat / Portal / Chatbot / IVRS ──┐
                ▼                                                          ▼
   ┌─────────────────────────┐                            ┌──────────────────────────┐
   │ Faster-Whisper STT      │                            │  PII Redaction           │
   │ Wav2Vec2 Emotion (SER)  │                            │  (phone/Aadhaar/email/   │
   │ librosa biomarkers      │                            │   self-disclosed names)  │
   └───────────┬─────────────┘                            └────────────┬─────────────┘
               │  transcript                                           │
               ▼                                                       ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │        Multilingual Distress Lexicon (backend/lexicon.py)                 │
   │   en · hi (Devanagari+Roman) · bn · ta · te · mr · kn · code-mixed        │
   │   11 weighted categories · longest-span matching · repetition saturation  │
   └───────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │  SVI Fusion Engine        │
                     │  expansion curves +       │
                     │  severity floors → 0–100  │
                     └─────────────┬─────────────┘
                                   ▼
              ┌────────────────────────────────────────┐
              │ FastAPI : risk band · interventions ·  │
              │ explainability · SQLite case store     │
              └────────────────────┬───────────────────┘
                                   ▼
                     React operator dashboard (Vite + Tailwind)
                     dashboard · case queue · live voice/text assessment
```

---

## 🚀 Run it

### Backend (FastAPI, port 8000)

```bash
# first time: create venv + install
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # Linux/Mac

.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

- Swagger docs: `http://127.0.0.1:8000/docs`
- The server starts instantly; SER + Whisper models download/load lazily on
  the first audio request. **Text analysis works immediately, no downloads.**

### Frontend (React dashboard, port 5173)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dashboard auto-detects the backend; if it
is down it shows demo data with a banner instead of crashing.

---

## 📊 API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/analyze-audio` | Voice call: STT + SER + acoustic biomarkers + lexicon → SVI. Form fields: `file`, `language`, `consent` |
| `POST` | `/api/v1/analyze-text` | Chat/portal/chatbot narrative → SVI. JSON: `{text, channel, language}` |
| `GET` | `/api/v1/cases` | Recent assessed cases (dashboard queue) |
| `GET` | `/api/v1/cases/{id}` | Full case record incl. redacted narrative |
| `GET` | `/api/v1/stats` | Aggregates: totals, risk/channel breakdown, avg SVI |
| `POST` | `/api/v1/interventions/respond` | Human-in-the-loop: operator acknowledges/escalates a case |
| `GET` | `/health` | Engine + model status |

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Woh log mujhe dhamki de rahe hain, main bahut darr gayi hoon", "channel": "chat", "language": "Hindi"}'
```

Response (trimmed):

```json
{
  "case_id": "NH-D5D496A8",
  "svi_metrics": {
    "final_svi_score": 75.94,
    "risk_band": "CRITICAL",
    "components": [{"name": "lexical", "score": 0.66, "weight": 1.0, "contribution": 75.94}]
  },
  "nlp_indicators": {
    "flagged_keywords": ["koi madad nahi", "bahishkar", "dhamki", "akela", "darr"],
    "distress_categories": ["Fear / intimidation / threats", "Social boycott / isolation / displacement"]
  },
  "recommended_interventions": ["Immediate Priority Transfer to Senior Trauma Counselor", "..."]
}
```

---

## 📊 SVI Scoring Model & Risk Bands

| SVI | Band | Color | Protocol |
| :--- | :--- | :--- | :--- |
| 0–24 | **LOW** | 🟢 `#00C853` | Routine grievance intake and logging |
| 25–49 | **MODERATE** | 🟡 `#FFD600` | Counselling intake queue + 24-h welfare check |
| 50–74 | **HIGH** | 🟠 `#FF6D00` | Senior supervisor + Psychological First Aid + legal-aid officer |
| 75–100 | **CRITICAL** | 🔴 `#D50000` | Senior trauma counsellor + district police desk alert + line-tracing |

Fusion (voice): `SVI = 0.30·acoustic + 0.45·emotion + 0.25·lexical`, each component
passed through a monotonic expansion curve, plus severity floors.

---

## 🗂️ Project Structure

```text
├── backend/
│   ├── main.py               # FastAPI routes (audio, text, cases, stats, HITL)
│   ├── svi_engine.py         # SVI fusion, SER, Whisper STT, persistence
│   ├── text_analyzer.py      # Shared narrative analysis (both modalities)
│   ├── lexicon.py            # Multilingual distress lexicon + PII redaction
│   ├── audio_processor.py    # librosa biomarkers (pitch semitones, RMS, F0)
│   ├── case_store.py         # SQLite case persistence (thread-safe)
│   ├── config.py             # Weights, risk bands, intervention protocols
│   ├── schemas.py            # Pydantic v2 API schemas
│   └── tests/                # Text-pipeline regression checks
├── frontend/                 # React 19 + Vite + Tailwind operator dashboard
│   └── src/pages/            # Dashboard · Cases · CaseDetails · LiveAssessment (voice+text)
├── sample_data/              # Multilingual test narratives + audio guidance
├── data/cases.db             # SQLite case store (created at runtime)
└── requirements.txt
```

---

## ⚖️ Ethics & Responsible-AI notes

- Assessments are **decision support for trained operators**, never an
  automatic determination of risk or a clinical diagnosis (enforced in UI copy).
- All narratives are PII-redacted **before** persistence; only redacted text
  is stored or returned.
- Informed consent is a mandatory, recorded step before any assessment runs.
- Severity floors raise scores only for genuine disclosures and always show
  their reasoning to the human reviewer.

## 👥 Stakeholders served

DoSJE · NHAA 14566 operators · State/UT governments · District
administrations · Counsellors & mental-health professionals · Law
enforcement · Rehabilitation & welfare authorities.
