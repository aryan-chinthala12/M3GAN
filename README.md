# M3GAN

## NHAA Real-Time Stress & Trauma Assessment Engine (SVI-Engine)

An AI-enabled, zero-latency acoustic processing engine developed for the **National Helpline Against Atrocities (NHAA - 14566)** under the **Department of Social Justice and Empowerment (DoSJE)**, Government of India.

The system extracts real-time vocal bio-markers—pitch variance, volume tremors, and silent pause ratios—to compute a dynamic **Stress Vulnerability Index (SVI)** (0–100) and automatically route emergency responses for distress callers.

---

## 📌 Problem Statement & Context

Victims of atrocities interacting through helpline 14566, integrated portals, chatbots, or mobile applications often experience severe trauma, fear, and psychological shock. Standard text or manual triage causes delays during critical emergency windows. 

This engine solves this by providing **instant, language-agnostic emotional triage** at the acoustic signal level—allowing call center agents and emergency responders to immediately identify high-risk individuals and prioritize intervention.

---

## ⚡ Core Features

* **Zero-Latency Acoustic Engine:** Operates directly on raw audio signals (`16kHz` mono) without Speech-to-Text transcription delays.
* **Language-Agnostic Triage:** Evaluates vocal bio-markers (pitch instability, volume tremors, and hesitation pauses) that indicate distress regardless of the spoken language or regional dialect.
* **Stress Vulnerability Index (SVI):** Mathematical scoring model mapping vocal strain to a `0–100` scale.
* **Automated Risk Categorization:** Categorizes callers into **Low**, **Moderate**, **High**, and **Critical** risk tiers.
* **Actionable Protocol Triggers:** Dynamically generates recommended actions ranging from standard grievance logging to high-priority police dispatch and trauma counselor transfers.
* **Microservice Architecture:** Modular FastAPI backend paired with an intuitive Streamlit UI dashboard.

---

## 📁 Project Directory Structure

```text
nhaa-svi-engine/
├── backend/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Feature weights, thresholds, and normalizations
│   ├── audio_processor.py   # Librosa DSP feature extraction routines
│   ├── svi_engine.py        # SVI scoring algorithms and protocol mapping
│   └── main.py              # FastAPI endpoints and gateway routes
├── frontend/
│   └── app.py               # Streamlit live agent dashboard
├── sample_data/
│   ├── calm_sample.wav      # Baseline audio sample (Low Risk)
│   └── panic_sample.wav     # High-distress audio sample (Critical Risk)
├── .gitignore               # System & python cache exclusions
├── requirements.txt         # Python dependency manifest
└── README.md                # Project documentation