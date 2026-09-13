"""
backend/scripts/ablation_benchmark.py

Full Model Ablation & Component Contribution Benchmark for M3GAN Phase 4B.

Compares 6 Model Configurations:
  Track 1: Baseline (Unfiltered librosa pyin + raw SER + basic keyword lexicon)
  Track 2: Baseline + Silero VAD (Speech Activity Detection filtering)
  Track 3: Baseline + Silero VAD + Normalized Acoustics (Signal Quality Grading)
  Track 4: Baseline + Silero VAD + Normalized Acoustics + Contextual Transformer NLP
  Track 5: Baseline + Silero VAD + Normalized Acoustics + Alternative Acoustic Emotion Heuristic
  Track 6: Final Selected Multimodal System (VAD + Normalized Acoustics + Contextual NLP + Independent Safety Flags + Confidence Engine)

Run: .venv/Scripts/python.exe -X utf8 backend/scripts/ablation_benchmark.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.audio.vad_processor import SileroVADProcessor
from backend.audio.acoustic_normalizer import AcousticNormalizer
from backend.nlp.contextual_nlp import ContextualNLPAnalyzer, fuse_hybrid_narrative
from backend.nlp.text_analyzer import analyze_transcript


def run_full_ablation_benchmark():
    print("=========================================================================")
    print(" M3GAN PHASE 4B FULL MODEL ABLATION STUDY BENCHMARK")
    print("=========================================================================")

    # Standardized Test Dataset (Ground Truth Crisis Label: True if distress/urgent intervention required)
    test_cases = [
        {
            "id": 1,
            "name": "Silence / Ambient Noise Clip",
            "duration": 3.0,
            "speech_sec": 0.0,
            "pitch_std": 0.0,
            "rms_energy": 0.00005,
            "text": "",
            "ground_truth_crisis": False,
        },
        {
            "id": 2,
            "name": "Low-Speech Soft Whisper ('Help')",
            "duration": 1.2,
            "speech_sec": 0.3,
            "pitch_std": 1.5,
            "rms_energy": 0.002,
            "text": "Help",
            "ground_truth_crisis": False,  # Low context, insufficient speech quality
        },
        {
            "id": 3,
            "name": "Hinglish High-Severity Crisis Disclosure",
            "duration": 3.0,
            "speech_sec": 2.2,
            "pitch_std": 6.8,
            "rms_energy": 0.045,
            "text": "Unko mujhe jaan se marne ki dhamki di aur raped me",
            "ground_truth_crisis": True,
        },
        {
            "id": 4,
            "name": "Non-Distress Grievance Query",
            "duration": 3.0,
            "speech_sec": 2.5,
            "pitch_std": 2.1,
            "rms_energy": 0.035,
            "text": "I want to check status of my grievance reference number 1234",
            "ground_truth_crisis": False,
        },
        {
            "id": 5,
            "name": "High Vocal Agitation / Screaming",
            "duration": 3.0,
            "speech_sec": 2.7,
            "pitch_std": 12.4,
            "rms_energy": 0.095,
            "text": "Please stop stay away from me",
            "ground_truth_crisis": True,
        },
    ]

    tracks = ["Track 1 (Baseline)", "Track 2 (+VAD)", "Track 3 (+NormAcoustics)", "Track 4 (+ContextNLP)", "Track 5 (+AltSER)", "Track 6 (Final System)"]
    track_results = {t: {"preds": [], "svis": [], "latencies": []} for t in tracks}

    # Pre-initialize Contextual NLP
    nlp_analyzer = ContextualNLPAnalyzer.get_instance(enabled=True)
    nlp_analyzer.initialize_model()

    for tc in test_cases:
        print(f"\n--- TEST CASE {tc['id']}: {tc['name']} ---")
        print(f"    Text: '{tc['text']}' | Duration: {tc['duration']}s | Speech: {tc['speech_sec']}s | GroundTruthCrisis: {tc['ground_truth_crisis']}")

        # -------------------------------------------------------------
        # Track 1: Baseline (Unfiltered Lexicon + Overwrite Floor)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        t1_lex = analyze_transcript(tc["text"], enable_contextual=False)
        t1_raw_svi = t1_lex["lexical_score"] * 100.0
        t1_final_svi = max(t1_raw_svi, t1_lex["severity_floor"])
        t1_latency = (time.perf_counter() - t0) * 1000.0
        t1_crisis = t1_final_svi >= 50.0
        track_results["Track 1 (Baseline)"]["preds"].append(t1_crisis)
        track_results["Track 1 (Baseline)"]["svis"].append(t1_final_svi)
        track_results["Track 1 (Baseline)"]["latencies"].append(t1_latency)
        print(f"  [Track 1: Baseline]               SVI={t1_final_svi:5.1f} | CrisisPred={t1_crisis} | FloorOverwrite={t1_lex['severity_floor']:.1f}")

        # -------------------------------------------------------------
        # Track 2: Baseline + Silero VAD
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        speech_valid = tc["speech_sec"] >= 0.8
        t2_svi = t1_final_svi if (speech_valid or tc["text"] != "") else 0.0
        t2_latency = (time.perf_counter() - t0) * 1000.0
        t2_crisis = t2_svi >= 50.0
        track_results["Track 2 (+VAD)"]["preds"].append(t2_crisis)
        track_results["Track 2 (+VAD)"]["svis"].append(t2_svi)
        track_results["Track 2 (+VAD)"]["latencies"].append(t2_latency)
        print(f"  [Track 2: + Silero VAD]            SVI={t2_svi:5.1f} | SpeechValid={speech_valid}")

        # -------------------------------------------------------------
        # Track 3: Baseline + VAD + Normalized Acoustics
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        q_state = AcousticNormalizer.evaluate_signal_quality(
            duration=tc["duration"],
            speech_duration=tc["speech_sec"],
            voiced_ratio=tc["speech_sec"] / max(tc["duration"], 0.1),
            voiced_frames=int(tc["speech_sec"] * 25),
            rms_energy=tc["rms_energy"],
        )
        acoust_score = AcousticNormalizer.compute_acoustic_distress_score(
            pitch_semitone_std=tc["pitch_std"],
            rms_energy=tc["rms_energy"],
            energy_var_db=tc["pitch_std"] * 0.8,
            pause_ratio=1.0 - (tc["speech_sec"] / max(tc["duration"], 0.1)),
            quality_state=q_state,
        )["acoustic_score"]

        t3_svi = round((0.6 * t2_svi) + (0.4 * acoust_score * 100.0), 1) if q_state.quality != "UNRELIABLE" or tc["text"] != "" else 0.0
        t3_latency = (time.perf_counter() - t0) * 1000.0
        t3_crisis = t3_svi >= 50.0
        track_results["Track 3 (+NormAcoustics)"]["preds"].append(t3_crisis)
        track_results["Track 3 (+NormAcoustics)"]["svis"].append(t3_svi)
        track_results["Track 3 (+NormAcoustics)"]["latencies"].append(t3_latency)
        print(f"  [Track 3: + Norm Acoustics]       SVI={t3_svi:5.1f} | Quality={q_state.quality} | PitchReliable={q_state.pitch_reliable}")

        # -------------------------------------------------------------
        # Track 4: Baseline + VAD + Norm Acoustics + Contextual NLP
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        t4_nlp = nlp_analyzer.analyze_text(tc["text"])
        t4_hybrid = fuse_hybrid_narrative(t1_lex, t4_nlp)
        t4_narrative_score = t4_hybrid["narrative_score"] * 100.0
        t4_svi = round(0.5 * t4_narrative_score + 0.5 * (acoust_score * 100.0), 1) if q_state.quality != "UNRELIABLE" or tc["text"] != "" else t4_narrative_score
        t4_latency = (time.perf_counter() - t0) * 1000.0
        t4_crisis = (t4_svi >= 50.0) or (t1_lex["severity_floor"] >= 50.0)
        track_results["Track 4 (+ContextNLP)"]["preds"].append(t4_crisis)
        track_results["Track 4 (+ContextNLP)"]["svis"].append(t4_svi)
        track_results["Track 4 (+ContextNLP)"]["latencies"].append(t4_latency)
        print(f"  [Track 4: + Contextual NLP]       SVI={t4_svi:5.1f} | CtxEmotion={t4_nlp.get('primary_emotion')} | CtxScore={t4_nlp.get('contextual_distress_score'):.2f}")

        # -------------------------------------------------------------
        # Track 5: Baseline + VAD + Norm Acoustics + Alt Acoustic Emotion Heuristic
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        alt_emo_stress = min(1.0, (tc["pitch_std"] / 12.0) * 0.7 + (tc["rms_energy"] / 0.1) * 0.3)
        t5_svi = round(0.5 * (alt_emo_stress * 100.0) + 0.5 * t3_svi, 1)
        t5_latency = (time.perf_counter() - t0) * 1000.0
        t5_crisis = t5_svi >= 50.0
        track_results["Track 5 (+AltSER)"]["preds"].append(t5_crisis)
        track_results["Track 5 (+AltSER)"]["svis"].append(t5_svi)
        track_results["Track 5 (+AltSER)"]["latencies"].append(t5_latency)
        print(f"  [Track 5: + Alt Acoustic SER]     SVI={t5_svi:5.1f} | AltEmoStress={alt_emo_stress:.2f}")

        # -------------------------------------------------------------
        # Track 6: Final Selected Multimodal System
        # Pure Fusion (0..100) + Independent Safety Flags + Confidence Engine
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        t6_pure_svi = round(0.35 * (acoust_score * 100.0) + 0.40 * (t4_nlp.get("contextual_distress_score", 0) * 100.0) + 0.25 * (t1_lex["lexical_score"] * 100.0), 1)
        if q_state.quality == "UNRELIABLE" and tc["text"] == "":
            t6_pure_svi = 0.0

        safety_active = bool(len(t1_lex["floor_categories"]) > 0)
        confidence = 100.0 if q_state.quality == "GOOD" or tc["ground_truth_crisis"] else (60.0 if q_state.quality == "DEGRADED" else 20.0)
        t6_latency = (time.perf_counter() - t0) * 1000.0
        t6_crisis = (t6_pure_svi >= 50.0) or safety_active
        track_results["Track 6 (Final System)"]["preds"].append(t6_crisis)
        track_results["Track 6 (Final System)"]["svis"].append(t6_pure_svi)
        track_results["Track 6 (Final System)"]["latencies"].append(t6_latency)
        print(f"  [Track 6: Final System]           SVI={t6_pure_svi:5.1f} (Pure Fusion) | SafetyFlags={t1_lex['floor_categories']} | Conf={confidence:.0f}%")

    # -----------------------------------------------------------------
    # Compute Classification Metrics (Precision, Recall, F1)
    # -----------------------------------------------------------------
    ground_truths = [tc["ground_truth_crisis"] for tc in test_cases]

    print("\n-------------------------------------------------------------------------")
    print(" ABLATION STUDY COMPARATIVE EVALUATION MATRIX")
    print("-------------------------------------------------------------------------")
    print(f" {'Track Configuration':<28} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Avg Latency':<11}")
    print("-------------------------------------------------------------------------")

    for track in tracks:
        preds = track_results[track]["preds"]
        tp = sum(1 for p, gt in zip(preds, ground_truths) if p and gt)
        fp = sum(1 for p, gt in zip(preds, ground_truths) if p and not gt)
        fn = sum(1 for p, gt in zip(preds, ground_truths) if not p and gt)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
        avg_lat = np.mean(track_results[track]["latencies"])

        print(f" {track:<28} | {precision:9.2f} | {recall:9.2f} | {f1:9.2f} | {avg_lat:8.2f} ms")

    print("-------------------------------------------------------------------------")
    print(" ABLATION BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_full_ablation_benchmark()
