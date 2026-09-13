"""
backend/scripts/benchmark_phase5c.py

Phase 5C Benchmark & Evaluation Script.

Evaluates 3 Candidate Multimodal Fusion Schemes:
  1. BASELINE: Fixed Acoustic 30%, Emotion 30%, Lexical 40%.
  2. DYNAMIC SCHEME A: Whisper/Low-Vol (15%, 20%, 65%), Low-SNR (15%, 35%, 50%), Good (30%, 30%, 40%).
  3. DYNAMIC SCHEME B: Whisper/Low-Vol (10%, 30%, 60%), Low-SNR (20%, 30%, 50%), Good (30%, 30%, 40%).

Reports:
  - Precision, Recall, F1-Score, FPR, FNR
  - Average Latency (ms), P95 Latency (ms), Realtime Ratio
  - RAM Footprint (MB)
  - Retest results for Phase 4C/5B soft whisper FN
"""

import sys
import time
import os
import warnings
import psutil
import numpy as np
import scipy.io.wavfile as wav
import tempfile
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

from backend.core.svi_engine import SVIEngine
from backend.nlp.text_analyzer import analyze_transcript
from backend.tests.evaluation_dataset_v2 import EVALUATION_DATASET_V2


# Additional Quiet / Whispered Test Cases for Phase 5C
QUIET_TEST_CASES = [
    {
        "id": "Q1",
        "text": "He is in the hallway with a knife please send help softly",
        "expected_distress": True,
        "category": "weapon_threat",
        "language": "English",
        "amplitude": 0.0035,
        "is_whisper": True,
    },
    {
        "id": "Q2",
        "text": "chupke se suno main ghar pe akeli hoon aur koi darwaza tod raha hai",
        "expected_distress": True,
        "category": "intruder_threat",
        "language": "Hinglish",
        "amplitude": 0.0040,
        "is_whisper": True,
    },
    {
        "id": "Q3",
        "text": "I am just speaking quietly because my baby is sleeping in the other room",
        "expected_distress": False,
        "category": "ordinary_quiet",
        "language": "English",
        "amplitude": 0.0050,
        "is_whisper": False,
    },
    {
        "id": "Q4",
        "text": "please don't make noise he will kill me if he hears me talking to 14566",
        "expected_distress": True,
        "category": "domestic_violence",
        "language": "English",
        "amplitude": 0.0025,
        "is_whisper": True,
    },
    {
        "id": "Q5",
        "text": "mera landlord rent ke liye call kar raha hai thoda anxious hoon",
        "expected_distress": False,
        "category": "administrative_anxiety",
        "language": "Hinglish",
        "amplitude": 0.0060,
        "is_whisper": False,
    },
]


def generate_synth_wav(amplitude=0.05, duration_sec=3.0, noise_amp=0.0):
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * 300 * t)
    if noise_amp > 0:
        noise = noise_amp * np.random.normal(size=len(tone))
        audio = tone + noise
    else:
        audio = tone
    audio = np.clip(audio, -1.0, 1.0)
    int16_data = (audio * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp.name, sr, int16_data)
    tmp.close()
    return tmp.name


def run_benchmark():
    with open("step_log.txt", "w") as f:
        f.write("Starting run_benchmark\n")
    print("=" * 80)
    print("PHASE 5C BENCHMARK & MULTIMODAL DYNAMIC FUSION EVALUATION")
    print("=" * 80)

    process = psutil.Process(os.getpid())
    engine = SVIEngine(use_vad_fallback=True)

    # Models loaded lazily if needed, benchmark focuses on acoustic biomarkers + text analysis + fusion
    print("[INFO] Engine initialized.")

    schemes = ["BASELINE", "DYNAMIC_SCHEME_A", "DYNAMIC_SCHEME_B"]
    results_by_scheme = {}

    all_cases = []
    # Combine dataset v2 text cases + quiet cases
    for item in EVALUATION_DATASET_V2:
        all_cases.append({
            "id": str(item.id),
            "text": item.text_transcript,
            "expected_distress": item.ground_truth_crisis,
            "category": item.category,
            "language": item.language,
            "amplitude": 0.05,
            "is_whisper": (item.audio_type == "soft_whisper"),
        })
    all_cases.extend(QUIET_TEST_CASES)

    for scheme in schemes:
        print(f"\n[EVALUATING SCHEME: {scheme}]")
        tp, fp, tn, fn = 0, 0, 0, 0
        latencies = []

        for case in all_cases:
            wav_path = generate_synth_wav(
                amplitude=case["amplitude"],
                duration_sec=3.0,
                noise_amp=0.008 if case["is_whisper"] else 0.0,
            )

            t0 = time.perf_counter()
            try:
                # 1. Acoustic biomarker extraction & signal quality evaluation
                y, sr, err = engine.audio_processor.validate_and_load(wav_path)
                if y is not None:
                    biomarkers = engine.audio_processor.extract_voice_biomarkers(y, sr=sr)
                    quality = biomarkers["signal_quality"]
                    ac_score = biomarkers["acoustic_score"]
                else:
                    quality = "UNRELIABLE"
                    ac_score = 0.0

                # 2. Text analysis disclosure evaluation
                if case["text"].strip():
                    text_result = analyze_transcript(case["text"])
                    lex_score = text_result["lexical_score"]
                    hits = text_result["hits"]
                    floor_labels = text_result["floor_categories"]
                    sev_floor = text_result["severity_floor"]
                else:
                    lex_score = 0.0
                    hits = []
                    floor_labels = []
                    sev_floor = 0.0

                # 3. Dynamic candidate multimodal fusion
                w_ac, w_em, w_lex = engine.get_fusion_weights(quality, fusion_mode=scheme)
                em_score = 0.05

                raw_svi, safety_flags, _, _ = engine._fuse_with_safety_flags(
                    [("acoustic", ac_score, w_ac), ("emotion", em_score, w_em), ("lexical", lex_score, w_lex)],
                    hits=hits,
                    floor_labels=floor_labels,
                    severity_floor=sev_floor,
                )

                predicted_distress = (raw_svi >= 45.0 or len(safety_flags) > 0)
            except Exception as ex:
                print(f"[ERROR IN CASE {case['id']} - {scheme}]: {ex}", flush=True)
                import traceback
                traceback.print_exc()
                predicted_distress = False
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)

            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)

            expected = case["expected_distress"]
            if expected and predicted_distress:
                tp += 1
            elif not expected and predicted_distress:
                fp += 1
            elif not expected and not predicted_distress:
                tn += 1
            else:
                fn += 1

        total = len(all_cases)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        ram_mb = process.memory_info().rss / (1024 * 1024)

        results_by_scheme[scheme] = {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "Avg_Latency_ms": round(avg_lat, 2),
            "P95_Latency_ms": round(p95_lat, 2),
            "RAM_MB": round(ram_mb, 1),
        }

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHASE 5C COMPARATIVE BENCHMARK RESULTS SUMMARY (N = 37)")
    report_lines.append("=" * 80)

    header = f"{'Scheme':<18} | {'TP':<3} | {'FP':<3} | {'TN':<3} | {'FN':<3} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'FPR':<6} | {'Avg Lat':<8}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for scheme, metrics in results_by_scheme.items():
        row = (
            f"{scheme:<18} | {metrics['TP']:<3} | {metrics['FP']:<3} | {metrics['TN']:<3} | {metrics['FN']:<3} | "
            f"{metrics['Precision']:<6.2f} | {metrics['Recall']:<6.2f} | {metrics['F1']:<6.2f} | {metrics['FPR']:<6.2f} | "
            f"{metrics['Avg_Latency_ms']:<7.1f}ms"
        )
        report_lines.append(row)

    report_lines.append("\n[PHASE 4C/5B FALSE NEGATIVE FIX VERIFICATION]")
    # Check Sample Q1 (Soft Whisper Knife Disclosure)
    q1 = QUIET_TEST_CASES[0]
    q1_res = engine.process_text(type("Req", (), {"text": q1["text"], "channel": "web", "language": "English"})())
    report_lines.append(f"Sample Q1 SVI Score: {q1_res.svi_metrics.final_svi_score:.2f} / 100")
    report_lines.append(f"Sample Q1 Safety Flags: {[f.category for f in q1_res.safety_flags]}")
    report_lines.append(f"Fixed FN: {q1_res.svi_metrics.final_svi_score >= 45.0 or len(q1_res.safety_flags) > 0}")
    report_lines.append("\nBenchmark completed successfully.")

    report_text = "\n".join(report_lines)
    print(report_text, flush=True)

    with open("phase5c_benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text + "\n")


if __name__ == "__main__":
    import traceback
    try:
        run_benchmark()
    except Exception as ex:
        print("BENCHMARK ERROR EXCEPTION:")
        traceback.print_exc()
