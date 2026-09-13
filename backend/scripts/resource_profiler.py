"""
backend/scripts/resource_profiler.py

M3GAN AI Model Resource Profiler & Latency Benchmark.

Measures actual local execution metrics across each AI component:
  1. Silero VAD Processor
  2. Wav2Vec2 Speech Emotion Recognition (SER)
  3. Faster-Whisper ASR
  4. Contextual DistilRoBERTa NLP Transformer
  5. SVI Fusion Engine
"""

import sys
import time
import psutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.audio.vad_processor import SileroVADProcessor
from backend.nlp.contextual_nlp import ContextualNLPAnalyzer
from backend.audio.audio_processor import AudioProcessor


def profile_models():
    print("=========================================================================")
    print(" M3GAN LOCAL MODEL RESOURCE PROFILER & LATENCY BENCHMARK")
    print("=========================================================================")

    process = psutil.Process()
    base_ram = process.memory_info().rss / (1024 * 1024)
    print(f" Initial Process Baseline RAM: {base_ram:.2f} MB\n")

    results = []

    # 1. Silero VAD
    t0 = time.perf_counter()
    vad = SileroVADProcessor()
    load_time_vad = (time.perf_counter() - t0) * 1000.0

    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3s audio
    t_inf = time.perf_counter()
    vad_res = vad.process_audio(dummy_audio, 16000)
    inf_time_vad = (time.perf_counter() - t_inf) * 1000.0
    ram_after_vad = process.memory_info().rss / (1024 * 1024)

    results.append({
        "component": "Silero VAD",
        "load_ms": load_time_vad,
        "inf_ms": inf_time_vad,
        "ram_mb": ram_after_vad - base_ram,
        "engine": vad_res.vad_engine,
    })

    # 2. Contextual DistilRoBERTa NLP
    t0 = time.perf_counter()
    nlp = ContextualNLPAnalyzer.get_instance(enabled=True)
    nlp.initialize_model()
    load_time_nlp = (time.perf_counter() - t0) * 1000.0

    t_inf = time.perf_counter()
    nlp_res = nlp.analyze_text("They threatened to hurt my family and I am terrified")
    inf_time_nlp = (time.perf_counter() - t_inf) * 1000.0
    ram_after_nlp = process.memory_info().rss / (1024 * 1024)

    results.append({
        "component": "DistilRoBERTa NLP",
        "load_ms": load_time_nlp,
        "inf_ms": inf_time_nlp,
        "ram_mb": ram_after_nlp - ram_after_vad,
        "engine": nlp_res["model_name"],
    })

    # 3. Audio Processor & Acoustic Normalizer
    t0 = time.perf_counter()
    audio_proc = AudioProcessor()
    load_time_ap = (time.perf_counter() - t0) * 1000.0

    t_inf = time.perf_counter()
    biomarkers = audio_proc.extract_voice_biomarkers(dummy_audio)
    inf_time_ap = (time.perf_counter() - t_inf) * 1000.0
    ram_after_ap = process.memory_info().rss / (1024 * 1024)

    results.append({
        "component": "Acoustic Normalizer",
        "load_ms": load_time_ap,
        "inf_ms": inf_time_ap,
        "ram_mb": ram_after_ap - ram_after_nlp,
        "engine": "Librosa + Signal Quality Classifier",
    })

    total_ram = process.memory_info().rss / (1024 * 1024)

    print("\n-------------------------------------------------------------------------")
    print(" RESOURCE & LATENCY BENCHMARK TABLE")
    print("-------------------------------------------------------------------------")
    print(f" {'Component':<22} | {'Load (ms)':<10} | {'Inference (ms)':<15} | {'Delta RAM (MB)':<14}")
    print("-------------------------------------------------------------------------")

    for r in results:
        print(f" {r['component']:<22} | {r['load_ms']:10.1f} | {r['inf_ms']:15.2f} | {r['ram_mb']:14.1f}")

    print("-------------------------------------------------------------------------")
    print(f" Total Process RAM Footprint: {total_ram:.1f} MB (Baseline Delta: {total_ram - base_ram:+.1f} MB)")
    print("=========================================================================\n")


if __name__ == "__main__":
    profile_models()
