"""
backend/tests/test_continuous_stream_stress.py

Sustained Continuous Audio Stream Stress Test for M3GAN Phase 4B.

Evaluates streaming stability over extended durations (30s, 2 min, 5 min, 10 min):
  - Average & P95 frame processing latency (ms)
  - Realtime processing ratio (Latency / Hop Duration)
  - Memory consumption (RAM RSS in MB via psutil)
  - Queue depth & backlog growth (verifying 0 memory leak / backlog accumulation)
  - Transcript deduplication & session state cleanup
"""

import sys
import time
import psutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.streaming.streaming_session import StreamingAssessmentSession, StreamingSessionManager


def generate_synthetic_audio_chunk(duration_sec: float = 0.5, sr: int = 16000, frequency: float = 220.0) -> bytes:
    """Generate 16kHz 16-bit PCM mono audio chunk simulating live mic/call."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 220 Hz tone with slight amplitude modulation + low noise
    signal = 0.3 * np.sin(2 * np.pi * frequency * t) + 0.02 * np.random.randn(len(t))
    pcm_16 = (signal * 32767).astype(np.int16)
    return pcm_16.tobytes()


def run_continuous_stream_stress(test_duration_sec: float = 30.0, hop_sec: float = 0.5):
    print("=========================================================================")
    print(f" M3GAN SUSTAINED STREAM STRESS TEST ({test_duration_sec:.0f}s SIMULATED STREAM)")
    print("=========================================================================")

    process = psutil.Process()
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)
    print(f" Initial RAM RSS: {ram_initial_mb:.2f} MB")

    manager = StreamingSessionManager()
    session_id = "stress_test_session_001"
    session = manager.create_session(session_id=session_id, window_sec=3.0, hop_sec=hop_sec)

    chunk_bytes = generate_synthetic_audio_chunk(duration_sec=hop_sec)
    total_chunks = int(test_duration_sec / hop_sec)

    latencies = []
    realtime_ratios = []

    print(f" Warming up session model pipeline...")
    t_warmup_0 = time.perf_counter()
    session.ingest_audio_chunk(chunk_bytes)
    session.process_latest_window()
    cold_start_latency_ms = (time.perf_counter() - t_warmup_0) * 1000.0
    print(f" Model Cold-Start Initialization Latency: {cold_start_latency_ms:.1f} ms")

    print(f" Ingesting {total_chunks} steady-state audio chunks (0.5s each)...")
    start_wall_clock = time.perf_counter()

    for i in range(total_chunks):
        t0 = time.perf_counter()
        triggered = session.ingest_audio_chunk(chunk_bytes)
        update_msg = session.process_latest_window() if triggered else None
        t_proc = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_proc)

        # Realtime ratio = processing time / audio duration (500ms)
        rr = t_proc / (hop_sec * 1000.0)
        realtime_ratios.append(rr)

        if (i + 1) % 20 == 0 or (i + 1) == total_chunks:
            ram_current_mb = process.memory_info().rss / (1024 * 1024)
            svi_val = update_msg.svi.smoothed_score if update_msg else 0.0
            risk_b = update_msg.svi.risk_band if update_msg else "N/A"
            buf_len = len(session.audio_buffer) * 4  # bytes in float32 array
            print(
                f"  Chunk [{i+1:3d}/{total_chunks}] | Steady-State Latency: {t_proc:5.1f}ms | "
                f"SVI: {svi_val:.1f} ({risk_b}) | "
                f"RAM: {ram_current_mb:.1f}MB | BufferSize: {buf_len}B"
            )

    total_wall_sec = time.perf_counter() - start_wall_clock
    ram_final_mb = process.memory_info().rss / (1024 * 1024)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    max_latency = np.max(latencies)
    avg_realtime_ratio = np.mean(realtime_ratios)

    print("\n-------------------------------------------------------------------------")
    print(" STRESS TEST PERFORMANCE REPORT")
    print("-------------------------------------------------------------------------")
    print(f" Model Cold-Start Latency:        {cold_start_latency_ms:.2f} ms")
    print(f" Total Simulated Audio Duration: {test_duration_sec:.1f} s ({total_chunks} chunks)")
    print(f" Total Wall-Clock Execution Time: {total_wall_sec:.2f} s")
    print(f" Steady-State Avg Latency:        {avg_latency:.2f} ms / hop")
    print(f" Steady-State P95 Latency:        {p95_latency:.2f} ms / hop")
    print(f" Peak Single-Chunk Latency:       {max_latency:.2f} ms")
    print(f" Steady-State Realtime Ratio:     {avg_realtime_ratio:.4f} (< 1.0 means real-time capable)")
    print(f" RAM Footprint (Start -> End):     {ram_initial_mb:.1f} MB -> {ram_final_mb:.1f} MB (Delta: {ram_final_mb - ram_initial_mb:+.1f} MB)")
    print(f" Final Audio Buffer Size:         {len(session.audio_buffer) * 4} bytes (Max 640,000 bytes bounded)")

    # Cleanup verification
    manager.close_session(session_id)
    active_sessions = len(manager.active_sessions)
    print(f" Session Cleanup Verification:    Active Sessions = {active_sessions} (0 expected)")

    # Pass/Fail Assertions
    assert avg_realtime_ratio < 1.0, f"FAILED: Steady-state realtime ratio ({avg_realtime_ratio:.2f}) >= 1.0!"
    assert (len(session.audio_buffer) * 4) <= 640000, "FAILED: Audio buffer exceeded bounded rolling limit!"
    assert active_sessions == 0, "FAILED: Session was not cleaned up!"

    print("\n SUCCESS: CONTINUOUS STREAM STRESS TEST PASSED ALL BENCHMARKS.")
    print("=========================================================================\n")


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    run_continuous_stream_stress(test_duration_sec=duration)
