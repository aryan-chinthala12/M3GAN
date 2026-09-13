"""
backend/vad_processor.py

Voice Activity Detection (VAD) processor using Silero VAD with fallback.

Responsibilities:
  - Classifies audio frames into speech vs. silence/unvoiced.
  - Extracts speech duration, voiced ratio, pause count, total pause duration,
    and mean pause duration.
  - Benchmarks execution latency on local hardware.
  - Fallback to energy/spectral-flatness VAD when Silero model is offline.
"""

import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class VADResult:
    speech_duration: float        # Total seconds of voiced speech
    total_duration: float         # Total audio duration in seconds
    voiced_ratio: float           # speech_duration / total_duration (0..1)
    pause_count: int              # Number of silence/pause segments
    total_pause_duration: float   # Total seconds of pause/silence
    mean_pause_duration: float    # Average duration per pause in seconds
    pause_ratio: float            # total_pause_duration / total_duration (0..1)
    speech_segments: List[Tuple[float, float]]  # List of (start_sec, end_sec)
    has_sufficient_speech: bool   # True if speech_duration >= 0.8s and voiced_ratio >= 0.10
    execution_time_ms: float      # Processing latency in milliseconds
    vad_engine: str               # "silero" or "energy_fallback"


class SileroVADProcessor:
    """Silero VAD wrapper with automatic fallback and latency tracking."""

    def __init__(self, force_fallback: bool = False):
        self.model = None
        self.get_speech_timestamps = None
        self.vad_engine = "none"
        self._load_failed = False

        if not force_fallback:
            self._init_silero()

    def _init_silero(self):
        try:
            # Attempt loading Silero VAD via torch.hub (cached if downloaded)
            torch.set_num_threads(1)
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.model = model
            self.get_speech_timestamps = utils[0]
            self.vad_engine = "silero"
            print("[INFO] Silero VAD initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Silero VAD load failed ({e}); using energy fallback VAD.")
            self._load_failed = True
            self.vad_engine = "energy_fallback"

    def process_audio(
        self,
        y: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.4,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
    ) -> VADResult:
        """
        Process mono float32 audio numpy array at sample_rate (default 16kHz).
        Returns structured VADResult with timing metrics and speech segments.
        """
        t0 = time.perf_counter()

        if y is None or len(y) == 0:
            return VADResult(
                speech_duration=0.0,
                total_duration=0.0,
                voiced_ratio=0.0,
                pause_count=0,
                total_pause_duration=0.0,
                mean_pause_duration=0.0,
                pause_ratio=0.0,
                speech_segments=[],
                has_sufficient_speech=False,
                execution_time_ms=0.0,
                vad_engine=self.vad_engine,
            )

        # Ensure float32 array
        if y.dtype != np.float32:
            y = y.astype(np.float32)

        total_duration = float(len(y) / sample_rate)

        speech_segments = []

        if self.vad_engine == "silero" and self.model is not None:
            try:
                wav_tensor = torch.from_numpy(y)
                timestamps = self.get_speech_timestamps(
                    wav_tensor,
                    self.model,
                    sampling_rate=sample_rate,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                )
                for ts in timestamps:
                    start_sec = round(ts["start"] / sample_rate, 3)
                    end_sec = round(ts["end"] / sample_rate, 3)
                    speech_segments.append((start_sec, end_sec))
            except Exception as ex:
                print(f"[WARNING] Silero VAD execution error ({ex}), falling back.")
                speech_segments = self._energy_fallback_vad(y, sample_rate=sample_rate)
                engine_used = "energy_fallback"
            else:
                engine_used = "silero"
        else:
            speech_segments = self._energy_fallback_vad(y, sample_rate=sample_rate)
            engine_used = "energy_fallback"

        # Calculate durations and pause metrics
        speech_duration = float(sum(end - start for start, end in speech_segments))
        speech_duration = min(speech_duration, total_duration)

        voiced_ratio = float(speech_duration / total_duration) if total_duration > 0 else 0.0

        # Calculate silence/pause intervals
        pause_intervals = []
        last_end = 0.0

        for start, end in speech_segments:
            if start > last_end + 0.1:  # Pauses > 100ms counted
                pause_intervals.append(start - last_end)
            last_end = end

        if total_duration > last_end + 0.1:
            pause_intervals.append(total_duration - last_end)

        pause_count = len(pause_intervals)
        total_pause_duration = float(sum(pause_intervals)) if pause_intervals else 0.0
        mean_pause_duration = (
            float(total_pause_duration / pause_count) if pause_count > 0 else 0.0
        )
        pause_ratio = (
            float(total_pause_duration / total_duration) if total_duration > 0 else 0.0
        )

        has_sufficient_speech = bool(speech_duration >= 0.8 and voiced_ratio >= 0.10)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return VADResult(
            speech_duration=round(speech_duration, 3),
            total_duration=round(total_duration, 3),
            voiced_ratio=round(voiced_ratio, 4),
            pause_count=pause_count,
            total_pause_duration=round(total_pause_duration, 3),
            mean_pause_duration=round(mean_pause_duration, 3),
            pause_ratio=round(pause_ratio, 4),
            speech_segments=speech_segments,
            has_sufficient_speech=has_sufficient_speech,
            execution_time_ms=latency_ms,
            vad_engine=engine_used,
        )

    def _energy_fallback_vad(
        self, y: np.ndarray, sample_rate: int = 16000, frame_length_ms: int = 30
    ) -> List[Tuple[float, float]]:
        """Short-time energy & RMS threshold VAD fallback."""
        frame_len = max(1, int(sample_rate * (frame_length_ms / 1000.0)))
        num_frames = len(y) // frame_len

        if num_frames == 0:
            return []

        frames = np.array_split(y[: num_frames * frame_len], num_frames)
        rms_values = np.array([np.sqrt(np.mean(f**2)) for f in frames])

        # Dynamic threshold based on median noise floor
        noise_floor = np.percentile(rms_values, 20)
        speech_thresh = max(noise_floor * 2.5, 0.005)

        is_speech = rms_values > speech_thresh

        # Merge contiguous speech frames into segments
        segments = []
        in_speech = False
        start_idx = 0

        for i, val in enumerate(is_speech):
            if val and not in_speech:
                in_speech = True
                start_idx = i
            elif not val and in_speech:
                in_speech = False
                segments.append(
                    (
                        round(start_idx * frame_length_ms / 1000.0, 3),
                        round(i * frame_length_ms / 1000.0, 3),
                    )
                )

        if in_speech:
            segments.append(
                (
                    round(start_idx * frame_length_ms / 1000.0, 3),
                    round(len(is_speech) * frame_length_ms / 1000.0, 3),
                )
            )

        return segments
