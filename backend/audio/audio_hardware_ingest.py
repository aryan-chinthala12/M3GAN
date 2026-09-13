"""
backend/audio_hardware_ingest.py

Phase 5D Phone / USB / External Audio Line Ingestion Module for M3GAN.

Responsibilities:
  - Discovers system audio input devices (microphones, USB interfaces, virtual audio cables).
  - Manages real-time soundcard audio capture using sounddevice.
  - Performs mono downmixing, 16kHz resampling (via scipy.signal), and digital gain adjustment.
  - Forwards 16kHz PCM float32 chunks into active StreamingAssessmentSession instances.
  - Tracks telemetry: captured_frames, dropped_chunks, input_peak, output_peak, clipping_count, last_error.
  - Thread-safe start, stop, reconnection, and status monitoring.
"""

import time
import math
import logging
import threading
import numpy as np
import scipy.signal
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception as e:
    logger.warning(f"sounddevice unavailable: {e}")
    SOUNDDEVICE_AVAILABLE = False


def list_input_devices() -> List[Dict[str, Any]]:
    """Query host system for valid audio input devices (filters out 0-input channel devices)."""
    if not SOUNDDEVICE_AVAILABLE:
        return []

    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0] if (sd.default and sd.default.device) else None

        input_list = []
        for idx, dev in enumerate(devices):
            max_in = dev.get("max_input_channels", 0)
            if max_in > 0:
                is_default = (idx == default_in)
                input_list.append({
                    "device_index": idx,
                    "name": dev.get("name", f"Device #{idx}"),
                    "channels": max_in,
                    "default_sample_rate": int(dev.get("default_samplerate", 44100)),
                    "is_default": is_default,
                })
        return input_list
    except Exception as e:
        logger.error(f"Error querying audio input devices: {e}")
        return []


class HardwareAudioIngestManager:
    """Thread-safe manager for hardware / line-input audio ingestion streams with full telemetry."""
    _instance = None

    def __init__(self):
        self.active_streams: Dict[str, dict] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = HardwareAudioIngestManager()
        return cls._instance

    def start_ingestion(
        self,
        session_id: str,
        session_obj: Any,
        device_index: Optional[int] = None,
        chunk_sec: float = 0.5,
        target_sr: int = 16000,
        gain_db: float = 0.0,
        websocket_broadcaster: Optional[Any] = None,
    ) -> bool:
        """
        Start recording from specified hardware input device into a StreamingAssessmentSession.
        If a stream already exists for session_id, it is safely stopped first (reconnection guard).
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice library is not installed or supported on this system.")

        with self._lock:
            # Reconnection safety: stop any existing stream for this session_id first
            if session_id in self.active_streams:
                logger.info(f"Reclosing existing stream for session '{session_id}' before reconnecting.")
                self._stop_stream_unlocked(session_id)

            devices = list_input_devices()
            if not devices:
                raise RuntimeError("No hardware audio input devices detected on host system.")

            if device_index is None:
                default_dev = next((d for d in devices if d["is_default"]), devices[0])
                device_index = default_dev["device_index"]
                device_name = default_dev["name"]
                native_sr = default_dev["default_sample_rate"]
                channels = min(2, default_dev["channels"])
            else:
                dev_info = next((d for d in devices if d["device_index"] == device_index), None)
                if not dev_info:
                    raise ValueError(f"Invalid device index: {device_index}")
                device_name = dev_info["name"]
                native_sr = dev_info["default_sample_rate"]
                channels = min(2, dev_info["channels"])

            gain_factor = 10.0 ** (gain_db / 20.0)
            target_chunk_samples = int(chunk_sec * target_sr)

            buffer_lock = threading.Lock()
            pcm_accumulator = np.array([], dtype=np.float32)

            telemetry = {
                "captured_frames": 0,
                "dropped_chunks": 0,
                "clipping_count": 0,
                "input_peak": 0.0,
                "output_peak": 0.0,
                "last_error": None,
                "state": "RECORDING",
            }

            def audio_callback(indata, frames, time_info, status):
                nonlocal pcm_accumulator
                try:
                    if status:
                        telemetry["last_error"] = str(status)
                        logger.warning(f"Audio stream status ({session_id}): {status}")
                        if "overflow" in str(status).lower() or "underflow" in str(status).lower():
                            telemetry["dropped_chunks"] += 1

                    if indata is None or len(indata) == 0:
                        return

                    telemetry["captured_frames"] += len(indata)
                    in_peak = float(np.max(np.abs(indata)))
                    telemetry["input_peak"] = max(telemetry["input_peak"], in_peak)

                    # 1. Downmix to mono if multi-channel
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        mono = np.mean(indata, axis=1)
                    else:
                        mono = indata.flatten()

                    mono_float = np.nan_to_num(mono.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

                    # 2. Resample to 16kHz if native sample rate differs
                    if native_sr != target_sr and len(mono_float) > 0:
                        num_out_samples = int(round(len(mono_float) * target_sr / native_sr))
                        if num_out_samples > 0:
                            resampled = scipy.signal.resample(mono_float, num_out_samples)
                        else:
                            resampled = mono_float
                    else:
                        resampled = mono_float

                    resampled = np.nan_to_num(resampled.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

                    # 3. Apply optional configured gain
                    if gain_db != 0.0:
                        resampled = resampled * gain_factor

                    # 4. Telemetry peak & clipping guard
                    out_peak = float(np.max(np.abs(resampled))) if len(resampled) > 0 else 0.0
                    telemetry["output_peak"] = max(telemetry["output_peak"], out_peak)

                    clips = int(np.sum(np.abs(resampled) >= 0.999))
                    telemetry["clipping_count"] += clips

                    # Clip amplitude [-1.0, 1.0]
                    resampled = np.clip(resampled, -1.0, 1.0)

                    with buffer_lock:
                        pcm_accumulator = np.concatenate([pcm_accumulator, resampled])

                        # Process accumulated chunk_sec (e.g. 0.5s) chunks
                        while len(pcm_accumulator) >= target_chunk_samples:
                            chunk = pcm_accumulator[:target_chunk_samples]
                            pcm_accumulator = pcm_accumulator[target_chunk_samples:]

                            try:
                                triggered = session_obj.ingest_audio_chunk(chunk)
                                if triggered:
                                    msg = session_obj.process_latest_window()
                                    if msg and websocket_broadcaster:
                                        threading.Thread(
                                            target=websocket_broadcaster,
                                            args=(msg.dict(),),
                                            daemon=True,
                                        ).start()
                            except Exception as ex:
                                telemetry["last_error"] = str(ex)
                                logger.error(f"Error in hardware audio chunk ingestion ({session_id}): {ex}")
                except Exception as cb_ex:
                    telemetry["last_error"] = str(cb_ex)
                    telemetry["state"] = "ERROR"
                    logger.error(f"Unhandled callback exception in audio ingestion stream ({session_id}): {cb_ex}")

            try:
                stream = sd.InputStream(
                    device=device_index,
                    samplerate=native_sr,
                    channels=channels,
                    dtype="float32",
                    callback=audio_callback,
                    blocksize=int(native_sr * 0.1),  # 100ms audio blocks
                )
                stream.start()
            except Exception as open_ex:
                telemetry["state"] = "ERROR"
                telemetry["last_error"] = str(open_ex)
                raise RuntimeError(f"Could not open audio input device #{device_index}: {open_ex}")

            self.active_streams[session_id] = {
                "session_id": session_id,
                "device_index": device_index,
                "device_name": device_name,
                "native_sr": native_sr,
                "target_sr": target_sr,
                "channels": channels,
                "gain_db": gain_db,
                "stream": stream,
                "telemetry": telemetry,
                "started_at": time.time(),
            }
            logger.info(
                f"Hardware audio ingestion started for session '{session_id}' "
                f"(Device #{device_index} '{device_name}', {native_sr}Hz -> {target_sr}Hz)."
            )
            return True

    def _stop_stream_unlocked(self, session_id: str) -> bool:
        """Internal helper to stop a stream when lock is already held."""
        if session_id not in self.active_streams:
            return False

        stream_info = self.active_streams.pop(session_id)
        try:
            stream = stream_info["stream"]
            stream.stop()
            stream.close()
            stream_info["telemetry"]["state"] = "STOPPED"
            logger.info(f"Stopped hardware audio ingestion for session '{session_id}'.")
            return True
        except Exception as e:
            logger.error(f"Error stopping hardware audio stream '{session_id}': {e}")
            return False

    def stop_ingestion(self, session_id: str) -> bool:
        """Stop active hardware audio recording stream for a session."""
        with self._lock:
            return self._stop_stream_unlocked(session_id)

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Return status and full telemetry of hardware ingestion for session."""
        with self._lock:
            if session_id in self.active_streams:
                info = self.active_streams[session_id]
                duration = round(time.time() - info["started_at"], 1)
                telem = info["telemetry"]
                return {
                    "active": True,
                    "session_id": session_id,
                    "device_index": info["device_index"],
                    "device_name": info["device_name"],
                    "duration_sec": duration,
                    "native_sample_rate": info["native_sr"],
                    "target_sample_rate": info["target_sr"],
                    "channels": info["channels"],
                    "gain_db": info["gain_db"],
                    "state": telem["state"],
                    "captured_frames": telem["captured_frames"],
                    "dropped_chunks": telem["dropped_chunks"],
                    "clipping_count": telem["clipping_count"],
                    "input_peak": round(telem["input_peak"], 4),
                    "output_peak": round(telem["output_peak"], 4),
                    "last_error": telem["last_error"],
                    "connection_health": "GOOD" if telem["last_error"] is None else "WARNING",
                }
            return {
                "active": False,
                "session_id": session_id,
                "state": "DISCONNECTED",
                "connection_health": "OFFLINE",
            }
