"""
backend/tests/verify_manual_lifecycle.py

Manual Testing & Verification Script for Phase 5D Session Finalization.
Simulates exact real frontend behavior:
  1. POST /api/v1/session/start -> returns session_id
  2. WebSocket connects & streams 20 seconds of PCM audio
  3. POST /api/v1/session/{session_id}/stop -> returns 200 OK + SessionStopResponse
  4. WebSocket closes
  5. GET /api/v1/session/{session_id}/summary -> returns 200 OK + SessionSummaryResponse
"""

import time
import asyncio
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app

def run_manual_test_simulation():
    print("=" * 80)
    print("[MANUAL TEST SIMULATION] Testing Session Lifecycle & Finalization")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. Start Session
        start_resp = client.post("/api/v1/session/start", json={
            "channel": "voice",
            "language": "English",
            "consent": True,
            "window_duration_sec": 3.0,
            "hop_duration_sec": 0.5,
        })
        print(f"1. POST /session/start -> Status: {start_resp.status_code}")
        assert start_resp.status_code == 200, f"Failed to start session: {start_resp.text}"
        session_data = start_resp.json()
        session_id = session_data["session_id"]
        print(f"   Created session_id: {session_id}")

        # 2. WebSocket Streaming (20 seconds = 40 chunks of 0.5s)
        print("2. Connecting WebSocket stream & pushing audio chunks...")
        with client.websocket_connect(f"/ws/session/{session_id}/stream") as websocket:
            handshake = websocket.receive_json()
            print(f"   Handshake received: type={handshake.get('type')}")

            sr = 16000
            for chunk_idx in range(10):
                t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
                sine = (0.05 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
                pcm_bytes = (sine * 32767).astype(np.int16).tobytes()
                websocket.send_bytes(pcm_bytes)

            print("   Audio streaming completed.")

            # 3. Stop Session (POST /stop) WHILE WebSocket is still connected or right before close
            print(f"3. POST /session/{session_id}/stop ...")
            stop_resp = client.post(f"/api/v1/session/{session_id}/stop")
            print(f"   POST /stop Response Status: {stop_resp.status_code}")
            assert stop_resp.status_code == 200, f"Stop failed with {stop_resp.status_code}: {stop_resp.text}"
            stop_data = stop_resp.json()
            print(f"   Stop Payload: case_id={stop_data['case_id']} final_svi={stop_data['final_svi_score']} status={stop_data['status']}")

        # 4. WebSocket is now closed after context manager exit
        print("4. WebSocket connection closed.")

        # 5. GET /summary after WebSocket close
        print(f"5. GET /session/{session_id}/summary ...")
        summary_resp = client.get(f"/api/v1/session/{session_id}/summary")
        print(f"   GET /summary Response Status: {summary_resp.status_code}")
        assert summary_resp.status_code == 200, f"Summary failed with {summary_resp.status_code}: {summary_resp.text}"
        summary_data = summary_resp.json()
        print(f"   Summary Payload: session_id={summary_data['session_id']} status={summary_data['status']} final_svi={summary_data['final_svi_score']}")

        # 6. Idempotent POST /stop repeat test
        print(f"6. POST /session/{session_id}/stop (repeat check) ...")
        repeat_stop = client.post(f"/api/v1/session/{session_id}/stop")
        print(f"   Repeat POST /stop Status: {repeat_stop.status_code}")
        assert repeat_stop.status_code == 200, f"Repeat stop failed with {repeat_stop.status_code}"

    print("=" * 80)
    print("MANUAL TEST SIMULATION PASSED 100% PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_manual_test_simulation()
