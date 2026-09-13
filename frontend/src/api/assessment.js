const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function parseResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // Backend returned a non-JSON response.
  }

  if (!response.ok) {
    throw new Error(
      payload?.detail ||
        `Request failed with HTTP ${response.status}.`
    );
  }

  return payload;
}

/**
 * Upload recorded audio for full multimodal assessment (batch fallback path).
 */
export async function analyzeAudio(blob, language, consent = false) {
  const formData = new FormData();
  const extension = blob.type.includes("ogg") ? "ogg" : "webm";

  formData.append("file", blob, `live-assessment.${extension}`);
  formData.append("language", language);
  formData.append("consent", consent ? "true" : "false");

  const response = await fetch(`${API_BASE_URL}/api/v1/analyze-audio`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response);
}

/**
 * Analyze a written narrative from chat / portal / chatbot channels.
 */
export async function analyzeText({ text, channel = "chat", language = "English" }) {
  const response = await fetch(`${API_BASE_URL}/api/v1/analyze-text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, channel, language }),
  });

  return parseResponse(response);
}

/** Recent assessed cases, newest first. */
export async function getCases(limit = 100) {
  const response = await fetch(`${API_BASE_URL}/api/v1/cases?limit=${limit}`);
  return parseResponse(response);
}

/** Aggregate risk / channel statistics. */
export async function getStats() {
  const response = await fetch(`${API_BASE_URL}/api/v1/stats`);
  return parseResponse(response);
}

/** Full case record by id. */
export async function getCase(caseId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/cases/${caseId}`);
  return parseResponse(response);
}

/** Record a human operator's acknowledgement / override for a case. */
export async function respondIntervention({ callId, operatorId, actionTaken, notes }) {
  const response = await fetch(`${API_BASE_URL}/api/v1/interventions/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      call_id: callId,
      operator_id: operatorId,
      action_taken: actionTaken,
      notes: notes || null,
    }),
  });

  return parseResponse(response);
}

// =============================================================
// REAL-TIME STREAMING SESSION LIFECYCLE & WEBSOCKET CLIENT
// =============================================================

/** Start a real-time streaming assessment session. */
export async function startSession({
  channel = "voice",
  language = "English",
  consent = true,
  window_duration_sec = 3.0,
  hop_duration_sec = 0.5,
}) {
  const response = await fetch(`${API_BASE_URL}/api/v1/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      channel,
      language,
      consent,
      window_duration_sec,
      hop_duration_sec,
    }),
  });

  return parseResponse(response);
}

/** Stop an active streaming session and get final summary. */
export async function stopSession(sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/session/${sessionId}/stop`, {
    method: "POST",
  });

  return parseResponse(response);
}

/** Fetch detailed timeline summary for a session. */
export async function getSessionSummary(sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/session/${sessionId}/summary`);
  return parseResponse(response);
}

/**
 * Connect to WebSocket stream for a session.
 * Handshakes, handles versioned messages, binary/text audio transmission, and disconnects.
 */
export function connectAssessmentStream(sessionId, callbacks = {}) {
  const { onOpen, onMessage, onError, onClose } = callbacks;

  const wsUrl = API_BASE_URL.replace(/^http/, "ws") + `/ws/session/${sessionId}/stream`;
  const socket = new WebSocket(wsUrl);

  socket.onopen = (event) => {
    onOpen?.(event);
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "error") {
        onError?.(new Error(data.message || `Streaming error: ${data.code}`));
      } else {
        onMessage?.(data);
      }
    } catch {
      // Non-JSON binary or raw data
    }
  };

  socket.onerror = (event) => {
    onError?.(new Error("WebSocket stream connection error. Ensure FastAPI backend is running."));
  };

  socket.onclose = (event) => {
    onClose?.(event);
  };

  return {
    socket,
    sendAudioChunk: (pcmFloat32Array) => {
      if (socket.readyState === WebSocket.OPEN && pcmFloat32Array.length > 0) {
        // Convert Float32Array to Int16 PCM bytes for compact network transmission
        const int16Buffer = new Int16Array(pcmFloat32Array.length);
        for (let i = 0; i < pcmFloat32Array.length; i++) {
          const s = Math.max(-1, Math.min(1, pcmFloat32Array[i]));
          int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        socket.send(int16Buffer.buffer);
      }
    },
    close: () => {
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close();
      }
    },
  };
}

// =============================================================
// HARDWARE / PHONE / USB LINE INGESTION API (PHASE 5A)
// =============================================================

/** List available system audio input devices (microphones, USB audio, virtual cables). */
export async function getHardwareDevices() {
  const response = await fetch(`${API_BASE_URL}/api/v1/hardware/devices`);
  return parseResponse(response);
}

/** Start hardware soundcard audio ingestion for a session. */
export async function startHardwareIngest({ sessionId, deviceIndex, gainDb = 0.0 }) {
  const response = await fetch(`${API_BASE_URL}/api/v1/hardware/start-ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      device_index: deviceIndex,
      gain_db: gainDb,
    }),
  });
  return parseResponse(response);
}

/** Stop active hardware soundcard audio ingestion for a session. */
export async function stopHardwareIngest(sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/hardware/stop-ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return parseResponse(response);
}

/** Get status of hardware ingestion stream for a session. */
export async function getHardwareStatus(sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/hardware/status/${sessionId}`);
  return parseResponse(response);
}

export { API_BASE_URL };

