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
 * Upload recorded audio for full multimodal assessment
 * (speech emotion + acoustic biomarkers + multilingual transcript analysis).
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

export { API_BASE_URL };
