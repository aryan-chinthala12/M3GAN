const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function analyzeAudio(blob, language) {
  const formData = new FormData();

  const extension = blob.type.includes("ogg") ? "ogg" : "webm";

  formData.append(
    "file",
    blob,
    `live-assessment.${extension}`
  );

  formData.append("language", language);

  const response = await fetch(
    `${API_BASE_URL}/api/v1/analyze-audio`,
    {
      method: "POST",
      body: formData,
    }
  );

  let payload = null;

  try {
    payload = await response.json();
  } catch {
    // Backend returned a non-JSON response.
  }

  if (!response.ok) {
    throw new Error(
      payload?.detail ||
      `Analysis failed with HTTP ${response.status}.`
    );
  }

  return payload;
}

export { API_BASE_URL };
