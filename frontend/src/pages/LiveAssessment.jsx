import { useEffect, useRef, useState } from "react";
import {
  Mic,
  Square,
  ShieldAlert,
  Activity,
  Languages,
  Clock,
  AlertCircle,
  CheckCircle2,
  Send,
} from "lucide-react";
import { analyzeAudio } from "../api/assessment";

const EMPTY_INDICATORS = [
  ["Top Emotion", "—"],
  ["Pitch Volatility", "—"],
  ["RMS Energy", "—"],
  ["Emotion Confidence", "—"],
  ["NLP Emotion", "—"],
  ["Threat Level", "—"],
];

const RISK_STYLES = {
  CRITICAL: "bg-red-100 text-red-800 border-red-300",
  HIGH: "bg-orange-100 text-orange-800 border-orange-300",
  MODERATE: "bg-yellow-100 text-yellow-800 border-yellow-300",
  LOW: "bg-green-100 text-green-800 border-green-300",
};

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  return `${String(minutes).padStart(2, "0")}:${String(
    remainingSeconds
  ).padStart(2, "0")}`;
}

function getRiskStyle(level) {
  return RISK_STYLES[level] || "bg-slate-100 text-slate-700 border-slate-300";
}

function formatIndicatorValue(key, value) {
  if (value === null || value === undefined) return "—";

  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(2) : "—";
  }

  return String(value).replaceAll("_", " ").toUpperCase();
}

async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioContextClass =
    window.AudioContext || window.webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error("This browser does not support audio decoding.");
  }

  const audioContext = new AudioContextClass();

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const wavBuffer = encodeWav(audioBuffer);
    return new Blob([wavBuffer], { type: "audio/wav" });
  } finally {
    await audioContext.close();
  }
}

function encodeWav(audioBuffer) {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const frameCount = audioBuffer.length;

  // Interleave all channels into signed 16-bit PCM.
  const interleaved = new Float32Array(frameCount * channelCount);

  for (let channel = 0; channel < channelCount; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);

    for (let frame = 0; frame < frameCount; frame += 1) {
      interleaved[frame * channelCount + channel] += channelData[frame];
    }
  }

  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      interleaved[frame * channelCount + channel] /= channelCount;
    }
  }

  const bytesPerSample = 2;
  const dataSize = interleaved.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;

  for (let index = 0; index < interleaved.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, interleaved[index]));
    const pcm = sample < 0 ? sample * 0x8000 : sample * 0x7fff;

    view.setInt16(offset, pcm, true);
    offset += 2;
  }

  return buffer;
}

export default function LiveAssessment() {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [language, setLanguage] = useState("Hindi");
  const [status, setStatus] = useState("idle");
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const chunksRef = useRef([]);

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startAssessment = async () => {
    setError("");
    setAnalysis(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone recording is not supported in this browser.");
      setStatus("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      chunksRef.current = [];

      const preferredMimeTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
      ];

      const supportedMimeType = preferredMimeTypes.find((mimeType) =>
        MediaRecorder.isTypeSupported(mimeType)
      );

      const recorder = supportedMimeType
        ? new MediaRecorder(stream, { mimeType: supportedMimeType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setError("The browser failed while recording audio.");
        setStatus("error");
        setIsRecording(false);
      };

      recorder.start();
      setIsRecording(true);
      setStatus("recording");
      setDuration(0);

      clearInterval(timerRef.current);
      timerRef.current = setInterval(() => {
        setDuration((previous) => previous + 1);
      }, 1000);
    } catch (captureError) {
      console.error("Microphone access error:", captureError);
      setError(
        "Microphone access was denied or is unavailable. Please allow microphone access and try again."
      );
      setStatus("error");
    }
  };

  const stopAssessment = () => {
    const recorder = mediaRecorderRef.current;

    if (!recorder || recorder.state === "inactive") {
      return;
    }

    recorder.onstop = async () => {
      clearInterval(timerRef.current);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      setIsRecording(false);
      setStatus("analyzing");
      setError("");

      try {
        const recordedBlob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });

        if (!recordedBlob.size) {
          throw new Error(
            "No audio was captured. Please record for a few seconds and try again."
          );
        }

        const wavBlob = await blobToWav(recordedBlob);
        const result = await analyzeAudio(wavBlob, language);

        setAnalysis(result);
        setStatus("complete");
      } catch (analysisError) {
        console.error("Assessment analysis error:", analysisError);
        setError(
          analysisError?.message ||
            "The audio could not be analyzed. Please check that the FastAPI backend is running."
        );
        setStatus("error");
      } finally {
        mediaRecorderRef.current = null;
        chunksRef.current = [];
      }
    };

    recorder.stop();
  };

  const riskLevel = analysis?.svi_metrics?.risk_band || null;
  const sviScore = analysis?.svi_metrics?.final_svi_score ?? null;
  const acoustic = analysis?.acoustic_indicators;
  const nlp = analysis?.nlp_indicators;
  const explainability = analysis?.explainability;
  const interventions = analysis?.recommended_interventions || [];

  const indicatorRows = analysis
    ? [
        ["Top Emotion", acoustic?.top_emotion],
        ["Pitch Volatility", acoustic?.pitch_volatility],
        ["RMS Energy", acoustic?.rms_energy],
        ["Emotion Confidence", acoustic?.confidence],
        ["NLP Emotion", nlp?.detected_emotion],
        ["Threat Level", nlp?.threat_level],
      ]
    : EMPTY_INDICATORS;

  const statusLabel = {
    idle: "Assessment engine ready",
    recording: "Listening to incoming voice...",
    analyzing: "Analyzing captured audio...",
    complete: "Analysis complete",
    error: "Assessment error",
  }[status];

  return (
    <div className="p-6 space-y-6 max-w-6xl">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">
          Live Victim Assessment
        </h2>

        <p className="text-sm text-slate-500 mt-1">
          AI-assisted voice stress and vulnerability screening
        </p>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div className="flex items-center gap-4">
            <div
              className={`w-14 h-14 rounded-full flex items-center justify-center ${
                isRecording
                  ? "bg-red-100 text-red-600"
                  : status === "analyzing"
                    ? "bg-amber-100 text-amber-700"
                    : "bg-slate-100 text-slate-600"
              }`}
            >
              {isRecording ? (
                <Activity className="w-7 h-7 animate-pulse" />
              ) : status === "analyzing" ? (
                <Activity className="w-7 h-7 animate-pulse" />
              ) : (
                <Mic className="w-7 h-7" />
              )}
            </div>

            <div>
              <h3 className="font-semibold text-slate-900">
                {isRecording
                  ? "Assessment in progress"
                  : status === "analyzing"
                    ? "Processing assessment"
                    : status === "complete"
                      ? "Assessment ready for review"
                      : "Ready for assessment"}
              </h3>

              <p className="text-sm text-slate-500">
                {status === "idle"
                  ? "Start the assessment when the complainant is ready."
                  : status === "recording"
                    ? "Listening to incoming voice..."
                    : status === "analyzing"
                      ? "Sending captured audio to the AI assessment engine."
                      : status === "complete"
                        ? "Review the generated indicators below."
                        : status === "error"
                          ? "Resolve the issue and try the assessment again."
                          : "Ready."}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-slate-600">
            <Clock className="w-5 h-5" />
            <span className="font-mono text-lg">
              {formatDuration(duration)}
            </span>
          </div>

          {!isRecording ? (
            <button
              onClick={startAssessment}
              disabled={status === "analyzing"}
              className="flex items-center justify-center gap-2 bg-slate-900 text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Mic className="w-5 h-5" />
              {status === "complete" ? "Start New Assessment" : "Start Assessment"}
            </button>
          ) : (
            <button
              onClick={stopAssessment}
              className="flex items-center justify-center gap-2 bg-red-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-red-700 transition-colors"
            >
              <Square className="w-5 h-5" />
              Stop & Analyze
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-900">
              Assessment failed
            </p>
            <p className="text-sm text-red-800 mt-1">{error}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <Languages className="w-5 h-5 text-amber-600" />
            <h3 className="font-semibold text-slate-800">Language</h3>
          </div>

          <select
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            disabled={isRecording || status === "analyzing"}
            className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm bg-white disabled:bg-slate-50"
          >
            <option>Hindi</option>
            <option>English</option>
            <option>Marathi</option>
            <option>Bengali</option>
            <option>Tamil</option>
            <option>Telugu</option>
            <option>Kannada</option>
          </select>

          <p className="text-xs text-slate-400 mt-2">
            Selected language is included in the API request for future
            language-aware backend processing.
          </p>
        </div>

        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <Mic className="w-5 h-5 text-amber-600" />
            <h3 className="font-semibold text-slate-800">Audio Input</h3>
          </div>

          <p className="text-sm text-slate-600">
            {isRecording ? "Microphone active" : "Microphone inactive"}
          </p>

          {isRecording && (
            <div className="mt-3 flex gap-1 items-end h-8">
              {[3, 6, 4, 8, 5, 9, 4, 7, 5, 8, 3, 6].map(
                (height, index) => (
                  <div
                    key={index}
                    className="w-1.5 bg-amber-500 rounded-full animate-pulse"
                    style={{ height: `${height * 3}px` }}
                  />
                )
              )}
            </div>
          )}
        </div>

        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <Activity className="w-5 h-5 text-amber-600" />
            <h3 className="font-semibold text-slate-800">AI Status</h3>
          </div>

          <div className="flex items-center gap-2">
            {status === "complete" ? (
              <CheckCircle2 className="w-4 h-4 text-green-600" />
            ) : status === "error" ? (
              <AlertCircle className="w-4 h-4 text-red-600" />
            ) : (
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  status === "recording" || status === "analyzing"
                    ? "bg-amber-500 animate-pulse"
                    : "bg-green-500"
                }`}
              />
            )}

            <span className="text-sm text-slate-600">{statusLabel}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
          <h3 className="font-semibold text-slate-800">
            Stress Vulnerability Index
          </h3>

          <p className="text-xs text-slate-500 mt-1">
            AI-assisted screening indicator
          </p>

          <div className="mt-6 flex items-end gap-2">
            <span className="text-5xl font-extrabold text-slate-900">
              {sviScore === null ? "--" : sviScore.toFixed(1)}
            </span>
            <span className="text-slate-500 mb-2">/ 100</span>
          </div>

          <div className="mt-4 w-full h-3 bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                riskLevel === "CRITICAL"
                  ? "bg-red-500"
                  : riskLevel === "HIGH"
                    ? "bg-orange-500"
                    : riskLevel === "MODERATE"
                      ? "bg-yellow-500"
                      : riskLevel === "LOW"
                        ? "bg-green-500"
                        : "bg-slate-300"
              }`}
              style={{
                width: `${Math.max(
                  0,
                  Math.min(100, Number(sviScore) || 0)
                )}%`,
              }}
            />
          </div>

          <div
            className={`mt-4 inline-flex px-3 py-1 rounded-full border text-xs font-semibold ${
              riskLevel
                ? getRiskStyle(riskLevel)
                : "border-slate-300 bg-slate-50 text-slate-500"
            }`}
          >
            {riskLevel || "Awaiting assessment"}
          </div>

          {analysis?.svi_metrics?.override_triggered && (
            <p className="text-xs text-red-700 mt-3 font-medium">
              Override triggered:{" "}
              {analysis.svi_metrics.override_reason || "High-risk indicator detected."}
            </p>
          )}
        </div>

        <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
          <h3 className="font-semibold text-slate-800">
            Detected Indicators
          </h3>

          <div className="mt-5 space-y-3">
            {indicatorRows.map(([label, value]) => (
              <div
                key={label}
                className="flex justify-between items-center py-2 border-b border-slate-100"
              >
                <span className="text-sm text-slate-600">{label}</span>
                <span
                  className={`text-sm font-semibold ${
                    value === "—" ? "text-slate-400" : "text-slate-800"
                  }`}
                >
                  {formatIndicatorValue(label, value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6">
            <h3 className="font-semibold text-slate-800">Transcript</h3>
            <div className="mt-4 rounded-lg bg-slate-50 border border-slate-100 p-4 text-sm leading-6 text-slate-700 whitespace-pre-wrap">
              {analysis.transcript || "No transcript returned."}
            </div>

            {nlp?.flagged_keywords?.length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Flagged keywords
                </p>
                <div className="flex flex-wrap gap-2 mt-2">
                  {nlp.flagged_keywords.map((keyword) => (
                    <span
                      key={keyword}
                      className="px-2.5 py-1 text-xs rounded-full border border-red-200 bg-red-50 text-red-700"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6">
            <h3 className="font-semibold text-slate-800">
              Explainability & Recommended Actions
            </h3>

            {explainability?.summary && (
              <p className="mt-4 text-sm leading-6 text-slate-600">
                {explainability.summary}
              </p>
            )}

            {explainability?.top_contributors?.length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Top contributors
                </p>
                <ul className="mt-2 space-y-2 text-sm text-slate-700">
                  {explainability.top_contributors.map((item) => (
                    <li
                      key={item}
                      className="border border-slate-100 bg-slate-50 rounded-md p-2.5"
                    >
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {interventions.length > 0 && (
              <div className="mt-5">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Recommended interventions
                </p>

                <ul className="mt-2 space-y-2">
                  {interventions.map((action) => (
                    <li
                      key={action}
                      className="flex items-start gap-2 text-sm text-slate-700 bg-amber-50 border border-amber-100 p-2.5 rounded"
                    >
                      <Send className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                      <span>{action}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />

        <div>
          <p className="text-sm font-semibold text-amber-900">
            Human oversight required
          </p>

          <p className="text-xs text-amber-800 mt-1">
            AI-generated indicators are intended to support authorized human
            review and should not be treated as a clinical diagnosis or
            automatic determination of risk.
          </p>
        </div>
      </div>
    </div>
  );
}
