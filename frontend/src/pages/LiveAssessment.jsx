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
  MessageSquare,
  Lock,
} from "lucide-react";
import { analyzeAudio, analyzeText } from "../api/assessment";

const RISK_STYLES = {
  CRITICAL: "bg-red-100 text-red-800 border-red-300",
  HIGH: "bg-orange-100 text-orange-800 border-orange-300",
  MODERATE: "bg-yellow-100 text-yellow-800 border-yellow-300",
  LOW: "bg-green-100 text-green-800 border-green-300",
};

const LANGUAGES = [
  "Hindi",
  "English",
  "Marathi",
  "Bengali",
  "Tamil",
  "Telugu",
  "Kannada",
];

const SAMPLE_TEXTS = [
  {
    label: "Hindi (Romanized) — intimidation",
    language: "Hindi",
    channel: "chat",
    text: "Woh log mujhe roz dhamki de rahe hain. Main bahut darr gayi hoon, koi madad nahi kar raha. Gaon walon ne bahishkar kar diya hai.",
  },
  {
    label: "English — severe distress",
    language: "English",
    channel: "chat",
    text: "They raped me and now they threaten to kill me if I file a complaint. I see no way out, I want to die.",
  },
  {
    label: "Tamil — fear / isolation",
    language: "Tamil",
    channel: "portal",
    text: "எனக்கு பயமாக இருக்கிறது, என்னை யாரும் உதவி செய்யவில்லை. என்னை ஊரிலிருந்து வெளியேற்றினார்கள்.",
  },
  {
    label: "English — routine grievance",
    language: "English",
    channel: "chat",
    text: "I applied for the scholarship three months ago and there is no update yet. Please check the status.",
  },
];

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
    if (!Number.isFinite(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return String(value).replaceAll("_", " ").toUpperCase();
}

function IndicatorTable({ rows }) {
  return (
    <div className="space-y-3">
      {rows.map(([label, value]) => (
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
  );
}

export default function LiveAssessment({ onAssessed }) {
  const [mode, setMode] = useState("voice"); // "voice" | "text"
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [language, setLanguage] = useState("Hindi");
  const [channel, setChannel] = useState("chat");
  const [textInput, setTextInput] = useState("");
  const [consent, setConsent] = useState(false);
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

  // ---------------- Voice mode ----------------
  const startAssessment = async () => {
    setError("");
    setAnalysis(null);

    if (!consent) {
      setError("Please record informed consent before starting the assessment.");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone recording is not supported in this browser.");
      setStatus("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
        if (event.data.size > 0) chunksRef.current.push(event.data);
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
    if (!recorder || recorder.state === "inactive") return;

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

        const result = await analyzeAudio(recordedBlob, language, consent);
        setAnalysis(result);
        setStatus("complete");
        onAssessed?.();
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

  // ---------------- Text mode ----------------
  const submitText = async () => {
    setError("");
    setAnalysis(null);

    if (!consent) {
      setError("Please record informed consent before running the assessment.");
      return;
    }

    if (!textInput.trim()) {
      setError("Please enter the complainant's narrative before analyzing.");
      return;
    }

    setStatus("analyzing");
    try {
      const result = await analyzeText({
        text: textInput,
        channel,
        language,
      });
      setAnalysis(result);
      setStatus("complete");
      onAssessed?.();
    } catch (analysisError) {
      console.error("Text analysis error:", analysisError);
      setError(
        analysisError?.message ||
          "The narrative could not be analyzed. Please check that the FastAPI backend is running."
      );
      setStatus("error");
    }
  };

  const useSample = (sample) => {
    setTextInput(sample.text);
    setLanguage(sample.language);
    setChannel(sample.channel);
    setAnalysis(null);
    setError("");
    setStatus("idle");
  };

  // ---------------- Derived view data ----------------
  const riskLevel = analysis?.svi_metrics?.risk_band || null;
  const sviScore = analysis?.svi_metrics?.final_svi_score ?? null;
  const acoustic = analysis?.acoustic_indicators;
  const nlp = analysis?.nlp_indicators;
  const explainability = analysis?.explainability;
  const interventions = analysis?.recommended_interventions || [];
  const components = analysis?.svi_metrics?.components || [];

  const indicatorRows = analysis
    ? mode === "voice"
      ? [
          ["Speech Emotion", acoustic?.top_emotion],
          ["Pitch Volatility (st)", acoustic?.pitch_volatility],
          ["RMS Energy", acoustic?.rms_energy],
          ["Energy Variation (dB)", acoustic?.energy_variation],
          ["Emotion Confidence", acoustic?.confidence],
          ["Threat Level", nlp?.threat_level],
          ["Distress Density", analysis?.text_indicators?.distress_density],
          ["Case ID", analysis?.case_id],
        ]
      : [
          ["Threat Level", nlp?.threat_level],
          ["Word Count", analysis?.text_indicators?.word_count],
          ["Distress Density", analysis?.text_indicators?.distress_density],
          ["PII Redactions", nlp?.pii_redactions],
          ["Case ID", analysis?.case_id],
        ]
    : [];

  const statusLabel = {
    idle: "Assessment engine ready",
    recording: "Listening to incoming voice...",
    analyzing: "Analyzing...",
    complete: "Analysis complete",
    error: "Assessment error",
  }[status];

  return (
    <div className="p-6 space-y-6 max-w-6xl">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">
            Live Victim Assessment
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            AI-assisted multilingual stress and vulnerability screening
          </p>
        </div>

        <div className="flex bg-white border border-slate-200 rounded-lg p-1 shadow-sm">
          <button
            onClick={() => {
              setMode("voice");
              setAnalysis(null);
              setStatus("idle");
              setError("");
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === "voice"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-50"
            }`}
          >
            <Mic className="w-4 h-4" /> Voice
          </button>
          <button
            onClick={() => {
              setMode("text");
              setAnalysis(null);
              setStatus("idle");
              setError("");
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === "text"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-50"
            }`}
          >
            <MessageSquare className="w-4 h-4" /> Text / Chat
          </button>
        </div>
      </div>

      {/* Consent gate */}
      <div
        className={`rounded-lg border p-4 flex gap-3 transition-colors ${
          consent
            ? "bg-green-50 border-green-200"
            : "bg-white border-slate-200"
        }`}
      >
        <Lock
          className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
            consent ? "text-green-600" : "text-slate-400"
          }`}
        />
        <div className="flex-1">
          <p className="text-sm font-semibold text-slate-800">
            Informed consent (mandatory)
          </p>
          <p className="text-xs text-slate-500 mt-1">
            The complainant has been informed, in their preferred language, that
            this interaction is analyzed by an AI system for stress and
            vulnerability indicators; that personally identifiable details are
            redacted before storage; and that the output supports — never
            replaces — a trained human reviewer.
          </p>
          <label className="flex items-center gap-2 mt-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setConsent(event.target.checked)}
              className="w-4 h-4 accent-amber-600"
            />
            Consent recorded for this assessment
          </label>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-900">Assessment failed</p>
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
            {LANGUAGES.map((lang) => (
              <option key={lang}>{lang}</option>
            ))}
          </select>

          <p className="text-xs text-slate-400 mt-2">
            Drives Whisper STT in voice mode; distress lexicon covers all
            listed languages plus Romanized speech.
          </p>
        </div>

        {mode === "voice" ? (
          <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <Mic className="w-5 h-5 text-amber-600" />
              <h3 className="font-semibold text-slate-800">Audio Input</h3>
            </div>

            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-slate-500" />
              <span className="font-mono text-lg">
                {formatDuration(duration)}
              </span>
            </div>

            {isRecording && (
              <div className="mt-3 flex gap-1 items-end h-8">
                {[3, 6, 4, 8, 5, 9, 4, 7, 5, 8, 3, 6].map((height, index) => (
                  <div
                    key={index}
                    className="w-1.5 bg-amber-500 rounded-full animate-pulse"
                    style={{ height: `${height * 3}px` }}
                  />
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <MessageSquare className="w-5 h-5 text-amber-600" />
              <h3 className="font-semibold text-slate-800">Channel</h3>
            </div>

            <select
              value={channel}
              onChange={(event) => setChannel(event.target.value)}
              disabled={status === "analyzing"}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm bg-white disabled:bg-slate-50"
            >
              <option value="chat">Chatbot / Chat</option>
              <option value="portal">Integrated Portal Form</option>
              <option value="ivrs">IVRS Transcript</option>
              <option value="other">Other Digital Interface</option>
            </select>

            <p className="text-xs text-slate-400 mt-2">
              The narrative is PII-redacted before analysis and storage.
            </p>
          </div>
        )}

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

      {/* Mode-specific input area */}
      {mode === "voice" ? (
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
                <Activity className="w-7 h-7 animate-pulse" />
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
      ) : (
        <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="font-semibold text-slate-900">
              Complainant narrative
            </h3>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_TEXTS.map((sample) => (
                <button
                  key={sample.label}
                  onClick={() => useSample(sample)}
                  disabled={status === "analyzing"}
                  className="text-xs px-2.5 py-1 rounded-full border border-slate-200 bg-slate-50 text-slate-600 hover:bg-amber-50 hover:border-amber-200 transition-colors disabled:opacity-50"
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          <textarea
            value={textInput}
            onChange={(event) => setTextInput(event.target.value)}
            disabled={status === "analyzing"}
            rows={6}
            placeholder="Paste or type what the complainant wrote (chat message, portal form, chatbot conversation transcript)..."
            className="w-full border border-slate-300 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 disabled:bg-slate-50"
          />

          <button
            onClick={submitText}
            disabled={status === "analyzing" || !textInput.trim()}
            className="flex items-center gap-2 bg-slate-900 text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Activity className="w-5 h-5" />
            {status === "analyzing" ? "Analyzing..." : "Analyze Narrative"}
          </button>
        </div>
      )}

      {/* Results */}
      {analysis && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
              <h3 className="font-semibold text-slate-800">
                Stress Vulnerability Index
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                AI-assisted screening indicator · Case {analysis.case_id}
              </p>

              <div className="mt-6 flex items-end gap-2">
                <span className="text-5xl font-extrabold text-slate-900">
                  {sviScore?.toFixed(1)}
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
                          : "bg-green-500"
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
                className={`mt-4 inline-flex px-3 py-1 rounded-full border text-xs font-semibold ${getRiskStyle(
                  riskLevel
                )}`}
              >
                {riskLevel} RISK
              </div>

              {analysis?.svi_metrics?.override_triggered && (
                <p className="text-xs text-red-700 mt-3 font-medium">
                  Severity floor applied:{" "}
                  {analysis.svi_metrics.override_reason}
                </p>
              )}

              {components.length > 0 && (
                <div className="mt-5 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    SVI components
                  </p>
                  {components.map((component) => (
                    <div key={component.name}>
                      <div className="flex justify-between text-xs text-slate-600">
                        <span className="capitalize">
                          {component.name} (w={component.weight})
                        </span>
                        <span className="font-mono">
                          +{component.contribution}
                        </span>
                      </div>
                      <div className="mt-1 w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500 rounded-full"
                          style={{
                            width: `${Math.min(
                              100,
                              component.contribution * 2
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
              <h3 className="font-semibold text-slate-800">
                Detected Indicators
              </h3>
              <div className="mt-5">
                <IndicatorTable rows={indicatorRows} />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6">
              <h3 className="font-semibold text-slate-800">
                {mode === "voice" ? "Transcript (PII-redacted)" : "Narrative (PII-redacted)"}
              </h3>
              <div className="mt-4 rounded-lg bg-slate-50 border border-slate-100 p-4 text-sm leading-6 text-slate-700 whitespace-pre-wrap">
                {mode === "voice"
                  ? analysis.transcript || "No transcript returned."
                  : analysis.redacted_text}
              </div>

              {nlp?.distress_categories?.length > 0 && (
                <div className="mt-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Distress categories
                  </p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {nlp.distress_categories.map((category) => (
                      <span
                        key={category}
                        className="px-2.5 py-1 text-xs rounded-full border border-orange-200 bg-orange-50 text-orange-700"
                      >
                        {category}
                      </span>
                    ))}
                  </div>
                </div>
              )}

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
        </>
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
