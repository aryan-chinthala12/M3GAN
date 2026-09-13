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
  Radio,
  Sliders,
} from "lucide-react";
import {
  analyzeText,
  startSession,
  stopSession,
  connectAssessmentStream,
  respondIntervention,
  getHardwareDevices,
  startHardwareIngest,
  stopHardwareIngest,
} from "../api/assessment";
import SVITrendTimeline from "../components/SVITrendTimeline";
import SafetyFlags from "../components/SafetyFlags";

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
  const remainingSeconds = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(
    remainingSeconds
  ).padStart(2, "0")}`;
}

function getRiskStyle(level) {
  return RISK_STYLES[level] || "bg-slate-100 text-slate-700 border-slate-300";
}

function formatIndicatorValue(label, value) {
  if (value === null || value === undefined) return "Unavailable";
  if (typeof value === "boolean") return value ? "Reliable" : "Unreliable";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "Unavailable";
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return String(value).replaceAll("_", " ").toUpperCase();
}

function IndicatorTable({ rows }) {
  return (
    <div className="space-y-3">
      {rows.map(([label, value]) => {
        const isUnavailable = value === "Unavailable" || value === null || value === undefined;
        return (
          <div
            key={label}
            className="flex justify-between items-center py-2 border-b border-slate-100"
          >
            <span className="text-sm text-slate-600">{label}</span>
            <span
              className={`text-sm font-semibold ${
                isUnavailable ? "text-slate-400 italic font-normal" : "text-slate-800"
              }`}
            >
              {formatIndicatorValue(label, value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function LiveAssessment({ onAssessed }) {
  const [mode, setMode] = useState("voice"); // "voice" | "text"
  const [audioSource, setAudioSource] = useState("browser"); // "browser" | "hardware"
  const [hardwareDevices, setHardwareDevices] = useState([]);
  const [selectedDeviceIndex, setSelectedDeviceIndex] = useState(null);

  const [sessionState, setSessionState] = useState("IDLE"); // IDLE | CONNECTING | CONNECTED | RECORDING | STOPPING | COMPLETED | ERROR
  const [duration, setDuration] = useState(0);
  const [language, setLanguage] = useState("Hindi");
  const [channel, setChannel] = useState("chat");
  const [textInput, setTextInput] = useState("");
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState("");

  // Live streaming states
  const [sessionId, setSessionId] = useState(null);
  const [liveMessage, setLiveMessage] = useState(null);
  const [timelineHistory, setTimelineHistory] = useState([]);
  const [finalSummary, setFinalSummary] = useState(null);
  const [escalateState, setEscalateState] = useState("idle");

  const streamClientRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const scriptNodeRef = useRef(null);
  const timerRef = useRef(null);

  // Clean up audio & WS connections on unmount
  useEffect(() => {
    return () => {
      cleanupAudioAndStream();
    };
  }, []);

  // Fetch hardware audio input devices when switching to hardware mode
  useEffect(() => {
    if (audioSource === "hardware") {
      getHardwareDevices()
        .then((res) => {
          if (res && res.devices) {
            setHardwareDevices(res.devices);
            const defDev = res.devices.find((d) => d.is_default) || res.devices[0];
            if (defDev) setSelectedDeviceIndex(defDev.device_index);
          }
        })
        .catch((err) => console.error("Error fetching hardware devices:", err));
    }
  }, [audioSource]);

  const cleanupAudioAndStream = () => {
    clearInterval(timerRef.current);
    if (scriptNodeRef.current) {
      try { scriptNodeRef.current.disconnect(); } catch {}
      scriptNodeRef.current = null;
    }
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch {}
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (streamClientRef.current) {
      streamClientRef.current.close();
      streamClientRef.current = null;
    }
  };

  // ---------------- Live Audio Assessment (Browser Mic & Hardware Line) ----------------
  const startLiveAssessment = async () => {
    setError("");
    setLiveMessage(null);
    setTimelineHistory([]);
    setFinalSummary(null);

    if (!consent) {
      setError("Please record informed consent before starting the live assessment.");
      return;
    }

    try {
      setSessionState("CONNECTING");

      // 1. Start Session on Backend
      const sessionRes = await startSession({
        channel: "voice",
        language,
        consent,
        window_duration_sec: 3.0,
        hop_duration_sec: 0.5,
      });

      const newSessionId = sessionRes.session_id;
      setSessionId(newSessionId);

      // 2. Connect WebSocket Stream for Real-Time Updates
      const client = connectAssessmentStream(newSessionId, {
        onOpen: () => {
          setSessionState("CONNECTED");
        },
        onMessage: (msg) => {
          if (msg.type === "assessment_update") {
            setLiveMessage(msg);
            setTimelineHistory((prev) => [
              ...prev,
              {
                timestamp_sec: msg.audio_duration,
                raw_svi: msg.svi.raw_score,
                smoothed_svi: msg.svi.smoothed_score,
                risk_band: msg.svi.risk_band,
                trend: msg.svi.trend,
              },
            ]);
          }
        },
        onError: (wsErr) => {
          console.error("Streaming WebSocket error:", wsErr);
          setError(wsErr.message || "Live assessment WebSocket connection lost.");
          setSessionState("ERROR");
        },
        onClose: () => {
          console.log("WebSocket connection closed.");
        },
      });

      streamClientRef.current = client;

      if (audioSource === "hardware") {
        // --- PHASE 5A: Hardware / USB Soundcard / Virtual Line Ingest Path ---
        await startHardwareIngest({
          sessionId: newSessionId,
          deviceIndex: selectedDeviceIndex,
        });

        setSessionState("RECORDING");
        setDuration(0);

        clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
          setDuration((prev) => prev + 1);
        }, 1000);
      } else {
        // --- Browser WebAudio PCM Microphone Streaming Path ---
        if (!navigator.mediaDevices?.getUserMedia) {
          setError("Microphone recording is not supported in this browser.");
          setSessionState("ERROR");
          return;
        }

        const mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
        mediaStreamRef.current = mediaStream;

        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioCtx();
        audioContextRef.current = audioCtx;

        const sourceNode = audioCtx.createMediaStreamSource(mediaStream);
        const scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
        scriptNodeRef.current = scriptNode;

        const inputSampleRate = audioCtx.sampleRate;
        const targetSampleRate = 16000;

        scriptNode.onaudioprocess = (audioEvent) => {
          const inputData = audioEvent.inputBuffer.getChannelData(0);
          if (!inputData || inputData.length === 0) return;

          let pcm16k;
          if (inputSampleRate === targetSampleRate) {
            pcm16k = inputData;
          } else {
            const ratio = inputSampleRate / targetSampleRate;
            const newLength = Math.floor(inputData.length / ratio);
            pcm16k = new Float32Array(newLength);
            for (let i = 0; i < newLength; i++) {
              const origIdx = Math.floor(i * ratio);
              pcm16k[i] = inputData[origIdx];
            }
          }

          if (streamClientRef.current) {
            streamClientRef.current.sendAudioChunk(pcm16k);
          }
        };

        sourceNode.connect(scriptNode);
        scriptNode.connect(audioCtx.destination);

        setSessionState("RECORDING");
        setDuration(0);

        clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
          setDuration((prev) => prev + 1);
        }, 1000);
      }
    } catch (err) {
      console.error("Live assessment start error:", err);
      cleanupAudioAndStream();
      setError(
        err.name === "NotAllowedError"
          ? "Microphone permission was denied. Please allow microphone access in your browser settings."
          : err.message || "Could not start live assessment. Verify the backend server is running."
      );
      setSessionState("ERROR");
    }
  };

  const stopLiveAssessment = async () => {
    if (!sessionId) return;

    setSessionState("STOPPING");
    clearInterval(timerRef.current);

    try {
      if (audioSource === "hardware") {
        await stopHardwareIngest(sessionId).catch((err) =>
          console.warn("Hardware ingest stop warning:", err)
        );
      }

      // 1. Call REST POST /stop to finalize backend session state first
      const summary = await stopSession(sessionId);
      setFinalSummary(summary);

      // 2. Clean up WebAudio & WebSocket stream after finalization
      cleanupAudioAndStream();

      setSessionState("COMPLETED");
      onAssessed?.();
    } catch (stopErr) {
      console.error("Error stopping session:", stopErr);
      cleanupAudioAndStream();
      setError("Could not retrieve final session summary from backend.");
      setSessionState("ERROR");
    }
  };


  // ---------------- Text mode (Batch Fallback) ----------------
  const submitText = async () => {
    setError("");
    setLiveMessage(null);

    if (!consent) {
      setError("Please record informed consent before analyzing narrative.");
      return;
    }

    if (!textInput.trim()) {
      setError("Please enter the complainant's narrative before analyzing.");
      return;
    }

    setSessionState("ANALYZING");
    try {
      const result = await analyzeText({ text: textInput, channel, language });
      // Map text result to liveMessage shape for visual consistency
      setLiveMessage({
        transcript: result.redacted_text,
        svi: {
          raw_score: result.svi_metrics.final_svi_score,
          smoothed_score: result.svi_metrics.final_svi_score,
          risk_band: result.svi_metrics.risk_band,
          risk_color: result.svi_metrics.risk_color,
          trend: "STABLE",
        },
        confidence: result.confidence_metrics || {
          overall_confidence: 100.0,
          confidence_rating: "HIGH",
          signal_quality: "GOOD",
        },
        signal_quality: "GOOD",
        emotion: { top_emotion: "NEUTRAL", confidence: 0.0, stress_score: 0.0 },
        acoustics: { pitch_volatility: 0.0, rms_energy: 0.0, pitch_reliable: false },
        indicators: result.nlp_indicators,
        safety_flags: result.svi_metrics.safety_flags || [],
        explainability: result.explainability,
        recommended_interventions: result.recommended_interventions || [],
      });
      setSessionState("COMPLETED");
      onAssessed?.();
    } catch (analysisError) {
      console.error("Text analysis error:", analysisError);
      setError(analysisError?.message || "The narrative could not be analyzed.");
      setSessionState("ERROR");
    }
  };

  const handleEscalateCase = async () => {
    if (!liveMessage && !finalSummary) return;
    setEscalateState("sending");
    try {
      await respondIntervention({
        callId: liveMessage?.session_id || finalSummary?.session_id || "LIVE-CALL",
        operatorId: "nhaa-operator-1",
        actionTaken: "Escalated for Urgent Human Review",
        notes: liveMessage?.transcript || finalSummary?.final_transcript || null,
      });
      setEscalateState("done");
    } catch {
      setEscalateState("error");
    }
  };

  // ---------------- View Helper Properties ----------------
  const sviScore = liveMessage?.svi?.smoothed_score ?? finalSummary?.final_svi_score ?? null;
  const riskBand = liveMessage?.svi?.risk_band ?? finalSummary?.final_risk_band ?? null;
  const trend = liveMessage?.svi?.trend ?? "INSUFFICIENT_DATA";
  const confidence = liveMessage?.confidence;
  const safetyFlags = liveMessage?.safety_flags ?? finalSummary?.safety_flags ?? [];
  const transcriptText = liveMessage?.transcript ?? finalSummary?.final_transcript ?? "";
  const acoustics = liveMessage?.acoustics;
  const nlp = liveMessage?.indicators;

  const indicatorRows = liveMessage
    ? mode === "voice"
      ? [
          ["Speech Emotion", liveMessage.emotion?.top_emotion],
          ["Pitch Volatility (st)", acoustics?.pitch_reliable ? acoustics?.pitch_volatility : "Unavailable"],
          ["RMS Energy", acoustics?.rms_energy],
          ["Voiced Speech Ratio", acoustics?.voiced_ratio],
          ["Pause Ratio", acoustics?.pause_ratio],
          ["Threat Level", nlp?.threat_level],
          ["Signal Quality", liveMessage.signal_quality],
          ["Session ID", sessionId],
        ]
      : [
          ["Threat Level", nlp?.threat_level],
          ["Distress Categories", nlp?.distress_categories?.join(", ")],
          ["Signal Quality", "GOOD"],
        ]
    : [];

  const statusLabel = {
    IDLE: "Assessment engine ready",
    CONNECTING: "Connecting WebSocket stream...",
    CONNECTED: "Stream connected — starting mic...",
    RECORDING: "Listening to incoming voice stream...",
    STOPPING: "Finalizing live assessment...",
    ANALYZING: "Analyzing text narrative...",
    COMPLETED: "Live assessment complete",
    ERROR: "Assessment error",
  }[sessionState];

  return (
    <div className="p-6 space-y-6 max-w-6xl">
      {/* Header & Mode Switcher */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">
            Live Victim Assessment
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            AI-assisted real-time multilingual stress, vulnerability and urgency triage
          </p>
        </div>

        <div className="flex bg-white border border-slate-200 rounded-lg p-1 shadow-sm">
          <button
            onClick={() => {
              setMode("voice");
              cleanupAudioAndStream();
              setSessionState("IDLE");
              setLiveMessage(null);
              setFinalSummary(null);
              setError("");
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === "voice"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-50"
            }`}
          >
            <Mic className="w-4 h-4" /> Live Voice Stream
          </button>
          <button
            onClick={() => {
              setMode("text");
              cleanupAudioAndStream();
              setSessionState("IDLE");
              setLiveMessage(null);
              setFinalSummary(null);
              setError("");
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === "text"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-50"
            }`}
          >
            <MessageSquare className="w-4 h-4" /> Text / Chat Narrative
          </button>
        </div>
      </div>

      {/* Mandatory Consent Gate */}
      <div
        className={`rounded-lg border p-4 flex gap-3 transition-colors ${
          consent ? "bg-green-50 border-green-200" : "bg-white border-slate-200"
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
            The caller has been informed that this interaction is analyzed in real time for observable stress and vulnerability indicators; personally identifiable details are redacted before persistence; and output supports human triage decisions without autonomous clinical diagnosis.
          </p>
          <label className="flex items-center gap-2 mt-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
              disabled={sessionState === "RECORDING"}
              className="w-4 h-4 accent-amber-600"
            />
            Consent recorded for this interaction
          </label>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-900">Assessment issue</p>
            <p className="text-sm text-red-800 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Operational Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <Languages className="w-5 h-5 text-amber-600" />
            <h3 className="font-semibold text-slate-800">Language</h3>
          </div>

          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            disabled={sessionState === "RECORDING"}
            className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm bg-white disabled:bg-slate-50"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang}>{lang}</option>
            ))}
          </select>
          <p className="text-xs text-slate-400 mt-2">
            Whisper ASR language target & multilingual lexicon coverage.
          </p>
        </div>

        {mode === "voice" ? (
          <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <Radio className="w-5 h-5 text-amber-600" />
              <h3 className="font-semibold text-slate-800">Stream Activity</h3>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-slate-500" />
                <span className="font-mono text-lg font-bold text-slate-800">
                  {formatDuration(duration)}
                </span>
              </div>
              <span
                className={`text-2xs font-bold uppercase px-2.5 py-1 rounded border ${
                  sessionState === "RECORDING"
                    ? "bg-red-100 text-red-700 border-red-200 animate-pulse"
                    : sessionState === "CONNECTED"
                    ? "bg-green-100 text-green-700 border-green-200"
                    : "bg-slate-100 text-slate-600 border-slate-200"
                }`}
              >
                {sessionState === "RECORDING" ? "Streaming Live" : sessionState}
              </span>
            </div>
          </div>
        ) : (
          <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <MessageSquare className="w-5 h-5 text-amber-600" />
              <h3 className="font-semibold text-slate-800">Channel</h3>
            </div>

            <select
              value={channel}
              onChange={(e) => setChannel(e.target.value)}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm bg-white"
            >
              <option value="chat">Chatbot / Web Chat</option>
              <option value="portal">Integrated Portal Form</option>
              <option value="ivrs">IVRS Transcript</option>
            </select>
          </div>
        )}

        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <Activity className="w-5 h-5 text-amber-600" />
            <h3 className="font-semibold text-slate-800">Engine Status</h3>
          </div>

          <div className="flex items-center gap-2">
            {sessionState === "COMPLETED" ? (
              <CheckCircle2 className="w-4 h-4 text-green-600" />
            ) : sessionState === "ERROR" ? (
              <AlertCircle className="w-4 h-4 text-red-600" />
            ) : (
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  sessionState === "RECORDING" || sessionState === "CONNECTING"
                    ? "bg-amber-500 animate-pulse"
                    : "bg-green-500"
                }`}
              />
            )}
            <span className="text-sm text-slate-600">{statusLabel}</span>
          </div>
        </div>
      </div>

      {/* Voice / Text Mode Controller */}
      {mode === "voice" ? (
        <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6 space-y-4">
          {/* Audio Source Mode Selection */}
          <div className="flex flex-wrap items-center justify-between gap-4 p-3 bg-slate-50 border border-slate-200 rounded-lg text-sm">
            <div className="flex items-center gap-4">
              <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-slate-500" /> Audio Ingest Source:
              </span>
              <label className="flex items-center gap-2 cursor-pointer font-medium text-slate-800">
                <input
                  type="radio"
                  name="audioSource"
                  value="browser"
                  checked={audioSource === "browser"}
                  onChange={() => setAudioSource("browser")}
                  disabled={sessionState === "RECORDING" || sessionState === "CONNECTING"}
                  className="accent-slate-900"
                />
                <span>Browser Mic (WebRTC)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer font-medium text-slate-800">
                <input
                  type="radio"
                  name="audioSource"
                  value="hardware"
                  checked={audioSource === "hardware"}
                  onChange={() => setAudioSource("hardware")}
                  disabled={sessionState === "RECORDING" || sessionState === "CONNECTING"}
                  className="accent-slate-900"
                />
                <span>Hardware / External Audio Source</span>
              </label>
            </div>

            {audioSource === "hardware" && (
              <div className="flex items-center gap-2">
                <span className="bg-indigo-100 text-indigo-800 border border-indigo-200 text-2xs font-extrabold px-2.5 py-0.5 rounded-full uppercase tracking-wider">
                  DEMO VOICE CHANNEL
                </span>
                {consent && (
                  <span className="bg-emerald-100 text-emerald-800 border border-emerald-200 text-2xs font-extrabold px-2.5 py-0.5 rounded-full uppercase tracking-wider">
                    CONSENT RECORDED
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Hardware Device Selection Dropdown & Telemetry Status */}
          {audioSource === "hardware" && (
            <div className="p-3 bg-blue-50/80 border border-blue-200 rounded-lg space-y-2 text-sm">
              <div className="flex flex-wrap items-center gap-3">
                <Radio className="w-4 h-4 text-blue-600 animate-pulse" />
                <span className="font-semibold text-blue-900">Select Input Interface:</span>
                <select
                  value={selectedDeviceIndex !== null ? selectedDeviceIndex : ""}
                  onChange={(e) => setSelectedDeviceIndex(Number(e.target.value))}
                  disabled={sessionState === "RECORDING" || sessionState === "CONNECTING"}
                  className="bg-white border border-slate-300 text-slate-800 rounded px-3 py-1.5 font-mono text-xs flex-1 outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {hardwareDevices.length === 0 ? (
                    <option value="">No hardware input devices found</option>
                  ) : (
                    hardwareDevices.map((dev) => (
                      <option key={dev.device_index} value={dev.device_index}>
                        #{dev.device_index} - {dev.name} ({dev.channels}ch, {dev.default_sample_rate}Hz)
                        {dev.is_default ? " [DEFAULT]" : ""}
                      </option>
                    ))
                  )}
                </select>
              </div>
              <p className="text-xs text-blue-700 font-medium">
                Prototype external voice-channel integration (Phone / USB Soundcard / Virtual Audio Cable).
              </p>
            </div>
          )}

          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 pt-2">
            <div className="flex items-center gap-4">
              <div
                className={`w-14 h-14 rounded-full flex items-center justify-center ${
                  sessionState === "RECORDING"
                    ? "bg-red-100 text-red-600"
                    : sessionState === "CONNECTING"
                    ? "bg-amber-100 text-amber-700"
                    : "bg-slate-100 text-slate-600"
                }`}
              >
                <Activity className="w-7 h-7 animate-pulse" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900">
                  {sessionState === "RECORDING"
                    ? "Real-Time Assessment Streaming"
                    : sessionState === "COMPLETED"
                    ? "Session Complete"
                    : "Ready to Stream"}
                </h3>
                <p className="text-sm text-slate-500">
                  {sessionState === "IDLE"
                    ? audioSource === "hardware"
                      ? "Click Start Live Assessment to ingest from selected soundcard/USB device."
                      : "Click Start Live Assessment when the caller is connected."
                    : sessionState === "RECORDING"
                    ? `16kHz PCM streaming to M3GAN backend over ${audioSource === "hardware" ? "sounddevice line ingest" : "WebRTC WebSocket"}...`
                    : "Session finalized. Review results below."}
                </p>
              </div>
            </div>

            {sessionState !== "RECORDING" && sessionState !== "CONNECTING" ? (
              <button
                onClick={startLiveAssessment}
                className="flex items-center justify-center gap-2 bg-slate-900 text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-800 transition-colors shadow-sm"
              >
                <Mic className="w-5 h-5" />
                {sessionState === "COMPLETED" ? "Start New Assessment" : "Start Live Assessment"}
              </button>
            ) : (
              <button
                onClick={stopLiveAssessment}
                className="flex items-center justify-center gap-2 bg-red-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-red-700 transition-colors shadow-sm"
              >
                <Square className="w-5 h-5" /> Stop & Finalize Stream
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="font-semibold text-slate-900">Narrative input</h3>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_TEXTS.map((sample) => (
                <button
                  key={sample.label}
                  onClick={() => {
                    setTextInput(sample.text);
                    setLanguage(sample.language);
                    setChannel(sample.channel);
                  }}
                  className="text-xs px-2.5 py-1 rounded-full border border-slate-200 bg-slate-50 text-slate-600 hover:bg-amber-50 hover:border-amber-200 transition-colors"
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            rows={5}
            placeholder="Type or paste complainant narrative..."
            className="w-full border border-slate-300 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
          />

          <button
            onClick={submitText}
            disabled={!textInput.trim()}
            className="flex items-center gap-2 bg-slate-900 text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-800 disabled:opacity-50 transition-colors"
          >
            <Activity className="w-5 h-5" /> Analyze Narrative
          </button>
        </div>
      )}

      {/* Independent Safety Flag Banner */}
      <SafetyFlags flags={safetyFlags} />

      {/* Main Results Display */}
      {(liveMessage || finalSummary) && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* SVI Gauge Card */}
            <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm space-y-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-semibold text-slate-800">
                    Normalized SVI Baseline Score
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    EMA smoothed live risk metric · {sessionId || "Session"}
                  </p>
                </div>
                {riskBand && (
                  <span className={`px-3 py-1 rounded-full border text-xs font-semibold ${getRiskStyle(riskBand)}`}>
                    {riskBand} RISK
                  </span>
                )}
              </div>

              <div className="flex items-end gap-2 my-2">
                <span className="text-5xl font-extrabold text-slate-900">
                  {sviScore !== null ? sviScore.toFixed(1) : "—"}
                </span>
                <span className="text-slate-500 mb-2 font-medium">/ 100</span>
              </div>

              <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    riskBand === "CRITICAL"
                      ? "bg-red-500"
                      : riskBand === "HIGH"
                      ? "bg-orange-500"
                      : riskBand === "MODERATE"
                      ? "bg-yellow-500"
                      : "bg-green-500"
                  }`}
                  style={{ width: `${Math.max(0, Math.min(100, sviScore || 0))}%` }}
                />
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2 text-xs border-t border-slate-100">
                <div>
                  <span className="text-slate-500">Analysis Confidence:</span>
                  <p className="font-semibold text-slate-800 mt-0.5">
                    {confidence?.overall_confidence?.toFixed(1) || "—"}% ({confidence?.confidence_rating || "LOW"})
                  </p>
                </div>
                <div>
                  <span className="text-slate-500">Signal Quality:</span>
                  <p className="font-semibold text-slate-800 mt-0.5">
                    {confidence?.signal_quality || liveMessage?.signal_quality || "GOOD"}
                  </p>
                </div>
              </div>
            </div>

            {/* Detected Indicators Table */}
            <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
              <h3 className="font-semibold text-slate-800 mb-4">
                Live Acoustic & Signal Indicators
              </h3>
              <IndicatorTable rows={indicatorRows} />
            </div>
          </div>

          {/* SVI Trajectory Timeline */}
          {mode === "voice" && (
            <SVITrendTimeline
              timeline={timelineHistory}
              currentTrend={trend}
              latencyInfo={{
                fast_path_latency_ms: liveMessage?.fast_path_latency_ms,
                heavy_path_latency_ms: liveMessage?.heavy_path_latency_ms,
                realtime_ratio: liveMessage?.realtime_ratio,
              }}
            />
          )}

          {/* Transcript & Explainability */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6 space-y-4">
              <h3 className="font-semibold text-slate-800">
                Live Transcript (PII-redacted)
              </h3>
              <div className="rounded-lg bg-slate-50 border border-slate-100 p-4 text-sm leading-6 text-slate-700 min-h-[120px] whitespace-pre-wrap font-sans">
                {transcriptText || "[Listening for caller voice...]" }
              </div>

              {nlp?.distress_categories?.length > 0 && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Distress categories
                  </p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {nlp.distress_categories.map((cat) => (
                      <span key={cat} className="px-2.5 py-1 text-xs rounded-full border border-orange-200 bg-orange-50 text-orange-700">
                        {cat}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6 space-y-4 flex flex-col justify-between">
              <div>
                <h3 className="font-semibold text-slate-800 mb-2">
                  Explainability & Human Action
                </h3>
                <p className="text-xs text-slate-500 mb-4">
                  Observational decision support indicators for NHAA operator triage.
                </p>

                <div className="space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Recommended Operator Actions
                  </p>
                  {finalSummary?.recommended_interventions?.length > 0 ? (
                    finalSummary.recommended_interventions.map((action, idx) => (
                      <div key={idx} className="flex items-start gap-2 text-xs text-slate-700 bg-amber-50 border border-amber-100 p-2.5 rounded">
                        <Send className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                        <span>{action}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-xs text-slate-500 bg-slate-50 p-3 rounded border border-slate-100">
                      Operator triage protocol: Evaluate caller safety and escalate if high distress indicators persist.
                    </div>
                  )}
                </div>
              </div>

              <div className="pt-4 border-t border-slate-100 space-y-2">
                <button
                  onClick={handleEscalateCase}
                  disabled={escalateState === "sending" || escalateState === "done"}
                  className="w-full bg-slate-900 text-white text-sm font-medium py-2.5 rounded-lg hover:bg-slate-800 transition-colors disabled:opacity-60"
                >
                  {escalateState === "sending"
                    ? "Recording Escalation..."
                    : escalateState === "done"
                    ? "✓ Escalation Logged"
                    : "Escalate for Senior Human Review"}
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Human Oversight Guardrail */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-semibold text-amber-900">
            Decision support only — Human oversight required
          </p>
          <p className="text-xs text-amber-800 mt-1">
            AI-generated SVI scores and safety indicators support human operator prioritization and do not provide medical or psychiatric diagnoses.
          </p>
        </div>
      </div>
    </div>
  );
}
