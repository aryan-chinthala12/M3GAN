import { useEffect, useRef, useState } from "react";
import {
  Mic,
  Square,
  ShieldAlert,
  Activity,
  Languages,
  Clock,
} from "lucide-react";

export default function LiveAssessment() {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [language, setLanguage] = useState("Hindi");

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startAssessment = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.start();

      setIsRecording(true);
      setDuration(0);

      timerRef.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      console.error("Microphone access error:", error);
      alert("Microphone access was denied or is unavailable.");
    }
  };

  const stopAssessment = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    clearInterval(timerRef.current);
    setIsRecording(false);
  };

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;

    return `${String(minutes).padStart(2, "0")}:${String(
      remainingSeconds
    ).padStart(2, "0")}`;
  };

  return (
    <div className="p-6 space-y-6 max-w-6xl">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">
          Live Victim Assessment
        </h2>

        <p className="text-sm text-slate-500 mt-1">
          AI-assisted real-time stress and vulnerability screening
        </p>
      </div>

      {/* Recording Section */}
      <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          
          {/* Recording Status */}
          <div className="flex items-center gap-4">
            <div
              className={`w-14 h-14 rounded-full flex items-center justify-center ${
                isRecording
                  ? "bg-red-100 text-red-600"
                  : "bg-slate-100 text-slate-600"
              }`}
            >
              {isRecording ? (
                <Activity className="w-7 h-7 animate-pulse" />
              ) : (
                <Mic className="w-7 h-7" />
              )}
            </div>

            <div>
              <h3 className="font-semibold text-slate-900">
                {isRecording
                  ? "Assessment in progress"
                  : "Ready for assessment"}
              </h3>

              <p className="text-sm text-slate-500">
                {isRecording
                  ? "Listening to incoming voice..."
                  : "Start the assessment when the complainant is ready."}
              </p>
            </div>
          </div>

          {/* Duration */}
          <div className="flex items-center gap-2 text-slate-600">
            <Clock className="w-5 h-5" />
            <span className="font-mono text-lg">
              {formatDuration(duration)}
            </span>
          </div>

          {/* Start / Stop */}
          {!isRecording ? (
            <button
              onClick={startAssessment}
              className="flex items-center justify-center gap-2 bg-slate-900 text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-800 transition-colors"
            >
              <Mic className="w-5 h-5" />
              Start Assessment
            </button>
          ) : (
            <button
              onClick={stopAssessment}
              className="flex items-center justify-center gap-2 bg-red-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-red-700 transition-colors"
            >
              <Square className="w-5 h-5" />
              Stop Assessment
            </button>
          )}
        </div>
      </div>

      {/* Assessment Information */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Language */}
        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <Languages className="w-5 h-5 text-amber-600" />
            <h3 className="font-semibold text-slate-800">Language</h3>
          </div>

          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm bg-white"
          >
            <option>Hindi</option>
            <option>English</option>
            <option>Marathi</option>
            <option>Bengali</option>
            <option>Tamil</option>
            <option>Telugu</option>
            <option>Kannada</option>
          </select>
        </div>

        {/* Audio Status */}
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

        {/* System Status */}
        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <Activity className="w-5 h-5 text-amber-600" />
            <h3 className="font-semibold text-slate-800">AI Status</h3>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-green-500 rounded-full" />
            <span className="text-sm text-slate-600">
              Assessment engine ready
            </span>
          </div>
        </div>
      </div>

      {/* Assessment Results */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SVI */}
        <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
          <h3 className="font-semibold text-slate-800">
            Stress Vulnerability Index
          </h3>

          <p className="text-xs text-slate-500 mt-1">
            AI-assisted screening indicator
          </p>

          <div className="mt-6 flex items-end gap-2">
            <span className="text-5xl font-extrabold text-slate-900">
              --
            </span>
            <span className="text-slate-500 mb-2">/ 100</span>
          </div>

          <div className="mt-4 w-full h-3 bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full w-0 bg-slate-300 rounded-full" />
          </div>

          <div className="mt-4 inline-flex px-3 py-1 rounded-full border border-slate-300 bg-slate-50 text-slate-500 text-xs font-semibold">
            Awaiting assessment
          </div>
        </div>

        {/* Indicators */}
        <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
          <h3 className="font-semibold text-slate-800">
            Detected Indicators
          </h3>

          <div className="mt-5 space-y-3">
            {[
              ["Fear", "—"],
              ["Anxiety", "—"],
              ["Speech Hesitation", "—"],
              ["Pitch Instability", "—"],
              ["Threat Detection", "—"],
            ].map(([label, value]) => (
              <div
                key={label}
                className="flex justify-between items-center py-2 border-b border-slate-100"
              >
                <span className="text-sm text-slate-600">{label}</span>
                <span className="text-sm font-semibold text-slate-400">
                  {value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Safety Notice */}
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