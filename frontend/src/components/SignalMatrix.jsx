import { Activity, Mic, Volume2, Pause, HeartPulse, Shield, AlertCircle } from "lucide-react";

function formatSignal(val, unit = "", fallback = "Unavailable") {
  if (val === null || val === undefined) return fallback;
  if (typeof val === "boolean") return val ? "Reliable" : "Unreliable";
  if (typeof val === "number") {
    if (!Number.isFinite(val)) return fallback;
    return `${Number.isInteger(val) ? val : val.toFixed(2)}${unit}`;
  }
  return String(val).replaceAll("_", " ").toUpperCase();
}

export default function SignalMatrix({ acoustics = {}, emotion = {}, indicators = {}, signalQuality = "GOOD" }) {
  const isPitchReliable = acoustics.pitch_reliable ?? true;
  const isSignalUnreliable = signalQuality === "UNRELIABLE";

  const matrixItems = [
    {
      label: "Speech Emotion",
      value: formatSignal(emotion.top_emotion, "", "Analyzing..."),
      detail: emotion.confidence ? `${emotion.confidence.toFixed(1)}% conf.` : null,
      icon: HeartPulse,
      color: "text-purple-600 bg-purple-50 border-purple-100",
    },
    {
      label: "Pitch Volatility",
      value: isPitchReliable && !isSignalUnreliable ? formatSignal(acoustics.pitch_volatility, " st") : "Unavailable",
      detail: !isPitchReliable ? "Insufficient voiced speech" : null,
      icon: Activity,
      color: isPitchReliable ? "text-amber-600 bg-amber-50 border-amber-100" : "text-slate-400 bg-slate-50 border-slate-100",
    },
    {
      label: "Voiced Ratio",
      value: formatSignal(acoustics.voiced_ratio ? acoustics.voiced_ratio * 100 : null, "%"),
      detail: "Active speech frame fraction",
      icon: Mic,
      color: "text-blue-600 bg-blue-50 border-blue-100",
    },
    {
      label: "Pause Ratio",
      value: formatSignal(acoustics.pause_ratio ? acoustics.pause_ratio * 100 : null, "%"),
      detail: "Silence / hesitation fraction",
      icon: Pause,
      color: "text-orange-600 bg-orange-50 border-orange-100",
    },
    {
      label: "RMS Energy",
      value: formatSignal(acoustics.rms_energy),
      detail: acoustics.energy_variation ? `Var: ${acoustics.energy_variation.toFixed(1)} dB` : null,
      icon: Volume2,
      color: "text-emerald-600 bg-emerald-50 border-emerald-100",
    },
    {
      label: "Threat Level",
      value: formatSignal(indicators.threat_level, "", "Scanning..."),
      detail: indicators.distress_categories?.length ? `${indicators.distress_categories.length} categories` : "No threat categories",
      icon: Shield,
      color: "text-red-600 bg-red-50 border-red-100",
    },
  ];

  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-slate-800 text-sm">Live Multimodal Signal Matrix</h3>
          <p className="text-2xs text-slate-500 mt-0.5">
            Real-time acoustic prosody, speech emotion, and lexical indicators
          </p>
        </div>

        <span
          className={`text-2xs font-bold uppercase px-2.5 py-1 rounded border ${
            signalQuality === "GOOD"
              ? "bg-green-50 text-green-700 border-green-200"
              : signalQuality === "DEGRADED"
              ? "bg-amber-50 text-amber-700 border-amber-200"
              : "bg-red-50 text-red-700 border-red-200"
          }`}
        >
          Quality: {signalQuality}
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        {matrixItems.map((item) => {
          const Icon = item.icon;
          const isUnavailable = item.value === "Unavailable" || item.value === "Analyzing...";
          return (
            <div key={item.label} className="p-3 rounded-lg border border-slate-100 bg-slate-50/50 space-y-1">
              <div className="flex items-center gap-2">
                <div className={`p-1.5 rounded-md border ${item.color}`}>
                  <Icon className="w-3.5 h-3.5" />
                </div>
                <span className="text-2xs font-medium text-slate-500 uppercase tracking-wide">
                  {item.label}
                </span>
              </div>

              <div className="pt-1">
                <p className={`text-base font-bold ${isUnavailable ? "text-slate-400 italic text-sm" : "text-slate-800"}`}>
                  {item.value}
                </p>
                {item.detail && <p className="text-3xs text-slate-400 mt-0.5">{item.detail}</p>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
