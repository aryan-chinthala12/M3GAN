import { TrendingUp, TrendingDown, Minus, HelpCircle, Activity } from "lucide-react";

const TREND_CONFIG = {
  INCREASING: {
    label: "Increasing",
    icon: TrendingUp,
    colorClass: "bg-red-500/10 text-red-700 border-red-200",
  },
  DECREASING: {
    label: "Decreasing",
    icon: TrendingDown,
    colorClass: "bg-green-500/10 text-green-700 border-green-200",
  },
  STABLE: {
    label: "Stable",
    icon: Minus,
    colorClass: "bg-slate-100 text-slate-700 border-slate-200",
  },
  INSUFFICIENT_DATA: {
    label: "Gathering Stream Data",
    icon: HelpCircle,
    colorClass: "bg-amber-50 text-amber-700 border-amber-200",
  },
};

export default function SVITrendTimeline({ timeline = [], currentTrend = "INSUFFICIENT_DATA", latencyInfo = null }) {
  const trendInfo = TREND_CONFIG[currentTrend] || TREND_CONFIG.INSUFFICIENT_DATA;
  const IconComponent = trendInfo.icon;

  const maxPoints = 40;
  const displayPoints = timeline.slice(-maxPoints);

  const svgWidth = 500;
  const svgHeight = 90;

  const pointsPath =
    displayPoints.length > 1
      ? displayPoints
          .map((item, idx) => {
            const x = (idx / (displayPoints.length - 1)) * svgWidth;
            const score = Math.max(0, Math.min(100, item.smoothed_svi ?? item.raw_svi ?? 0));
            const y = svgHeight - (score / 100) * (svgHeight - 16) - 8;
            return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(" ")
      : "";

  const areaPath =
    displayPoints.length > 1
      ? `${pointsPath} L ${svgWidth},${svgHeight} L 0,${svgHeight} Z`
      : "";

  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-amber-600" />
            <h3 className="font-semibold text-slate-800 text-sm">Real-Time SVI Trajectory & Trend Timeline</h3>
          </div>
          <p className="text-2xs text-slate-500 mt-0.5">
            Dynamic Exponential Moving Average (EMA $\alpha=0.35$) score history over stream
          </p>
        </div>

        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold ${trendInfo.colorClass}`}>
          <IconComponent className="w-3.5 h-3.5" />
          <span>{trendInfo.label}</span>
        </div>
      </div>

      {displayPoints.length > 1 ? (
        <div className="relative w-full overflow-hidden bg-slate-900 rounded-lg p-3 border border-slate-800">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-24 overflow-visible">
            <defs>
              <linearGradient id="sviGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.4" />
                <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Threshold reference lines */}
            <line x1="0" y1="18" x2={svgWidth} y2="18" stroke="#ef4444" strokeDasharray="3,3" strokeOpacity="0.4" />
            <text x="5" y="15" fill="#ef4444" fontSize="8" opacity="0.6">75 (CRITICAL)</text>

            <line x1="0" y1="42" x2={svgWidth} y2="42" stroke="#f97316" strokeDasharray="3,3" strokeOpacity="0.4" />
            <text x="5" y="39" fill="#f97316" fontSize="8" opacity="0.6">50 (HIGH)</text>

            <line x1="0" y1="66" x2={svgWidth} y2="66" stroke="#eab308" strokeDasharray="3,3" strokeOpacity="0.4" />
            <text x="5" y="63" fill="#eab308" fontSize="8" opacity="0.6">25 (MODERATE)</text>

            {/* Area Fill */}
            <path d={areaPath} fill="url(#sviGradient)" />

            {/* Trajectory Stroke */}
            <path d={pointsPath} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

            {/* Current point pulsing dot */}
            {displayPoints.length > 0 && (() => {
              const lastIdx = displayPoints.length - 1;
              const lastX = svgWidth;
              const lastScore = Math.max(0, Math.min(100, displayPoints[lastIdx].smoothed_svi ?? 0));
              const lastY = svgHeight - (lastScore / 100) * (svgHeight - 16) - 8;
              return (
                <g>
                  <circle cx={lastX} cy={lastY} r="5" fill="#fbbf24" className="animate-ping" />
                  <circle cx={lastX} cy={lastY} r="3" fill="#ffffff" />
                </g>
              );
            })()}
          </svg>
        </div>
      ) : (
        <div className="h-24 flex flex-col items-center justify-center text-xs text-slate-400 border border-dashed border-slate-200 rounded-lg bg-slate-50/50 space-y-1">
          <Activity className="w-5 h-5 text-slate-300 animate-pulse" />
          <span>Real-time timeline accumulating incoming audio stream...</span>
        </div>
      )}

      {latencyInfo && (
        <div className="flex flex-wrap justify-between items-center text-2xs text-slate-500 pt-1 border-t border-slate-100 font-mono">
          <span>Fast VAD: <strong className="text-slate-700">{latencyInfo.fast_path_latency_ms || 0}ms</strong></span>
          <span>Heavy ML: <strong className="text-slate-700">{latencyInfo.heavy_path_latency_ms || 0}ms</strong></span>
          <span>End-to-End: <strong className="text-slate-700">{latencyInfo.end_to_end_latency_ms || 0}ms</strong></span>
          <span>Realtime Ratio: <strong className={(latencyInfo.realtime_ratio || 0) < 1.0 ? "text-green-600" : "text-amber-600"}>{(latencyInfo.realtime_ratio || 0).toFixed(2)}x</strong></span>
        </div>
      )}
    </div>
  );
}
