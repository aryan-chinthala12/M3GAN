import RiskBadge from "../components/RiskBadge";
import SVIIndicator from "../components/SVIIndicator";

export default function CaseDetails({ caseData, onBack }) {
  if (!caseData) return <div className="p-6">No case selected.</div>;

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <button
        onClick={onBack}
        className="text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
      >
        ← Back to Cases
      </button>

      <div className="flex items-center justify-between border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Case ID: {caseData.id}</h2>
          <p className="text-sm text-slate-500">Logged at {caseData.timestamp} • Channel: {caseData.channel}</p>
        </div>
        <RiskBadge level={caseData.riskLevel} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SVI Assessment */}
<div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm space-y-5">
  <div>
    <h3 className="font-semibold text-slate-800">
      Stress Vulnerability Index (SVI)
    </h3>

    <p className="text-xs text-slate-500 mt-1">
      AI-assisted screening indicator
    </p>
  </div>

  <SVIIndicator
    score={caseData.sviScore}
    riskLevel={caseData.riskLevel}
  />

  <div className="pt-3 border-t border-slate-100">
    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
      Acoustic & Behavioral Indicators
    </h4>

    {Object.entries(caseData.indicators || {}).map(([key, val]) => (
      <div
        key={key}
        className="flex justify-between items-center text-sm py-2 border-b border-slate-50"
      >
        <span className="capitalize text-slate-600">
          {key.replace(/([A-Z])/g, " $1")}
        </span>

        <RiskBadge level={val} />
      </div>
    ))}
  </div>

  <p className="text-xs text-slate-400 pt-2">
    This assessment is intended to support authorized human review
    and is not a clinical diagnosis.
  </p>
</div>

        {/* Support Actions */}
        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="font-semibold text-slate-800 mb-4">Recommended Actions</h3>
            <ul className="space-y-2">
              {(caseData.recommendedActions || []).map((action, idx) => (
                <li key={idx} className="flex items-center space-x-2 text-sm text-slate-700 bg-slate-50 p-2.5 rounded border border-slate-100">
                  <span className="text-amber-600">•</span>
                  <span>{action}</span>
                </li>
              ))}
            </ul>
          </div>
          <button className="w-full mt-6 bg-slate-900 text-white font-medium py-2 rounded-lg hover:bg-slate-800 transition-colors">
            Escalate for Human Review
          </button>
        </div>
      </div>
    </div>
  );
}