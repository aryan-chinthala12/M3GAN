import { useState } from "react";
import RiskBadge from "../components/RiskBadge";
import SVIIndicator from "../components/SVIIndicator";
import { respondIntervention } from "../api/assessment";

export default function CaseDetails({ caseData, onBack }) {
  const [escalateState, setEscalateState] = useState("idle");

  if (!caseData) return <div className="p-6">No case selected.</div>;

  const handleEscalate = async () => {
    setEscalateState("sending");
    try {
      await respondIntervention({
        callId: caseData.id,
        operatorId: "dashboard-operator",
        actionTaken: "Escalated for Human Review",
        notes: caseData.preview || null,
      });
      setEscalateState("done");
    } catch {
      setEscalateState("error");
    }
  };

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
          <h2 className="text-2xl font-bold text-slate-900">
            Case ID: {caseData.id}
          </h2>
          <p className="text-sm text-slate-500">
            Logged at {caseData.timestamp} • Channel: {caseData.channel} •
            Language: {caseData.language}
          </p>
        </div>
        <RiskBadge level={caseData.riskLevel} />
      </div>

      {caseData.preview && (
        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm">
          <h3 className="font-semibold text-slate-800 mb-3">
            Narrative (PII-redacted)
          </h3>
          <div className="rounded-lg bg-slate-50 border border-slate-100 p-4 text-sm leading-6 text-slate-700 whitespace-pre-wrap">
            {caseData.preview}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm space-y-5">
          <div>
            <h3 className="font-semibold text-slate-800">
              Stress Vulnerability Index (SVI)
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              AI-assisted screening indicator
            </p>
          </div>

          <SVIIndicator score={caseData.sviScore} riskLevel={caseData.riskLevel} />

          {Object.keys(caseData.indicators || {}).length > 0 && (
            <div className="pt-3 border-t border-slate-100">
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                Acoustic & Behavioral Indicators
              </h4>
              {Object.entries(caseData.indicators).map(([key, val]) => (
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
          )}

          <p className="text-xs text-slate-400 pt-2">
            This assessment is intended to support authorized human review and
            is not a clinical diagnosis.
          </p>
        </div>

        <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="font-semibold text-slate-800 mb-4">
              Recommended Actions
            </h3>
            <ul className="space-y-2">
              {(caseData.recommendedActions || []).length > 0 ? (
                caseData.recommendedActions.map((action, idx) => (
                  <li
                    key={idx}
                    className="flex items-center space-x-2 text-sm text-slate-700 bg-slate-50 p-2.5 rounded border border-slate-100"
                  >
                    <span className="text-amber-600">•</span>
                    <span>{action}</span>
                  </li>
                ))
              ) : (
                <li className="text-sm text-slate-500">
                  Actions are recorded with the full assessment payload. The
                  case is in “{caseData.status}” status.
                </li>
              )}
            </ul>
          </div>

          <div className="mt-6 space-y-2">
            <button
              onClick={handleEscalate}
              disabled={escalateState === "sending" || escalateState === "done"}
              className="w-full bg-slate-900 text-white font-medium py-2 rounded-lg hover:bg-slate-800 transition-colors disabled:opacity-60"
            >
              {escalateState === "sending"
                ? "Recording..."
                : escalateState === "done"
                  ? "✓ Escalation recorded"
                  : "Escalate for Human Review"}
            </button>
            {escalateState === "error" && (
              <p className="text-xs text-red-600">
                Could not reach the backend to record the escalation.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
