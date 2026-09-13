import { useState } from "react";
import { CheckCircle2, ShieldAlert, ArrowUpRight, Square, Send } from "lucide-react";
import { respondIntervention } from "../api/assessment";

export default function OperatorTriageBar({ sessionId, currentSvi, riskBand, onStopSession, disabled }) {
  const [actionStatus, setActionStatus] = useState(null); // 'acknowledging' | 'escalating' | 'done' | 'error'
  const [lastAction, setLastAction] = useState("");

  const handleAction = async (actionLabel) => {
    setActionStatus("sending");
    setLastAction(actionLabel);

    try {
      await respondIntervention({
        callId: sessionId || "LIVE-SESSION",
        operatorId: "NHAA-OPERATOR-14566",
        actionTaken: actionLabel,
        notes: `Operator action recorded during live assessment (SVI: ${currentSvi || 0}, Risk: ${riskBand || "LOW"})`,
      });
      setActionStatus("done");
    } catch {
      setActionStatus("error");
    }
  };

  return (
    <div className="bg-slate-900 text-white p-4 rounded-xl border border-slate-800 shadow-lg space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="font-semibold text-sm text-slate-200">Human Operator Triage Console</h3>
          <p className="text-2xs text-slate-400">
            AI decision-support indicators guide operator action — medical/legal intervention decisions remain strictly human.
          </p>
        </div>

        {actionStatus === "done" && (
          <span className="text-xs bg-green-900/60 text-green-300 border border-green-700/50 px-3 py-1 rounded-md flex items-center gap-1.5 font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" /> Action Logged: {lastAction}
          </span>
        )}

        {actionStatus === "error" && (
          <span className="text-xs bg-red-900/60 text-red-300 border border-red-700/50 px-3 py-1 rounded-md font-medium">
            Could not record action with backend case store.
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <button
          onClick={() => handleAction("Acknowledge & Continue Triage")}
          disabled={disabled || actionStatus === "sending"}
          className="flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold py-2.5 px-3 rounded-lg transition-colors disabled:opacity-50"
        >
          <CheckCircle2 className="w-4 h-4 text-green-400" />
          Acknowledge & Continue
        </button>

        <button
          onClick={() => handleAction("Flagged for Psychological First Aid (PFA)")}
          disabled={disabled || actionStatus === "sending"}
          className="flex items-center justify-center gap-2 bg-amber-950/60 hover:bg-amber-900/80 text-amber-200 border border-amber-800/60 text-xs font-semibold py-2.5 px-3 rounded-lg transition-colors disabled:opacity-50"
        >
          <ShieldAlert className="w-4 h-4 text-amber-400" />
          Flag for PFA Support
        </button>

        <button
          onClick={() => handleAction("Escalated to Senior Trauma Supervisor")}
          disabled={disabled || actionStatus === "sending"}
          className="flex items-center justify-center gap-2 bg-red-950/80 hover:bg-red-900 text-red-100 border border-red-800 text-xs font-semibold py-2.5 px-3 rounded-lg transition-colors disabled:opacity-50"
        >
          <ArrowUpRight className="w-4 h-4 text-red-400" />
          Escalate to Supervisor
        </button>

        <button
          onClick={onStopSession}
          disabled={disabled && !onStopSession}
          className="flex items-center justify-center gap-2 bg-slate-100 hover:bg-white text-slate-900 text-xs font-bold py-2.5 px-3 rounded-lg transition-colors"
        >
          <Square className="w-4 h-4 text-red-600 fill-red-600" />
          End Assessment
        </button>
      </div>
    </div>
  );
}
