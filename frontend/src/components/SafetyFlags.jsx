import { ShieldAlert, AlertTriangle } from "lucide-react";

export default function SafetyFlags({ flags = [] }) {
  if (!flags || flags.length === 0) return null;

  return (
    <div className="bg-red-50 border-2 border-red-400 rounded-lg p-5 shadow-sm space-y-3">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center text-red-600 flex-shrink-0">
          <ShieldAlert className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <h3 className="font-bold text-red-950 text-base uppercase tracking-wide">
            Urgent Human Review Required
          </h3>
          <p className="text-xs text-red-800 font-medium mt-0.5">
            Independent safety indicator triggered — review caller disclosure immediately.
          </p>
        </div>
      </div>

      <div className="space-y-2 pt-2 border-t border-red-200">
        {flags.map((flag, idx) => (
          <div key={idx} className="bg-white border border-red-200 rounded-md p-3 text-sm flex gap-3 items-start shadow-2xs">
            <AlertTriangle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-red-900">{flag.category}</span>
                <span className="text-2xs font-bold uppercase px-2 py-0.5 rounded bg-red-100 text-red-800">
                  {flag.severity || "CRITICAL"}
                </span>
              </div>
              <p className="text-xs text-slate-700">{flag.message}</p>
              {flag.matched_terms && flag.matched_terms.length > 0 && (
                <div className="flex gap-1.5 flex-wrap pt-1">
                  <span className="text-2xs text-slate-500 font-medium">Matched terms:</span>
                  {flag.matched_terms.map((term) => (
                    <span key={term} className="text-2xs px-1.5 py-0.5 rounded bg-red-50 border border-red-200 text-red-700 font-mono">
                      "{term}"
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <p className="text-2xs text-red-700 italic">
        Note: Safety flags alert the operator for immediate priority review without altering the calibrated SVI numerical calculation.
      </p>
    </div>
  );
}
