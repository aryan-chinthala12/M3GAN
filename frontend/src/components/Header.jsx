import { Shield } from "lucide-react";

export default function Header() {
  return (
    <header className="bg-slate-900 text-white p-4 flex items-center justify-between border-b border-slate-800">
      <div className="flex items-center space-x-3">
        <Shield className="h-8 w-8 text-amber-500" />
        <div>
          <h1 className="text-lg font-bold leading-tight">National Helpline Against Atrocities</h1>
          <p className="text-xs text-slate-400">Department of Social Justice & Empowerment • GoI</p>
        </div>
      </div>
      <div className="text-right">
        <span className="text-xs font-mono bg-amber-500/20 text-amber-300 px-2.5 py-1 rounded border border-amber-500/30">
          Toll Free: 14566
        </span>
      </div>
    </header>
  );
}