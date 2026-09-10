import StatCard from "../components/StatCard";
import RiskBadge from "../components/RiskBadge";

export default function Dashboard({ cases, onSelectCase }) {
  const lowCount = cases.filter((item) => item.riskLevel === "LOW").length;
  const moderateCount = cases.filter((item) => item.riskLevel === "MODERATE").length;
  const highCount = cases.filter((item) => item.riskLevel === "HIGH").length;
  const criticalCount = cases.filter((item) => item.riskLevel === "CRITICAL").length;
  const avgSvi =
    cases.length > 0
      ? (
          cases.reduce((sum, item) => sum + (Number(item.sviScore) || 0), 0) /
          cases.length
        ).toFixed(1)
      : "—";

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-800">System Overview</h2>
        <p className="text-sm text-slate-500">
          Real-time victim vulnerability assessments across channels.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard label="Total Cases" count={cases.length} colorClass="text-slate-900" />
        <StatCard label="Avg SVI" count={avgSvi} colorClass="text-slate-700" />
        <StatCard label="Low Risk" count={lowCount} colorClass="text-green-600" />
        <StatCard label="Moderate Risk" count={moderateCount} colorClass="text-yellow-600" />
        <StatCard label="High Risk" count={highCount} colorClass="text-orange-600" />
        <StatCard label="Critical Risk" count={criticalCount} colorClass="text-red-600" />
      </div>

      <div className="bg-white rounded-lg border border-slate-200 shadow-sm">
        <div className="p-4 border-b border-slate-200">
          <h3 className="font-semibold text-slate-800">Recent Cases</h3>
        </div>
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
            <tr>
              <th className="p-3">Case ID</th>
              <th className="p-3">Time</th>
              <th className="p-3">Channel</th>
              <th className="p-3">Language</th>
              <th className="p-3">SVI Score</th>
              <th className="p-3">Risk Level</th>
              <th className="p-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {cases.map((item) => (
              <tr
                key={item.id}
                onClick={() => onSelectCase(item)}
                className="hover:bg-slate-50 cursor-pointer"
              >
                <td className="p-3 font-semibold text-amber-700">{item.id}</td>
                <td className="p-3 text-slate-600">{item.timestamp}</td>
                <td className="p-3 text-slate-600">{item.channel}</td>
                <td className="p-3 text-slate-600">{item.language}</td>
                <td className="p-3 font-mono font-medium">{item.sviScore} / 100</td>
                <td className="p-3"><RiskBadge level={item.riskLevel} /></td>
                <td className="p-3 text-slate-600">{item.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
