import RiskBadge from "./RiskBadge";

export default function CaseTable({ cases, onSelectCase }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
          <tr>
            <th className="p-3 font-semibold">Case ID</th>
            <th className="p-3 font-semibold">Time</th>
            <th className="p-3 font-semibold">Channel</th>
            <th className="p-3 font-semibold">Language</th>
            <th className="p-3 font-semibold">SVI Score</th>
            <th className="p-3 font-semibold">Risk Level</th>
            <th className="p-3 font-semibold">Status</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-slate-100">
          {cases.map((item) => (
            <tr
              key={item.id}
              onClick={() => onSelectCase(item)}
              className="hover:bg-slate-50 cursor-pointer transition-colors"
            >
              <td className="p-3">
                <span className="font-semibold text-amber-700">
                  {item.id}
                </span>
              </td>

              <td className="p-3 text-slate-600">
                {item.timestamp}
              </td>

              <td className="p-3 text-slate-600">
                {item.channel}
              </td>

              <td className="p-3 text-slate-600">
                {item.language}
              </td>

              <td className="p-3">
                <span className="font-mono font-semibold text-slate-800">
                  {item.sviScore}
                </span>
                <span className="text-slate-400"> / 100</span>
              </td>

              <td className="p-3">
                <RiskBadge level={item.riskLevel} />
              </td>

              <td className="p-3 text-slate-600">
                {item.status}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}