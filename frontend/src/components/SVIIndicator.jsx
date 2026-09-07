export default function SVIIndicator({ score, riskLevel }) {
  const getBarColor = () => {
    if (riskLevel === "CRITICAL") return "bg-red-500";
    if (riskLevel === "HIGH") return "bg-orange-500";
    if (riskLevel === "MODERATE") return "bg-yellow-500";
    return "bg-green-500";
  };

  return (
    <div className="space-y-3">
      <div className="flex items-end gap-2">
        <span className="text-4xl font-extrabold text-slate-900">
          {score}
        </span>

        <span className="text-slate-500 font-medium mb-1">
          / 100
        </span>
      </div>

      <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${getBarColor()}`}
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );
}