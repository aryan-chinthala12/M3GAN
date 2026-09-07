export default function StatCard({ label, count, colorClass }) {
  return (
    <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${colorClass}`}>{count}</p>
    </div>
  );
}