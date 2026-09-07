export default function RiskBadge({ level }) {
  const styles = {
    CRITICAL: "bg-red-100 text-red-800 border-red-300",
    HIGH: "bg-orange-100 text-orange-800 border-orange-300",
    MODERATE: "bg-yellow-100 text-yellow-800 border-yellow-300",
    LOW: "bg-green-100 text-green-800 border-green-300",
  };

  return (
    <span className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${styles[level] || styles.LOW}`}>
      {level}
    </span>
  );
}