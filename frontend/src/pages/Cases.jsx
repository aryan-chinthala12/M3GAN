import { Search } from "lucide-react";
import { useState } from "react";
import CaseTable from "../components/CaseTable";

export default function Cases({ cases, onSelectCase }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");

  const filteredCases = cases.filter((item) => {
    const matchesSearch = item.id
      .toLowerCase()
      .includes(searchTerm.toLowerCase());

    const matchesRisk =
      riskFilter === "ALL" || item.riskLevel === riskFilter;

    return matchesSearch && matchesRisk;
  });

  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div>
        <h2 className="text-xl font-bold text-slate-800">
          All Cases
        </h2>

        <p className="text-sm text-slate-500">
          View and manage incoming NHAA cases and vulnerability assessments.
        </p>
      </div>

      {/* Search + Filter */}
      <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
        <div className="flex flex-col md:flex-row gap-3">
          
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />

            <input
              type="text"
              placeholder="Search by Case ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
            />
          </div>

          {/* Risk Filter */}
          <select
            value={riskFilter}
            onChange={(e) => setRiskFilter(e.target.value)}
            className="px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-amber-500"
          >
            <option value="ALL">All Risk Levels</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MODERATE">Moderate</option>
            <option value="LOW">Low</option>
          </select>
        </div>
      </div>

      {/* Results */}
      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <h3 className="font-semibold text-slate-800">
            Cases
          </h3>

          <span className="text-xs text-slate-500">
            {filteredCases.length} case
            {filteredCases.length !== 1 ? "s" : ""}
          </span>
        </div>

        {filteredCases.length > 0 ? (
          <CaseTable
            cases={filteredCases}
            onSelectCase={onSelectCase}
          />
        ) : (
          <div className="p-10 text-center text-sm text-slate-500">
            No cases found.
          </div>
        )}
      </div>
    </div>
  );
}