import { useCallback, useEffect, useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import Cases from "./pages/Cases";
import CaseDetails from "./pages/CaseDetails";
import LiveAssessment from "./pages/LiveAssessment";
import { mockCases } from "./data/mockCases";
import { getCases } from "./api/assessment";

const DEMO_NOTE =
  "Showing demo data — the SVI backend could not be reached. Run the FastAPI server for live cases.";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [selectedCase, setSelectedCase] = useState(null);
  const [cases, setCases] = useState(mockCases);
  const [isDemo, setIsDemo] = useState(true);

  const refreshCases = useCallback(async () => {
    try {
      const backendCases = await getCases(100);
      if (Array.isArray(backendCases) && backendCases.length > 0) {
        // Map backend CaseSummary to the shape the UI already consumes.
        setCases(
          backendCases.map((c) => ({
            id: c.case_id,
            timestamp: c.created_at?.replace("T", " ").slice(0, 16) + " UTC",
            channel: c.channel === "audio" ? "Voice" : c.channel,
            language: c.language,
            sviScore: c.svi_score,
            riskLevel: c.risk_band,
            status: c.status,
            preview: c.preview,
            indicators: {},
            recommendedActions: [],
          }))
        );
        setIsDemo(false);
      } else {
        setCases(mockCases);
        setIsDemo(true);
      }
    } catch {
      setCases(mockCases);
      setIsDemo(true);
    }
  }, []);

  useEffect(() => {
    refreshCases();

    // Poll every 20s so newly-assessed cases appear on the dashboard.
    const interval = setInterval(refreshCases, 20000);
    return () => clearInterval(interval);
  }, [refreshCases]);

  // Refresh when returning to dashboard/cases after a live assessment.
  const handleSelectTab = (tab) => {
    setActiveTab(tab);
    if (tab === "dashboard" || tab === "cases") {
      refreshCases();
    }
  };

  const handleSelectCase = (c) => {
    setSelectedCase(c);
    setActiveTab("case-details");
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-100 text-slate-900">
      <Header />
      {isDemo && (
        <div className="bg-amber-100 border-b border-amber-200 text-amber-900 text-xs px-6 py-2">
          {DEMO_NOTE}
        </div>
      )}
      <div className="flex flex-1">
        <Sidebar activeTab={activeTab} setActiveTab={handleSelectTab} />
        <main className="flex-1 overflow-y-auto">
          {activeTab === "dashboard" && (
            <Dashboard cases={cases} onSelectCase={handleSelectCase} />
          )}
          {activeTab === "cases" && (
            <Cases cases={cases} onSelectCase={handleSelectCase} />
          )}
          {activeTab === "case-details" && (
            <CaseDetails
              caseData={selectedCase}
              onBack={() => setActiveTab("cases")}
            />
          )}
          {activeTab === "live" && (
            <LiveAssessment onAssessed={refreshCases} />
          )}
        </main>
      </div>
    </div>
  );
}
