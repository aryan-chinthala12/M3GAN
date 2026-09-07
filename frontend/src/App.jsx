import { useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import Cases from "./pages/Cases";
import CaseDetails from "./pages/CaseDetails";
import LiveAssessment from "./pages/LiveAssessment";
import { mockCases } from "./data/mockCases";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [selectedCase, setSelectedCase] = useState(null);

  const handleSelectCase = (c) => {
    setSelectedCase(c);
    setActiveTab("case-details");
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-100 text-slate-900">
      <Header />
      <div className="flex flex-1">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="flex-1 overflow-y-auto">
          {activeTab === "dashboard" && (
            <Dashboard cases={mockCases} onSelectCase={handleSelectCase} />
          )}
          {activeTab === "cases" && (
            <Cases cases={mockCases} onSelectCase={handleSelectCase} />
          )}
          {activeTab === "case-details" && (
            <CaseDetails 
              caseData={selectedCase} 
              onBack={() => setActiveTab("cases")} 
            />
          )}
          {activeTab === "live" && <LiveAssessment />}
        </main>
      </div>
    </div>
  );
}