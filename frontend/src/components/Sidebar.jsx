import { LayoutDashboard, PhoneCall, Radio, FileText, Settings } from "lucide-react";

export default function Sidebar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "cases", label: "Cases", icon: PhoneCall },
    { id: "live", label: "Live Assessment", icon: Radio },
  ];

  return (
    <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col justify-between border-r border-slate-800">
      <nav className="p-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive ? "bg-amber-600 text-white" : "hover:bg-slate-800 hover:text-white"
              }`}
            >
              <Icon className="h-5 w-5" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="p-4 border-t border-slate-800 text-xs text-slate-500">
        AI Victim Vulnerability Module v1.0
      </div>
    </aside>
  );
}