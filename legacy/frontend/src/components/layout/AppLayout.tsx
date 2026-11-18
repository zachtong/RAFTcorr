import { Outlet } from "react-router-dom";
import { WorkflowStepper } from "./WorkflowStepper";

export function AppLayout() {
  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-3 py-1.5 border-b border-[var(--border)] bg-[var(--card)]">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold tracking-tight">RAFTcorr</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[var(--muted-foreground)]">
            v2.0
          </span>
        </div>
        <WorkflowStepper />
        <div className="w-24" /> {/* Spacer for balance */}
      </header>

      {/* Content */}
      <Outlet />
    </div>
  );
}
