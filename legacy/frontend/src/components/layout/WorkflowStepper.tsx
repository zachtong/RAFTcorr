import { Crosshair, Move, BarChart3 } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAppStore } from "@/stores/appStore";
import { cn } from "@/lib/utils";

const steps = [
  { path: "/", label: "ROI Selection", icon: Crosshair },
  { path: "/displacement", label: "Displacement", icon: Move },
  { path: "/post-processing", label: "Post-Processing", icon: BarChart3 },
] as const;

export function WorkflowStepper() {
  const location = useLocation();
  const navigate = useNavigate();
  const roiConfirmed = useAppStore((s) => s.roiConfirmed);
  const hasResults = useAppStore((s) => s.hasResults);

  const isEnabled = (idx: number) => {
    if (idx === 0) return true;
    if (idx === 1) return roiConfirmed;
    if (idx === 2) return hasResults;
    return false;
  };

  return (
    <div className="flex items-center gap-1">
      {steps.map((step, idx) => {
        const active = location.pathname === step.path;
        const enabled = isEnabled(idx);
        const Icon = step.icon;
        return (
          <button
            key={step.path}
            onClick={() => enabled && navigate(step.path)}
            disabled={!enabled}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors",
              active
                ? "bg-[var(--primary)] text-white"
                : enabled
                  ? "text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)]"
                  : "text-[var(--muted-foreground)]/40 cursor-not-allowed"
            )}
          >
            <Icon size={14} />
            {step.label}
          </button>
        );
      })}
    </div>
  );
}
