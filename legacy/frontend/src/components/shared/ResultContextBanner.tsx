import { useAppStore } from "@/stores/appStore";
import { AlertTriangle, History } from "lucide-react";

function formatTimestamp(ms: number | null): string {
  if (ms == null) return "—";
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/**
 * Thin banner shown at the top of Displacement / Post-Processing pages.
 * Tells the user which ROI version the currently displayed results came from,
 * and flags staleness when the ROI has been edited since the last run.
 */
export function ResultContextBanner() {
  const hasResults = useAppStore((s) => s.hasResults);
  const roiVersion = useAppStore((s) => s.roiVersion);
  const lastRunRoiVersion = useAppStore((s) => s.lastRunRoiVersion);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const lastRunModel = useAppStore((s) => s.lastRunModel);
  const lastRunAt = useAppStore((s) => s.lastRunAt);
  const referenceFrame = useAppStore((s) => s.referenceFrame);

  if (!hasResults) return null;

  const roiChanged = roiVersion > lastRunRoiVersion;
  // Empty lastRunModel means "no run yet tracked" (legacy results) — don't
  // falsely flag those as stale.
  const modelChanged = lastRunModel !== "" && selectedModel !== lastRunModel;
  const stale = roiChanged || modelChanged;

  if (stale) {
    const reasons: string[] = [];
    if (roiChanged) reasons.push("ROI modified");
    if (modelChanged) reasons.push("model changed");
    return (
      <div
        role="status"
        className="flex items-center gap-2 px-3 py-1.5 bg-amber-500/15 border-b border-amber-500/40 text-amber-300 text-[11px]"
      >
        <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
        <span className="font-medium">
          {reasons.join(" + ")} since last run.
        </span>
        <span className="text-amber-300/80">
          Displayed results may be out of date — re-run on the ROI page to
          refresh.
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--secondary)]/40 border-b border-[var(--border)] text-[var(--muted-foreground)] text-[11px]">
      <History className="w-3.5 h-3.5 shrink-0 opacity-70" />
      <span>
        Showing results for <b className="text-[var(--foreground)]">ROI v{lastRunRoiVersion}</b>
        {" · "}reference frame <b className="text-[var(--foreground)]">{referenceFrame + 1}</b>
        {lastRunAt != null && (
          <>
            {" · "}
            <span title={new Date(lastRunAt).toISOString()}>
              {formatTimestamp(lastRunAt)}
            </span>
          </>
        )}
      </span>
    </div>
  );
}
