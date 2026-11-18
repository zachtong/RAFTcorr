import { useAppStore } from "@/stores/appStore";
import { runProcessing, pauseProcessing, resumeProcessing, stopProcessing } from "@/api/processing";
import { Play, Pause, Square, RotateCcw } from "lucide-react";

export function RunControl() {
  const active = useAppStore((s) => s.processingActive);
  const paused = useAppStore((s) => s.processingPaused);
  const progress = useAppStore((s) => s.processingProgress);
  const current = useAppStore((s) => s.processingCurrent);
  const total = useAppStore((s) => s.processingTotal);
  const roiConfirmed = useAppStore((s) => s.roiConfirmed);

  const handleRun = async () => {
    if (active) return;
    try {
      await runProcessing();
    } catch (e: any) {
      console.error("Failed to start processing:", e);
    }
  };

  const handlePause = async () => {
    try {
      await pauseProcessing();
    } catch (e: any) {
      console.error("Failed to pause processing:", e);
    }
  };

  const handleResume = async () => {
    try {
      await resumeProcessing();
    } catch (e: any) {
      console.error("Failed to resume processing:", e);
    }
  };

  const handleCancel = async () => {
    try {
      await stopProcessing();
    } catch (e: any) {
      console.error("Failed to cancel processing:", e);
    }
  };

  const statusText = active
    ? paused
      ? "Paused"
      : "Processing..."
    : "Ready";

  const statusColor = active
    ? paused
      ? "bg-amber-500"
      : "bg-yellow-500 animate-pulse"
    : "bg-green-500";

  return (
    <div className="border-t border-[var(--border)] p-2 space-y-2">
      {/* Progress info */}
      <div className="flex items-center gap-1 text-[10px] text-[var(--muted-foreground)]">
        <span>
          Frame: {current} / {total}
        </span>
        <span className="ml-auto">{progress.toFixed(0)}%</span>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-[var(--secondary)] rounded-full h-1">
        <div
          className={`h-1 rounded-full transition-all duration-200 ${
            paused ? "bg-amber-500" : "bg-[var(--primary)]"
          }`}
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Buttons: different states */}
      {!active ? (
        /* Idle: show Run button */
        <button
          onClick={handleRun}
          disabled={!roiConfirmed}
          className="w-full flex items-center justify-center gap-1 bg-[var(--primary)] hover:bg-[var(--primary)]/90 text-white rounded py-1.5 text-[11px] disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Play size={12} /> Run
        </button>
      ) : (
        /* Active: show Pause/Resume + Cancel */
        <div className="flex gap-2">
          {paused ? (
            <button
              onClick={handleResume}
              className="flex-1 flex items-center justify-center gap-1 bg-[var(--primary)] hover:bg-[var(--primary)]/90 text-white rounded py-1.5 text-[11px]"
            >
              <RotateCcw size={12} /> Continue
            </button>
          ) : (
            <button
              onClick={handlePause}
              className="flex-1 flex items-center justify-center gap-1 bg-amber-600 hover:bg-amber-600/90 text-white rounded py-1.5 text-[11px]"
            >
              <Pause size={12} /> Pause
            </button>
          )}
          <button
            onClick={handleCancel}
            className="flex-1 flex items-center justify-center gap-1 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 text-red-400 rounded py-1.5 text-[11px]"
          >
            <Square size={12} /> Cancel
          </button>
        </div>
      )}

      {/* Status indicator */}
      <div className="flex items-center gap-1">
        <div className={`w-2 h-2 rounded-full ${statusColor}`} />
        <span className="text-[10px] text-[var(--muted-foreground)]">
          {statusText}
        </span>
      </div>
    </div>
  );
}
