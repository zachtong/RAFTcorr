import { useEffect, useCallback, useRef } from "react";
import { useRoiStore } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";
import { getFramesRoiStatus, clearAllFrameRois } from "@/api/roi";
import { Trash2 } from "lucide-react";

/**
 * pyALDIC-style frame list for per-frame ROI editing.
 *
 * Three columns:  #  |  Filename  |  ROI Status Button
 *
 * Status buttons:
 *   "Edit" (green)  — frame has ROI
 *   "Need" (red)    — reference frame, ROI required but missing
 *   "Add"  (gray)   — non-reference frame, ROI optional
 *
 * Reference (key) frames get a subtle accent background row highlight.
 */
export function FrameRoiSelector() {
  const imageFiles = useAppStore((s) => s.imageFiles);
  const keyFrames = useAppStore((s) => s.keyFrames);
  const keyFrameMode = useAppStore((s) => s.keyFrameMode);
  const keyFrameInterval = useAppStore((s) => s.keyFrameInterval);
  const mode = useAppStore((s) => s.mode);
  const editingFrameIdx = useRoiStore((s) => s.editingFrameIdx);
  const setEditingFrameIdx = useRoiStore((s) => s.setEditingFrameIdx);
  const framesWithRoi = useRoiStore((s) => s.framesWithRoi);
  const setFramesWithRoi = useRoiStore((s) => s.setFramesWithRoi);
  const setMaskUrl = useRoiStore((s) => s.setMaskUrl);

  const listRef = useRef<HTMLDivElement>(null);
  const totalFrames = imageFiles.length;

  // ── Fetch ROI status ──
  const refreshStatus = useCallback(async () => {
    if (totalFrames === 0) return;
    try {
      const status = await getFramesRoiStatus();
      setFramesWithRoi(status.frames_with_roi);
    } catch {
      // ignore
    }
  }, [totalFrames, setFramesWithRoi]);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Refresh after drawing (maskUrl changes)
  const maskUrl = useRoiStore((s) => s.maskUrl);
  useEffect(() => {
    if (maskUrl) {
      const t = setTimeout(refreshStatus, 300);
      return () => clearTimeout(t);
    }
  }, [maskUrl, refreshStatus]);

  // Auto-scroll to active row when editingFrameIdx changes
  useEffect(() => {
    if (!listRef.current) return;
    const row = listRef.current.querySelector(`[data-frame="${editingFrameIdx}"]`);
    if (row) {
      row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [editingFrameIdx]);

  if (totalFrames === 0) return null;

  // ── Compute reference frame set (0-indexed) ──
  const refFrameSet = new Set<number>();
  refFrameSet.add(0); // Frame 0 is always reference

  if (mode === "incremental") {
    if (keyFrameMode === "every_frame") {
      // Every frame except the last is a reference
      for (let i = 0; i < totalFrames - 1; i++) refFrameSet.add(i);
    } else if (keyFrameMode === "every_n") {
      for (let i = 0; i < totalFrames; i += keyFrameInterval) refFrameSet.add(i);
    } else {
      // "custom" — keyFrames are 1-indexed in appStore
      for (const kf of keyFrames) refFrameSet.add(kf - 1);
    }
  }

  const roiSet = new Set(framesWithRoi);

  // ── Summary counts ──
  const withRoi = framesWithRoi.length;
  const needRoi = [...refFrameSet].filter((k) => !roiSet.has(k)).length;

  return (
    <div className="border-b border-[var(--border)] bg-[var(--card)] flex flex-col">
      {/* Header row */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-[var(--border)]">
        <span className="text-[10px] font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
          Frames
        </span>
        <span className="ml-auto flex items-center gap-2 text-[10px] text-[var(--muted-foreground)]">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
            {withRoi}
          </span>
          {needRoi > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 inline-block" />
              {needRoi}
            </span>
          )}
          {withRoi > 0 && (
            <button
              className="p-0.5 rounded hover:bg-red-500/20 text-[var(--muted-foreground)] hover:text-red-500 transition-colors"
              title="Clear all ROIs"
              onClick={async () => {
                await clearAllFrameRois();
                setFramesWithRoi([]);
                setMaskUrl(null);
                setEditingFrameIdx(0);
                useAppStore.getState().bumpRoiVersion();
              }}
            >
              <Trash2 size={12} />
            </button>
          )}
        </span>
      </div>

      {/* Frame scrub slider — quick jump across all frames */}
      {totalFrames > 1 && (
        <div className="flex items-center gap-2 px-2 py-1 border-b border-[var(--border)]/70">
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            step={1}
            value={editingFrameIdx}
            onChange={(e) => setEditingFrameIdx(parseInt(e.target.value, 10))}
            className="flex-1 h-1 accent-[var(--primary)] cursor-pointer"
            aria-label="Frame scrub slider"
          />
          <span className="text-[10px] tabular-nums text-[var(--muted-foreground)] shrink-0 min-w-[52px] text-right">
            {editingFrameIdx + 1} / {totalFrames}
          </span>
        </div>
      )}

      {/* Column headers */}
      <div className="flex items-center px-2 py-0.5 text-[9px] text-[var(--muted-foreground)] uppercase tracking-wider border-b border-[var(--border)] bg-[var(--background)]">
        <span className="w-7 text-center shrink-0">#</span>
        <span className="flex-1 min-w-0 px-1">Filename</span>
        <span className="w-12 text-center shrink-0">ROI</span>
      </div>

      {/* Scrollable frame list */}
      <div
        ref={listRef}
        className="overflow-y-auto min-h-0"
        style={{ maxHeight: "200px" }}
      >
        {imageFiles.map((filename, i) => {
          const hasRoi = roiSet.has(i);
          const isRef = refFrameSet.has(i);
          const isActive = i === editingFrameIdx;

          // ROI button state
          let btnText: string;
          let btnClass: string;
          if (hasRoi) {
            btnText = "Edit";
            btnClass = "bg-emerald-600 text-white hover:bg-emerald-700";
          } else if (isRef) {
            btnText = "Need";
            btnClass = "bg-red-600 text-white hover:bg-red-700";
          } else {
            btnText = "Add";
            btnClass =
              "bg-[var(--secondary)] text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)]";
          }

          // Row styling
          const rowBg = isActive
            ? "bg-[var(--primary)]/10"
            : isRef
              ? "bg-[var(--accent)]/30"
              : "";

          return (
            <div
              key={i}
              data-frame={i}
              className={`flex items-center px-2 py-[3px] cursor-pointer transition-colors border-b border-[var(--border)]/50 ${rowBg} hover:bg-[var(--accent)]/50`}
              onClick={() => setEditingFrameIdx(i)}
            >
              {/* Frame number */}
              <span
                className={`w-7 text-center shrink-0 text-[10px] font-mono ${
                  isActive
                    ? "text-[var(--primary)] font-bold"
                    : isRef
                      ? "text-[var(--foreground)] font-semibold"
                      : "text-[var(--muted-foreground)]"
                }`}
              >
                {String(i).padStart(2, "0")}
              </span>

              {/* Filename */}
              <span
                className={`flex-1 min-w-0 px-1 text-[10px] truncate ${
                  isActive
                    ? "text-[var(--primary)] font-medium"
                    : isRef
                      ? "text-[var(--foreground)] font-medium"
                      : "text-[var(--muted-foreground)]"
                }`}
                title={filename}
              >
                {filename}
              </span>

              {/* ROI status button */}
              <button
                className={`w-12 shrink-0 text-center text-[9px] font-semibold py-[1px] rounded ${btnClass} transition-colors`}
                onClick={(e) => {
                  e.stopPropagation();
                  setEditingFrameIdx(i);
                }}
              >
                {btnText}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
