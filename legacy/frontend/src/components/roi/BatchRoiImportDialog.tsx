import { useState, useEffect, useCallback } from "react";
import { useRoiStore } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";
import { batchImportRois, getFramesRoiStatus } from "@/api/roi";
import { browseDirectory, type BrowseEntry } from "@/api/images";
import { confirmRoi } from "@/api/roi";
import { useToast } from "@/components/shared/Toast";
import { Folder, Image, ArrowUp, X, Check, Layers } from "lucide-react";

/**
 * Batch ROI import dialog — lets users select a folder of mask images
 * and assign them to frames using sequential or auto-match strategy.
 *
 * Follows pyALDIC's batch import pattern:
 *   1. Browse to mask folder
 *   2. Choose strategy (sequential / auto-match by name)
 *   3. Preview assignments
 *   4. Confirm import
 */
export function BatchRoiImportDialog() {
  const show = useRoiStore((s) => s.showBatchImportDialog);
  const setShow = useRoiStore((s) => s.setShowBatchImportDialog);
  const setFramesWithRoi = useRoiStore((s) => s.setFramesWithRoi);
  const refreshMaskUrl = useRoiStore((s) => s.refreshMaskUrl);
  const imageFiles = useAppStore((s) => s.imageFiles);
  const { toast } = useToast();

  // Browse state
  const [currentPath, setCurrentPath] = useState("");
  const [parentPath, setParentPath] = useState("");
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [browseError, setBrowseError] = useState("");

  // Import state
  const [strategy, setStrategy] = useState<"sequential" | "auto_match">("sequential");
  const [importing, setImporting] = useState(false);
  const [maskCount, setMaskCount] = useState(0);

  const browse = useCallback(async (path: string) => {
    setBrowseLoading(true);
    setBrowseError("");
    try {
      const result = await browseDirectory(path);
      setCurrentPath(result.path);
      setParentPath(result.parent);
      setEntries(result.entries);
      // Count image files (potential masks)
      setMaskCount(result.entries.filter((e) => e.type === "image").length);
    } catch (e: any) {
      setBrowseError(e.response?.data?.error ?? "Failed to browse");
    } finally {
      setBrowseLoading(false);
    }
  }, []);

  useEffect(() => {
    if (show) {
      setStrategy("sequential");
      setImporting(false);
      browse("");
    }
  }, [show, browse]);

  const handleImport = async () => {
    if (!currentPath || maskCount === 0) return;
    setImporting(true);
    try {
      const result = await batchImportRois(currentPath, strategy);

      // Refresh ROI status
      const status = await getFramesRoiStatus();
      setFramesWithRoi(status.frames_with_roi);

      // If frame 0 got a mask, confirm it
      if (status.frame_0_confirmed || status.frames_with_roi.includes(0)) {
        await confirmRoi();
        useAppStore.getState().setRoiConfirmed(true);
      }
      useAppStore.getState().bumpRoiVersion();

      refreshMaskUrl();

      toast(
        "success",
        `Batch imported ${result.imported} masks for ${Object.keys(result.assignments).length} frames`
      );
      setShow(false);
    } catch (e: any) {
      toast("error", e?.response?.data?.error || "Batch import failed");
    } finally {
      setImporting(false);
    }
  };

  const handleClose = () => setShow(false);

  if (!show) return null;

  const dirs = entries.filter((e) => e.type === "dir");
  const images = entries.filter((e) => e.type === "image");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg shadow-xl w-[560px] max-h-[75vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-[var(--primary)]" />
            <h3 className="text-[13px] font-semibold text-[var(--foreground)]">
              Batch Import ROI Masks
            </h3>
          </div>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]"
          >
            <X size={14} />
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--border)] bg-[var(--background)]">
          <button
            onClick={() => browse(parentPath)}
            disabled={currentPath === parentPath}
            className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)] disabled:opacity-30"
          >
            <ArrowUp size={14} />
          </button>
          <input
            type="text"
            value={currentPath}
            onChange={(e) => setCurrentPath(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") browse(currentPath);
            }}
            placeholder="Paste mask folder path and press Enter"
            className="flex-1 min-w-0 h-6 bg-transparent border-none text-[11px] text-[var(--foreground)] font-mono focus:outline-none placeholder:text-[var(--muted-foreground)]"
          />
        </div>

        {/* Strategy selector */}
        <div className="flex items-center gap-3 px-4 py-2 border-b border-[var(--border)]">
          <span className="text-[10px] text-[var(--muted-foreground)] font-semibold uppercase tracking-wider">
            Strategy
          </span>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="radio"
              name="strategy"
              checked={strategy === "sequential"}
              onChange={() => setStrategy("sequential")}
              className="w-3 h-3 accent-[var(--primary)]"
            />
            <span className="text-[11px] text-[var(--foreground)]">
              Sequential
            </span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="radio"
              name="strategy"
              checked={strategy === "auto_match"}
              onChange={() => setStrategy("auto_match")}
              className="w-3 h-3 accent-[var(--primary)]"
            />
            <span className="text-[11px] text-[var(--foreground)]">
              Auto-match by name
            </span>
          </label>
        </div>

        {/* File list */}
        <div
          className="flex-1 overflow-y-auto min-h-0 px-2 py-1"
          style={{ maxHeight: "40vh" }}
        >
          {browseLoading && (
            <div className="flex items-center justify-center py-8 text-[11px] text-[var(--muted-foreground)]">
              Loading...
            </div>
          )}
          {browseError && (
            <div className="text-[11px] text-red-400 py-4 text-center">
              {browseError}
            </div>
          )}
          {!browseLoading && !browseError && entries.length === 0 && (
            <div className="text-[11px] text-[var(--muted-foreground)] py-4 text-center">
              Empty directory
            </div>
          )}

          {/* Folders */}
          {!browseLoading &&
            dirs.map((entry) => (
              <button
                key={entry.name}
                onClick={() => browse(currentPath + "/" + entry.name)}
                className="flex items-center gap-2 w-full px-2 py-1.5 rounded hover:bg-[var(--secondary)] text-left"
              >
                <Folder size={14} className="text-[var(--primary)] shrink-0" />
                <span className="text-[11px] text-[var(--foreground)] truncate">
                  {entry.name}
                </span>
              </button>
            ))}

          {/* Mask image files */}
          {!browseLoading && images.length > 0 && (
            <div className="mt-1 pt-1 border-t border-[var(--border)]">
              <span className="text-[9px] text-[var(--muted-foreground)] px-2 uppercase tracking-wider">
                {images.length} mask files found
              </span>
              {images.map((entry) => (
                <div
                  key={entry.name}
                  className="flex items-center gap-2 w-full px-2 py-1 text-left"
                >
                  <Image
                    size={12}
                    className="text-[var(--muted-foreground)] shrink-0"
                  />
                  <span className="text-[10px] text-[var(--foreground)] truncate font-mono">
                    {entry.name}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer with summary + import button */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-[var(--border)]">
          <div className="text-[10px] text-[var(--muted-foreground)]">
            {maskCount > 0 ? (
              <>
                <span className="font-medium text-[var(--foreground)]">
                  {maskCount}
                </span>{" "}
                masks →{" "}
                <span className="font-medium text-[var(--foreground)]">
                  {imageFiles.length}
                </span>{" "}
                frames (
                {strategy === "sequential"
                  ? "assigned in order"
                  : "matched by filename numbers"}
                )
              </>
            ) : (
              "Navigate to a folder containing mask images"
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleClose}
              className="px-3 py-1.5 rounded text-[11px] text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
            >
              Cancel
            </button>
            <button
              onClick={handleImport}
              disabled={importing || maskCount === 0}
              className="flex items-center gap-1 px-3 py-1.5 rounded text-[11px] bg-[var(--primary)] text-white hover:opacity-90 disabled:opacity-40"
            >
              <Check size={12} />
              {importing ? "Importing..." : "Import All"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
