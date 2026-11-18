import { useState, useEffect, useCallback, useRef } from "react";
import { useRoiStore } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";
import { importMask } from "@/api/roi";
import { browseDirectory, type BrowseEntry } from "@/api/images";
import { useToast } from "@/components/shared/Toast";
import { Folder, Image, ArrowUp, X, Check } from "lucide-react";

export function RoiImportDialog() {
  const show = useRoiStore((s) => s.showImportDialog);
  const setShow = useRoiStore((s) => s.setShowImportDialog);
  const setMaskUrl = useRoiStore((s) => s.setMaskUrl);
  const editingFrameIdx = useRoiStore((s) => s.editingFrameIdx);
  const refreshMaskUrl = useRoiStore((s) => s.refreshMaskUrl);
  const imageWidth = useAppStore((s) => s.imageWidth);
  const imageHeight = useAppStore((s) => s.imageHeight);
  const { toast } = useToast();

  // Browse state
  const [currentPath, setCurrentPath] = useState("");
  const [parentPath, setParentPath] = useState("");
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [browseError, setBrowseError] = useState("");

  // Selection & adjust state
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [minAreaPct, setMinAreaPct] = useState(0); // percentage of total image area
  const [smoothRadius, setSmoothRadius] = useState(0);
  const [areaPx, setAreaPx] = useState(0);
  const [importing, setImporting] = useState(false);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const totalPixels = imageWidth * imageHeight;

  const pctToPixels = (pct: number) =>
    totalPixels > 0 ? Math.round((pct / 100) * totalPixels) : 0;

  const browse = useCallback(async (path: string) => {
    setBrowseLoading(true);
    setBrowseError("");
    try {
      const result = await browseDirectory(path);
      setCurrentPath(result.path);
      setParentPath(result.parent);
      setEntries(result.entries);
    } catch (e: any) {
      setBrowseError(e.response?.data?.error ?? "Failed to browse");
    } finally {
      setBrowseLoading(false);
    }
  }, []);

  useEffect(() => {
    if (show) {
      setSelectedFile(null);
      setMinAreaPct(0);
      setSmoothRadius(0);
      setAreaPx(0);
      browse("");
    }
  }, [show, browse]);

  const doImport = useCallback(
    async (filePath: string, minPx: number, sr: number) => {
      setImporting(true);
      try {
        const result = await importMask(filePath, minPx, sr, editingFrameIdx);
        setAreaPx(result.area_px);
        refreshMaskUrl();
      } catch (e: any) {
        toast("error", e?.response?.data?.error || "Import failed");
      } finally {
        setImporting(false);
      }
    },
    [editingFrameIdx, refreshMaskUrl, toast],
  );

  const handleFileClick = (fileName: string) => {
    const sep = currentPath.includes("\\") ? "\\" : "/";
    const fullPath = currentPath + sep + fileName;
    setSelectedFile(fullPath);
    setMinAreaPct(0);
    setSmoothRadius(0);
    doImport(fullPath, 0, 0);
  };

  const handleSliderChange = (newPct: number, newSmooth: number) => {
    if (!selectedFile) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      doImport(selectedFile, pctToPixels(newPct), newSmooth);
    }, 250);
  };

  const handleMinAreaChange = (val: number) => {
    setMinAreaPct(val);
    handleSliderChange(val, smoothRadius);
  };

  const handleSmoothChange = (val: number) => {
    setSmoothRadius(val);
    handleSliderChange(minAreaPct, val);
  };

  const handleDone = () => {
    if (areaPx > 0) {
      const frameLabel = editingFrameIdx === 0 ? "" : ` (Frame ${editingFrameIdx + 1})`;
      toast("success", `Mask applied${frameLabel}: ${areaPx.toLocaleString()} px`);
      // Only auto-confirm ROI for the reference frame (frame 0)
      if (editingFrameIdx === 0) {
        useAppStore.getState().setRoiConfirmed(true);
      }
      // Mark any dependent results stale regardless of target frame.
      useAppStore.getState().bumpRoiVersion();
    }
    setShow(false);
  };

  const handleClose = () => {
    setShow(false);
  };

  if (!show) return null;

  const dirs = entries.filter((e) => e.type === "dir");
  const images = entries.filter((e) => e.type === "image");

  // ── Phase 2: Floating adjust panel (no backdrop, doesn't block image) ──
  if (selectedFile) {
    return (
      <div className="fixed bottom-14 right-4 z-50 w-72 bg-[var(--card)] border border-[var(--border)] rounded-lg shadow-2xl">
        <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
          <h3 className="text-[11px] font-semibold text-[var(--foreground)]">
            Adjust Mask
          </h3>
          <button
            onClick={handleClose}
            className="p-0.5 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]"
          >
            <X size={12} />
          </button>
        </div>

        <div className="flex flex-col gap-2.5 px-3 py-2.5">
          {/* Selected file */}
          <div className="flex items-center gap-1.5">
            <Image size={12} className="text-[var(--primary)] shrink-0" />
            <span className="text-[10px] text-[var(--foreground)] truncate font-mono flex-1">
              {selectedFile.split(/[\\/]/).pop()}
            </span>
            <button
              onClick={() => setSelectedFile(null)}
              className="text-[9px] text-[var(--muted-foreground)] hover:text-[var(--foreground)] underline shrink-0"
            >
              Change
            </button>
          </div>

          {/* Area display */}
          <div className="text-[10px] text-[var(--muted-foreground)]">
            Mask area:{" "}
            <span className="text-[var(--foreground)] font-medium">
              {importing ? "..." : areaPx.toLocaleString()}
            </span>{" "}
            px
          </div>

          {/* Min area slider (percentage) */}
          <div>
            <div className="flex items-center justify-between mb-0.5">
              <label className="text-[10px] text-[var(--muted-foreground)]">
                Remove Small Blobs
              </label>
              <span className="text-[10px] text-[var(--foreground)] font-mono">
                {minAreaPct === 0
                  ? "off"
                  : `${minAreaPct.toFixed(2)}% (${pctToPixels(minAreaPct)} px)`}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={minAreaPct}
              onChange={(e) => handleMinAreaChange(Number(e.target.value))}
              className="w-full h-1 rounded-full appearance-none bg-[var(--secondary)] accent-[var(--primary)] cursor-pointer"
            />
          </div>

          {/* Smooth radius slider */}
          <div>
            <div className="flex items-center justify-between mb-0.5">
              <label className="text-[10px] text-[var(--muted-foreground)]">
                Edge Smoothing
              </label>
              <span className="text-[10px] text-[var(--foreground)] font-mono">
                {smoothRadius === 0 ? "off" : `${smoothRadius} px`}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={10}
              step={1}
              value={smoothRadius}
              onChange={(e) => handleSmoothChange(Number(e.target.value))}
              className="w-full h-1 rounded-full appearance-none bg-[var(--secondary)] accent-[var(--primary)] cursor-pointer"
            />
          </div>

          {/* Action buttons */}
          <div className="flex justify-end gap-1.5 pt-0.5">
            <button
              onClick={() => setSelectedFile(null)}
              className="h-6 px-2 text-[10px] rounded border border-[var(--border)] text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
            >
              Back
            </button>
            <button
              onClick={handleDone}
              disabled={importing || areaPx === 0}
              className="flex items-center gap-1 h-6 px-2 text-[10px] rounded bg-[var(--primary)] text-white hover:opacity-90 disabled:opacity-40"
            >
              <Check size={10} />
              Done
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Phase 1: Browse modal ──
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg shadow-xl w-[520px] max-h-[70vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <h3 className="text-[13px] font-semibold text-[var(--foreground)]">
            Select Mask Image
          </h3>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]"
          >
            <X size={14} />
          </button>
        </div>

        {/* Path bar — editable */}
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
            placeholder="Paste folder path and press Enter"
            className="flex-1 min-w-0 h-6 bg-transparent border-none text-[11px] text-[var(--foreground)] font-mono focus:outline-none placeholder:text-[var(--muted-foreground)]"
          />
        </div>

        {/* File list */}
        <div
          className="flex-1 overflow-y-auto min-h-0 px-2 py-1"
          style={{ maxHeight: "50vh" }}
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
                <Folder
                  size={14}
                  className="text-[var(--primary)] shrink-0"
                />
                <span className="text-[11px] text-[var(--foreground)] truncate">
                  {entry.name}
                </span>
              </button>
            ))}

          {/* Image files — clickable */}
          {!browseLoading && images.length > 0 && (
            <div className="mt-1 pt-1 border-t border-[var(--border)]">
              <span className="text-[9px] text-[var(--muted-foreground)] px-2 uppercase tracking-wider">
                Click an image to use as mask
              </span>
              {images.map((entry) => (
                <button
                  key={entry.name}
                  onClick={() => handleFileClick(entry.name)}
                  className="flex items-center gap-2 w-full px-2 py-1.5 rounded hover:bg-[var(--primary)]/20 text-left group"
                >
                  <Image
                    size={14}
                    className="text-[var(--muted-foreground)] group-hover:text-[var(--primary)] shrink-0"
                  />
                  <span className="text-[11px] text-[var(--foreground)] truncate">
                    {entry.name}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end px-4 py-3 border-t border-[var(--border)]">
          <button
            onClick={handleClose}
            className="px-3 py-1.5 rounded text-[11px] text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
