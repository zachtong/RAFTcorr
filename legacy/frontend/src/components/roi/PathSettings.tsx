import { useState } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { SelectField } from "@/components/shared/SelectField";
import { useAppStore } from "@/stores/appStore";
import { useRoiStore } from "@/stores/roiStore";
import { useToast } from "@/components/shared/Toast";
import { loadImages } from "@/api/images";
import { clearRoi } from "@/api/roi";
import { clearProbes } from "@/api/probes";
import { listModels, describeModel, selectModel } from "@/api/models";
import { FolderOpen, FolderSearch } from "lucide-react";
import { FileBrowser } from "@/components/shared/FileBrowser";

export function PathSettings() {
  const imageDir = useAppStore((s) => s.imageDir);
  const setImageDir = useAppStore((s) => s.setImageDir);
  const setImageFiles = useAppStore((s) => s.setImageFiles);
  const resetSession = useAppStore((s) => s.resetSession);
  const imageFiles = useAppStore((s) => s.imageFiles);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const setModel = useAppStore((s) => s.setModel);
  const modelMetadata = useAppStore((s) => s.modelMetadata);
  const processingActive = useAppStore((s) => s.processingActive);
  const resetRoi = useRoiStore((s) => s.setMaskUrl);
  const clearDrawing = useRoiStore((s) => s.setDrawingMode);

  const { toast } = useToast();
  const [models, setModels] = useState<{ value: string; label: string }[]>([]);
  const [dirInput, setDirInput] = useState(imageDir);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [naturalSort, setNaturalSort] = useState(() => {
    const saved = localStorage.getItem("dic_natural_sort");
    return saved !== null ? saved === "true" : true; // default ON
  });
  const [showFileList, setShowFileList] = useState(false);

  const handleLoad = async (overridePath?: string, sortOverride?: boolean) => {
    const dir = (overridePath ?? dirInput).trim();
    const useNaturalSort = sortOverride ?? naturalSort;
    if (!dir) {
      setError("Please enter a directory path");
      toast("error", "Please enter a directory path");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const result = await loadImages(dir, useNaturalSort);
      // Explicitly clear backend ROI + probes + frontend state
      await Promise.all([
        clearRoi().catch(() => {}),
        clearProbes().catch(() => {}),
      ]);
      resetSession();
      resetRoi(null);
      clearDrawing(null);
      setDirInput(dir);
      setImageDir(dir);
      setImageFiles(result.files, result.width, result.height);
      toast("success", `Loaded ${result.count} images (${result.width}x${result.height})`);
      if (result.sort_suggestion === "natural") {
        toast("info", "Filenames contain numbers — consider enabling Natural Sort for correct ordering");
      }
      // Also fetch models
      const modelList = await listModels();
      setModels(modelList.map((m) => ({ value: m.path, label: m.label })));
      if (modelList.length > 0 && !selectedModel) {
        const first = modelList[0];
        await handleModelSelect(first.path);
        // Make the silent auto-selection visible to the user.
        if (modelList.length > 1) {
          toast(
            "info",
            `Using model: ${first.label} (${modelList.length - 1} other${modelList.length > 2 ? "s" : ""} available — change in the Model dropdown)`,
          );
        } else {
          toast("info", `Using model: ${first.label}`);
        }
      }
    } catch (e: any) {
      const msg = e.response?.data?.error ?? "Failed to load images — is the Flask server running?";
      setError(msg);
      toast("error", msg);
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelect = async (path: string) => {
    try {
      const meta = await describeModel(path);
      await selectModel(path);
      setModel(path, meta);
    } catch {
      setModel(path, null);
    }
  };

  return (
    <CollapsibleSection title="Path Settings">
      {/* Directory — full-width stacked layout */}
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-[var(--muted-foreground)]">Image Directory</span>
        <div className="flex items-center gap-1">
          <input
            type="text"
            value={dirInput}
            onChange={(e) => setDirInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            placeholder="/path/to/images"
            className="flex-1 min-w-0 h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[11px] text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none"
          />
          <button
            onClick={() => setBrowserOpen(true)}
            className="h-6 px-1.5 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 text-[var(--foreground)] rounded text-[10px] shrink-0"
            title="Browse"
          >
            <FolderSearch size={12} />
          </button>
          <button
            onClick={() => handleLoad()}
            disabled={loading}
            className="h-6 px-2 bg-[var(--primary)] text-white rounded text-[10px] hover:opacity-90 disabled:opacity-50 flex items-center gap-1 shrink-0"
          >
            <FolderOpen size={10} />
            {loading ? "..." : "Confirm"}
          </button>
        </div>
      </div>

      {/* Natural sort toggle */}
      <label className="flex items-center gap-1.5 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={naturalSort}
          onChange={(e) => {
            const checked = e.target.checked;
            setNaturalSort(checked);
            localStorage.setItem("dic_natural_sort", String(checked));
            if (imageFiles.length > 0) {
              handleLoad(undefined, checked);
            }
          }}
          className="accent-[var(--primary)] w-3 h-3"
        />
        <span className="text-[10px] text-[var(--muted-foreground)]">
          Natural sort (image1, image2, ... image11)
        </span>
      </label>

      <FileBrowser
        open={browserOpen}
        onClose={() => setBrowserOpen(false)}
        onSelect={(path) => {
          setBrowserOpen(false);
          handleLoad(path);
        }}
        initialPath={dirInput}
      />

      {error && (
        <p className="text-[10px] text-red-400">{error}</p>
      )}

      {imageFiles.length > 0 && (
        <div className="flex flex-col gap-1">
          <button
            onClick={() => setShowFileList(!showFileList)}
            className="text-[10px] text-[var(--muted-foreground)] hover:text-[var(--foreground)] text-left flex items-center gap-1"
          >
            <span className="text-[8px]">{showFileList ? "\u25BC" : "\u25B6"}</span>
            {imageFiles.length} images loaded
            {naturalSort ? " (natural sort)" : " (lexicographic)"}
          </button>
          {showFileList && (
            <div className="max-h-32 overflow-y-auto border border-[#3a3d45] rounded bg-[var(--input)] p-1">
              {imageFiles.map((f, i) => (
                <div key={i} className="text-[9px] text-[var(--muted-foreground)] font-mono leading-tight px-1">
                  <span className="text-[var(--primary)] mr-1">{String(i).padStart(3, "\u00A0")}</span>
                  {f}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {models.length > 0 && (
        <FieldRow label={`Model${models.length > 1 ? ` (${models.length})` : ""}`}>
          <SelectField
            value={selectedModel}
            options={models}
            onChange={handleModelSelect}
            className="flex-1 min-w-0"
            disabled={processingActive}
            title={
              processingActive
                ? "Model locked while processing is running — cancel or finish the run first"
                : undefined
            }
          />
        </FieldRow>
      )}

      {selectedModel && (
        <div className="text-[10px] text-[var(--muted-foreground)] -mt-1">
          <span className="opacity-70">In use: </span>
          <span className="font-mono text-[var(--foreground)]">
            {models.find((m) => m.value === selectedModel)?.label ?? selectedModel}
          </span>
        </div>
      )}

      {modelMetadata && (
        <div className="flex gap-1 flex-wrap">
          <span className="px-1.5 py-0.5 bg-[var(--secondary)] rounded text-[9px] text-[var(--muted-foreground)]">
            {modelMetadata.small ? "RAFT-Small" : "RAFT-Large"}
          </span>
          <span className="px-1.5 py-0.5 bg-[var(--secondary)] rounded text-[9px] text-[var(--muted-foreground)]">
            {modelMetadata.full_resolution ? "Full res" : "1/8 res"}
          </span>
          {modelMetadata.corr_levels && (
            <span className="px-1.5 py-0.5 bg-[var(--secondary)] rounded text-[9px] text-[var(--muted-foreground)]">
              {modelMetadata.corr_levels}-level pyramid
            </span>
          )}
        </div>
      )}
    </CollapsibleSection>
  );
}
