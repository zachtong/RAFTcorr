import { useState } from "react";
import { Save, FolderOpen, Loader2 } from "lucide-react";
import { useAppStore } from "@/stores/appStore";
import { useToast } from "@/components/shared/Toast";
import { saveSession, loadSession } from "@/api/session";
import { FileBrowser } from "@/components/shared/FileBrowser";

type BrowseTarget = "save" | "load" | null;

export function SessionControls() {
  const hasResults = useAppStore((s) => s.hasResults);
  const restoreFromSession = useAppStore((s) => s.restoreFromSession);
  const resetSession = useAppStore((s) => s.resetSession);
  const { toast } = useToast();

  const [browseTarget, setBrowseTarget] = useState<BrowseTarget>(null);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSave = async (filePath: string) => {
    setSaving(true);
    try {
      const result = await saveSession({ file_path: filePath });
      toast("success", `Session saved (${result.size_mb} MB)`);
    } catch (e: any) {
      const data = e.response?.data;
      if (data?.exists) {
        // File exists — ask to overwrite
        if (confirm(`File already exists:\n${data.path}\n\nOverwrite?`)) {
          try {
            const result = await saveSession({ file_path: filePath, overwrite: true });
            toast("success", `Session saved (${result.size_mb} MB)`);
          } catch (e2: any) {
            toast("error", e2.response?.data?.error ?? "Save failed");
          }
        }
      } else {
        toast("error", data?.error ?? "Save failed");
      }
    } finally {
      setSaving(false);
    }
  };

  const handleLoad = async (filePath: string) => {
    setLoading(true);
    try {
      resetSession();
      const result = await loadSession({ file_path: filePath });

      // Restore frontend state from summary
      restoreFromSession(result.summary);

      // Show warnings if any
      if (result.warnings.length > 0) {
        toast("warning", result.warnings.join("\n"));
      }

      const s = result.summary;
      toast(
        "success",
        `Loaded: ${s.num_displacement_frames} frames` +
        (s.num_strain_frames > 0 ? `, ${s.num_strain_frames} strain frames` : "") +
        (s.num_probes > 0 ? `, ${s.num_probes} probes` : ""),
      );
    } catch (e: any) {
      toast("error", e.response?.data?.error ?? "Load failed");
    } finally {
      setLoading(false);
    }
  };

  const busy = saving || loading;

  // Default filename: RAFTcorr_YYYY-MM-DD
  const today = new Date().toISOString().slice(0, 10);
  const defaultSaveFileName = `RAFTcorr_${today}.raftproj`;

  return (
    <>
      <div className="flex items-center gap-1 px-3 py-2 border-b border-[var(--border)]">
        <button
          onClick={() => setBrowseTarget("save")}
          disabled={!hasResults || busy}
          title="Save session"
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium
            bg-[var(--secondary)] hover:bg-[var(--secondary)]/80
            text-[var(--foreground)] disabled:opacity-30 transition-colors"
        >
          {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
          Save
        </button>
        <button
          onClick={() => setBrowseTarget("load")}
          disabled={busy}
          title="Load session"
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium
            bg-[var(--secondary)] hover:bg-[var(--secondary)]/80
            text-[var(--foreground)] disabled:opacity-30 transition-colors"
        >
          {loading ? <Loader2 size={12} className="animate-spin" /> : <FolderOpen size={12} />}
          Load
        </button>
      </div>

      {browseTarget && (
        <FileBrowser
          open
          onClose={() => setBrowseTarget(null)}
          onSelect={(path) => {
            setBrowseTarget(null);
            if (browseTarget === "save") handleSave(path);
            else handleLoad(path);
          }}
          initialPath=""
          mode="file"
          fileFilter={[".raftproj"]}
          title={browseTarget === "save" ? "Save Session" : "Load Session"}
          defaultFileName={browseTarget === "save" ? defaultSaveFileName : ""}
        />
      )}
    </>
  );
}
