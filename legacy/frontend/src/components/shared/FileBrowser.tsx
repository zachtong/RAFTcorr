import { useState, useEffect, useCallback } from "react";
import { browseDirectory, type BrowseEntry } from "@/api/images";
import { Folder, Image, ArrowUp, X, Check, FileText } from "lucide-react";

interface FileBrowserProps {
  open: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  initialPath?: string;
  /** "directory" selects a folder (default), "file" selects a file */
  mode?: "directory" | "file";
  /** File extensions to show when mode="file" (e.g. [".raftproj"]) */
  fileFilter?: string[];
  /** Title override */
  title?: string;
  /** Pre-populated filename for save dialogs (mode="file") */
  defaultFileName?: string;
}

export function FileBrowser({
  open, onClose, onSelect, initialPath = "",
  mode = "directory", fileFilter, title, defaultFileName = "",
}: FileBrowserProps) {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [parentPath, setParentPath] = useState("");
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [imageCount, setImageCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  // For save mode: user can type a new filename
  const [newFileName, setNewFileName] = useState("");
  // Editable path bar — user can type/paste a path and press Enter
  const [pathInput, setPathInput] = useState(initialPath);
  const [editingPath, setEditingPath] = useState(false);

  const browse = useCallback(async (path: string) => {
    setLoading(true);
    setError("");
    setSelectedFile(null);
    try {
      const result = await browseDirectory(path, fileFilter);
      setCurrentPath(result.path);
      setParentPath(result.parent);
      setEntries(result.entries);
      setImageCount(result.image_count);
      setPathInput(result.path);
    } catch (e: any) {
      setError(e.response?.data?.error ?? "Failed to browse directory");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      browse(initialPath);
      setNewFileName(defaultFileName);
      setPathInput(initialPath);
      setEditingPath(false);
    }
  }, [open, initialPath, browse, defaultFileName]);

  if (!open) return null;

  const dirs = entries.filter((e) => e.type === "dir");
  const images = entries.filter((e) => e.type === "image");

  // In file mode, show entries of type "file" (returned by backend with file_extensions filter)
  const allFiles = mode === "file"
    ? entries.filter((e) => e.type === "file")
    : [];

  const isDirectoryMode = mode === "directory";
  const displayTitle = title ?? (isDirectoryMode ? "Select Image Directory" : "Select File");

  const handleSelect = () => {
    if (isDirectoryMode) {
      onSelect(currentPath);
    } else if (selectedFile) {
      onSelect(currentPath + "/" + selectedFile);
    } else if (newFileName.trim()) {
      const name = newFileName.trim();
      // Ensure extension if fileFilter provided
      const hasExt = fileFilter?.some((ext) => name.toLowerCase().endsWith(ext.toLowerCase()));
      const finalName = hasExt || !fileFilter?.length ? name : name + fileFilter[0];
      onSelect(currentPath + "/" + finalName);
    }
    onClose();
  };

  const canSelect = isDirectoryMode
    ? imageCount > 0
    : !!(selectedFile || newFileName.trim());

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg shadow-xl w-[500px] max-h-[70vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <h3 className="text-[13px] font-semibold text-[var(--foreground)]">{displayTitle}</h3>
          <button onClick={onClose} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
            <X size={14} />
          </button>
        </div>

        {/* Path bar — click to edit, Enter to navigate */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--border)] bg-[var(--background)]">
          <button
            onClick={() => browse(parentPath)}
            disabled={currentPath === parentPath}
            className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)] disabled:opacity-30"
          >
            <ArrowUp size={14} />
          </button>
          {editingPath ? (
            <input
              autoFocus
              type="text"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  setEditingPath(false);
                  browse(pathInput.trim());
                } else if (e.key === "Escape") {
                  setEditingPath(false);
                  setPathInput(currentPath);
                }
              }}
              onBlur={() => {
                setEditingPath(false);
                if (pathInput.trim() && pathInput.trim() !== currentPath) {
                  browse(pathInput.trim());
                } else {
                  setPathInput(currentPath);
                }
              }}
              className="flex-1 text-[11px] text-[var(--foreground)] font-mono bg-[var(--input)] border border-[var(--primary)] rounded px-1.5 py-0.5 outline-none"
            />
          ) : (
            <button
              onClick={() => setEditingPath(true)}
              className="flex-1 text-left text-[11px] text-[var(--foreground)] truncate font-mono hover:bg-[var(--input)] rounded px-1.5 py-0.5 transition-colors"
              title="Click to edit path"
            >
              {currentPath}
            </button>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto min-h-0 px-2 py-1">
          {loading && (
            <div className="flex items-center justify-center py-8 text-[11px] text-[var(--muted-foreground)]">
              Loading...
            </div>
          )}
          {error && (
            <div className="text-[11px] text-red-400 py-4 text-center">{error}</div>
          )}
          {!loading && !error && entries.length === 0 && (
            <div className="text-[11px] text-[var(--muted-foreground)] py-4 text-center">
              Empty directory
            </div>
          )}
          {!loading && dirs.map((entry) => (
            <button
              key={entry.name}
              onClick={() => browse(currentPath + "/" + entry.name)}
              className="flex items-center gap-2 w-full px-2 py-1.5 rounded hover:bg-[var(--secondary)] text-left"
            >
              <Folder size={14} className="text-[var(--primary)] shrink-0" />
              <span className="text-[11px] text-[var(--foreground)] truncate">{entry.name}</span>
            </button>
          ))}

          {/* File mode: show matching files */}
          {!loading && mode === "file" && allFiles.length > 0 && (
            <div className="mt-1 pt-1 border-t border-[var(--border)]">
              <span className="text-[9px] text-[var(--muted-foreground)] px-2 uppercase tracking-wider">
                {allFiles.length} file{allFiles.length !== 1 ? "s" : ""}
              </span>
              {allFiles.map((entry) => (
                <button
                  key={entry.name}
                  onClick={() => { setSelectedFile(entry.name); setNewFileName(""); }}
                  className={`flex items-center gap-2 w-full px-2 py-1.5 rounded text-left ${
                    selectedFile === entry.name
                      ? "bg-[var(--primary)]/20 ring-1 ring-[var(--primary)]"
                      : "hover:bg-[var(--secondary)]"
                  }`}
                >
                  <FileText size={14} className="text-[var(--primary)] shrink-0" />
                  <span className="text-[11px] text-[var(--foreground)] truncate">{entry.name}</span>
                </button>
              ))}
            </div>
          )}

          {/* Directory mode: show image preview */}
          {!loading && isDirectoryMode && images.length > 0 && (
            <div className="mt-1 pt-1 border-t border-[var(--border)]">
              <span className="text-[9px] text-[var(--muted-foreground)] px-2 uppercase tracking-wider">
                {images.length} image{images.length !== 1 ? "s" : ""} in this directory
              </span>
              {images.slice(0, 5).map((entry) => (
                <div key={entry.name} className="flex items-center gap-2 px-2 py-1">
                  <Image size={12} className="text-[var(--muted-foreground)] shrink-0" />
                  <span className="text-[10px] text-[var(--muted-foreground)] truncate">{entry.name}</span>
                </div>
              ))}
              {images.length > 5 && (
                <span className="text-[9px] text-[var(--muted-foreground)] px-2">
                  ... and {images.length - 5} more
                </span>
              )}
            </div>
          )}
        </div>

        {/* File mode: new filename input */}
        {mode === "file" && (
          <div className="px-4 py-2 border-t border-[var(--border)] bg-[var(--background)]">
            <input
              type="text"
              value={newFileName}
              onChange={(e) => { setNewFileName(e.target.value); setSelectedFile(null); }}
              placeholder={`New filename${fileFilter ? ` (${fileFilter.join(", ")})` : ""}`}
              className="w-full px-2 py-1.5 rounded text-[11px] bg-[var(--card)] border border-[var(--border)] text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:border-[var(--primary)] outline-none"
            />
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-[var(--border)]">
          <span className="text-[10px] text-[var(--muted-foreground)]">
            {isDirectoryMode
              ? imageCount > 0
                ? `${imageCount} image${imageCount !== 1 ? "s" : ""} found`
                : "No images in this directory"
              : selectedFile
                ? selectedFile
                : newFileName.trim()
                  ? `New: ${newFileName.trim()}`
                  : "Select or type a filename"
            }
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded text-[11px] text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
            >
              Cancel
            </button>
            <button
              onClick={handleSelect}
              disabled={!canSelect}
              className="flex items-center gap-1 px-3 py-1.5 bg-[var(--primary)] hover:bg-[var(--primary)]/90 text-white rounded text-[11px] disabled:opacity-40"
            >
              <Check size={12} />
              {isDirectoryMode ? `Select (${imageCount})` : "Select"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
