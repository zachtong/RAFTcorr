import { useRoiStore, type ShapeMode } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";
import { clearRoi, clearFrameRoi, invertMask, exportMask } from "@/api/roi";
import {
  Pentagon,
  RectangleHorizontal,
  Circle,
  Plus,
  Scissors,
  Upload,
  FolderUp,
  Download,
  RotateCcw,
  Trash2,
  ChevronUp,
  Undo2,
  Redo2,
} from "lucide-react";

const SHAPES: {
  id: ShapeMode;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { id: "polygon", label: "Polygon", icon: Pentagon },
  { id: "rectangle", label: "Rectangle", icon: RectangleHorizontal },
  { id: "circle", label: "Circle", icon: Circle },
];

/** Flyout tool group — hover to reveal shape sub-tools (Photoshop-style). */
function ToolGroup({
  label,
  icon: GroupIcon,
  isCut,
  disabled,
}: {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  isCut: boolean;
  disabled: boolean;
}) {
  const drawingMode = useRoiStore((s) => s.drawingMode);
  const cutMode = useRoiStore((s) => s.cutMode);
  const activateTool = useRoiStore((s) => s.activateTool);

  const isGroupActive = drawingMode !== null && cutMode === isCut;
  const activeShape = isGroupActive
    ? SHAPES.find((s) => s.id === drawingMode)
    : null;
  const DisplayIcon = activeShape?.icon ?? GroupIcon;

  const activeColor = isCut
    ? "bg-red-600 text-white"
    : "bg-blue-600 text-white";

  return (
    <div className="relative group/toolgroup">
      {/* Main button */}
      <button
        disabled={disabled}
        className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-colors disabled:opacity-30 ${
          isGroupActive
            ? activeColor
            : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)]"
        }`}
      >
        <DisplayIcon className="w-3 h-3" />
        {label}
        <ChevronUp className="w-2.5 h-2.5 opacity-50" />
      </button>

      {/* Flyout — appears above on hover */}
      <div className="absolute bottom-full left-0 pb-1 hidden group-hover/toolgroup:block z-50">
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-md shadow-lg py-1 min-w-[110px]">
          {SHAPES.map((shape) => {
            const isActive = drawingMode === shape.id && cutMode === isCut;
            return (
              <button
                key={shape.id}
                disabled={disabled}
                onClick={() => activateTool(shape.id, isCut)}
                className={`w-full flex items-center gap-2 px-3 py-1.5 text-[11px] transition-colors disabled:opacity-30 ${
                  isActive
                    ? isCut
                      ? "bg-red-600/20 text-red-400"
                      : "bg-blue-600/20 text-blue-400"
                    : "text-[var(--foreground)] hover:bg-[var(--secondary)]"
                }`}
              >
                <shape.icon className="w-3.5 h-3.5" />
                {shape.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function RoiToolbar() {
  const clearPoints = useRoiStore((s) => s.clearPoints);
  const setMaskUrl = useRoiStore((s) => s.setMaskUrl);
  const editingFrameIdx = useRoiStore((s) => s.editingFrameIdx);
  const refreshMaskUrl = useRoiStore((s) => s.refreshMaskUrl);
  const snapshotBeforeEdit = useRoiStore((s) => s.snapshotBeforeEdit);
  const undo = useRoiStore((s) => s.undo);
  const redo = useRoiStore((s) => s.redo);
  const undoLen = useRoiStore((s) => s.undoStack.length);
  const redoLen = useRoiStore((s) => s.redoStack.length);
  const setRoiConfirmed = useAppStore((s) => s.setRoiConfirmed);
  const imageFiles = useAppStore((s) => s.imageFiles);
  const maskUrl = useRoiStore((s) => s.maskUrl);

  const disabled = imageFiles.length === 0;

  const handleClear = async () => {
    await snapshotBeforeEdit(editingFrameIdx);
    if (editingFrameIdx === 0) {
      await clearRoi();
      setMaskUrl(null);
      setRoiConfirmed(false);
    } else {
      await clearFrameRoi(editingFrameIdx);
      refreshMaskUrl();
    }
    clearPoints();
  };

  const handleInvert = async () => {
    await snapshotBeforeEdit(editingFrameIdx);
    const result = await invertMask(editingFrameIdx);
    if (result) refreshMaskUrl();
  };

  const handleImport = () => {
    useRoiStore.getState().setShowImportDialog(true);
  };

  const handleBatchImport = () => {
    useRoiStore.getState().setShowBatchImportDialog(true);
  };

  const handleSave = async () => {
    try {
      await exportMask();
    } catch (err) {
      console.error("Failed to export mask:", err);
    }
  };

  const actionBtnClass =
    "flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-colors disabled:opacity-30 text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)]";

  return (
    <div className="flex items-center justify-between px-3 py-1.5 bg-[var(--card)] border-t border-[var(--border)]">
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)] mr-2 uppercase tracking-wider font-semibold">
          ROI Tools
        </span>

        {/* Shape tool groups */}
        <ToolGroup
          label="Add"
          icon={Plus}
          isCut={false}
          disabled={disabled}
        />
        <ToolGroup
          label="Cut"
          icon={Scissors}
          isCut={true}
          disabled={disabled}
        />

        {/* Separator */}
        <div className="w-px h-4 bg-[var(--border)] mx-1" />

        {/* Undo / Redo */}
        <button
          onClick={() => undo()}
          disabled={disabled || undoLen === 0}
          className={actionBtnClass}
          title="Undo last ROI change (Ctrl+Z)"
        >
          <Undo2 className="w-3 h-3" />
          Undo
          {undoLen > 0 && (
            <span className="text-[8px] opacity-60 ml-0.5">{undoLen}</span>
          )}
        </button>
        <button
          onClick={() => redo()}
          disabled={disabled || redoLen === 0}
          className={actionBtnClass}
          title="Redo (Ctrl+Shift+Z)"
        >
          <Redo2 className="w-3 h-3" />
          Redo
          {redoLen > 0 && (
            <span className="text-[8px] opacity-60 ml-0.5">{redoLen}</span>
          )}
        </button>

        <div className="w-px h-4 bg-[var(--border)] mx-1" />

        {/* Action buttons */}
        <button
          onClick={handleImport}
          disabled={disabled}
          className={actionBtnClass}
          title="Import mask for current frame"
        >
          <Upload className="w-3 h-3" />
          Import
        </button>
        <button
          onClick={handleBatchImport}
          disabled={disabled}
          className={actionBtnClass}
          title="Batch import masks for multiple frames"
        >
          <FolderUp className="w-3 h-3" />
          Batch
        </button>
        <button
          onClick={handleSave}
          disabled={disabled || !maskUrl}
          className={actionBtnClass}
        >
          <Download className="w-3 h-3" />
          Save
        </button>
        <button
          onClick={handleInvert}
          disabled={disabled || !maskUrl}
          className={actionBtnClass}
        >
          <RotateCcw className="w-3 h-3" />
          Invert
        </button>
        <button
          onClick={handleClear}
          disabled={disabled}
          className={actionBtnClass}
        >
          <Trash2 className="w-3 h-3" />
          Clear
        </button>
      </div>
    </div>
  );
}
