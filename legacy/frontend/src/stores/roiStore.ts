import { create } from "zustand";
import { fetchMaskBlob, uploadMask } from "@/api/roi";
import { useAppStore } from "@/stores/appStore";

/** Notify the app store that the ROI data has changed so the stale-result
 *  banner on Displacement / Post-Processing pages stays in sync.  Works for
 *  every frame, not just frame 0 — the earlier implementation only bumped
 *  on frame-0 confirmation and let per-frame edits slip through silently. */
function markRoiChanged() {
  useAppStore.getState().bumpRoiVersion();
}

export type ShapeMode = "polygon" | "rectangle" | "circle";
export type DrawingMode = ShapeMode | null;

/** One saved mask snapshot. `null` blob means "empty mask". */
interface HistoryEntry {
  frameIdx: number;
  blob: Blob | null;
}

const HISTORY_LIMIT = 20;

interface RoiState {
  drawingMode: DrawingMode;
  cutMode: boolean;
  currentPoints: [number, number][];
  maskUrl: string | null;
  showImportDialog: boolean;
  showBatchImportDialog: boolean;

  // Per-frame ROI editing
  editingFrameIdx: number; // 0-based frame index being edited
  framesWithRoi: number[]; // which frames have ROI

  // Undo/redo stacks — each entry records which frame it belonged to so that
  // travelling back through history re-selects the edited frame.
  undoStack: HistoryEntry[];
  redoStack: HistoryEntry[];

  setDrawingMode: (mode: DrawingMode) => void;
  setCutMode: (cut: boolean) => void;
  activateTool: (shape: ShapeMode, cut: boolean) => void;
  addPoint: (x: number, y: number) => void;
  clearPoints: () => void;
  setMaskUrl: (url: string | null) => void;
  setShowImportDialog: (v: boolean) => void;
  setShowBatchImportDialog: (v: boolean) => void;

  // Per-frame actions
  setEditingFrameIdx: (idx: number) => void;
  setFramesWithRoi: (frames: number[]) => void;
  refreshMaskUrl: () => void;

  // Undo/redo
  snapshotBeforeEdit: (frameIdx: number) => Promise<void>;
  canUndo: () => boolean;
  canRedo: () => boolean;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  clearHistory: () => void;
}

export const useRoiStore = create<RoiState>((set, get) => ({
  drawingMode: null,
  cutMode: false,
  currentPoints: [],
  maskUrl: null,
  showImportDialog: false,
  showBatchImportDialog: false,
  editingFrameIdx: 0,
  framesWithRoi: [],
  undoStack: [],
  redoStack: [],

  setDrawingMode: (mode) => set({ drawingMode: mode, currentPoints: [] }),
  setCutMode: (cut) => set({ cutMode: cut }),
  activateTool: (shape, cut) => {
    const s = get();
    // Toggle off if same tool re-clicked
    if (s.drawingMode === shape && s.cutMode === cut) {
      set({ drawingMode: null, currentPoints: [] });
    } else {
      set({ drawingMode: shape, cutMode: cut, currentPoints: [] });
    }
  },
  addPoint: (x, y) =>
    set((s) => ({ currentPoints: [...s.currentPoints, [x, y]] })),
  clearPoints: () => set({ currentPoints: [] }),
  setMaskUrl: (url) => set({ maskUrl: url }),
  setShowImportDialog: (v) => set({ showImportDialog: v }),
  setShowBatchImportDialog: (v) => set({ showBatchImportDialog: v }),

  setEditingFrameIdx: (idx) => {
    set({ editingFrameIdx: idx, currentPoints: [], drawingMode: null });
    // Update mask URL for the new frame
    get().refreshMaskUrl();
  },
  setFramesWithRoi: (frames) => set({ framesWithRoi: frames }),
  refreshMaskUrl: () => {
    const idx = get().editingFrameIdx;
    set({ maskUrl: `/api/roi/mask?frame_idx=${idx}&t=${Date.now()}` });
  },

  /**
   * Capture the current (pre-edit) mask for `frameIdx` onto the undo stack
   * and drop any queued redo states.  Call this *before* each mutating
   * operation so Ctrl+Z restores what the user saw just before clicking.
   */
  snapshotBeforeEdit: async (frameIdx: number) => {
    try {
      const blob = await fetchMaskBlob(frameIdx);
      set((s) => {
        const next = [...s.undoStack, { frameIdx, blob }];
        const overflow = Math.max(0, next.length - HISTORY_LIMIT);
        return {
          undoStack: overflow > 0 ? next.slice(overflow) : next,
          redoStack: [], // new edit invalidates any redo future
        };
      });
      // Every snapshot corresponds to an upcoming mutation — flag the ROI
      // as changed so downstream result pages show the stale banner.
      markRoiChanged();
    } catch {
      // Non-fatal: worst case one edit is non-undoable.
    }
  },

  canUndo: () => get().undoStack.length > 0,
  canRedo: () => get().redoStack.length > 0,

  undo: async () => {
    const s = get();
    if (s.undoStack.length === 0) return;
    const entry = s.undoStack[s.undoStack.length - 1];
    try {
      // Push the current state to redo BEFORE overwriting it.
      const currentBlob = await fetchMaskBlob(entry.frameIdx);
      await uploadMask(entry.frameIdx, entry.blob);
      set((st) => ({
        undoStack: st.undoStack.slice(0, -1),
        redoStack: [...st.redoStack, { frameIdx: entry.frameIdx, blob: currentBlob }],
      }));
      if (s.editingFrameIdx !== entry.frameIdx) {
        set({ editingFrameIdx: entry.frameIdx });
      }
      get().refreshMaskUrl();
      markRoiChanged();
    } catch {
      // Leave stacks intact if the upload failed.
    }
  },

  redo: async () => {
    const s = get();
    if (s.redoStack.length === 0) return;
    const entry = s.redoStack[s.redoStack.length - 1];
    try {
      const currentBlob = await fetchMaskBlob(entry.frameIdx);
      await uploadMask(entry.frameIdx, entry.blob);
      set((st) => ({
        redoStack: st.redoStack.slice(0, -1),
        undoStack: [...st.undoStack, { frameIdx: entry.frameIdx, blob: currentBlob }],
      }));
      if (s.editingFrameIdx !== entry.frameIdx) {
        set({ editingFrameIdx: entry.frameIdx });
      }
      get().refreshMaskUrl();
      markRoiChanged();
    } catch {
      /* ignore */
    }
  },

  clearHistory: () => set({ undoStack: [], redoStack: [] }),
}));
