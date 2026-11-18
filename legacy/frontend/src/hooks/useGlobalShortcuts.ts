import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useRoiStore } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";

/**
 * Global keyboard shortcuts.
 *
 * Space, ←, → are already handled inside useFrameNav (playback); here we add
 * the rest:
 *   Ctrl/Cmd + Z          ROI undo
 *   Ctrl/Cmd + Shift + Z  ROI redo (Ctrl + Y also supported)
 *   Shift + ← / →         Jump 10 frames
 *   1 / 2 / 3             Switch to ROI / Displacement / Post-Processing
 *   ?                     Toggle the shortcut help modal
 *   Escape                Cancel active probe-placement mode
 *
 * Typing in inputs / textareas is always respected — the handler short-
 * circuits when an input element has focus.
 */
export function useGlobalShortcuts() {
  const navigate = useNavigate();
  const location = useLocation();
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target?.isContentEditable
      ) {
        return;
      }

      const mod = e.ctrlKey || e.metaKey;

      // ROI undo/redo — only active on the ROI page
      if (mod && !e.shiftKey && (e.key === "z" || e.key === "Z")) {
        if (location.pathname === "/") {
          e.preventDefault();
          useRoiStore.getState().undo();
        }
        return;
      }
      if (
        (mod && e.shiftKey && (e.key === "z" || e.key === "Z")) ||
        (mod && (e.key === "y" || e.key === "Y"))
      ) {
        if (location.pathname === "/") {
          e.preventDefault();
          useRoiStore.getState().redo();
        }
        return;
      }

      // Jump 10 frames with Shift + Arrow on result pages
      if (e.shiftKey && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
        const store = useAppStore.getState();
        if (store.numFrames > 0) {
          e.preventDefault();
          const delta = e.key === "ArrowRight" ? 10 : -10;
          const next = Math.max(
            0,
            Math.min(store.currentFrame + delta, store.numFrames - 1),
          );
          store.setCurrentFrame(next);
        }
        return;
      }

      // Page switches — 1/2/3 follow WorkflowStepper gating (disable when
      // prerequisites aren't met to match the UI).
      if (!mod && !e.shiftKey && !e.altKey) {
        if (e.key === "1") {
          e.preventDefault();
          navigate("/");
          return;
        }
        if (e.key === "2") {
          const st = useAppStore.getState();
          if (st.roiConfirmed) {
            e.preventDefault();
            navigate("/displacement");
          }
          return;
        }
        if (e.key === "3") {
          const st = useAppStore.getState();
          if (st.hasResults) {
            e.preventDefault();
            navigate("/post-processing");
          }
          return;
        }
        if (e.key === "?") {
          e.preventDefault();
          setHelpOpen((v) => !v);
          return;
        }
        if (e.key === "Escape") {
          const st = useAppStore.getState();
          if (st.probePlacingMode) {
            st.setProbePlacingMode(null);
            st.clearAreaPolyPoints();
          }
          return;
        }
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [navigate, location.pathname]);

  return { helpOpen, setHelpOpen };
}
