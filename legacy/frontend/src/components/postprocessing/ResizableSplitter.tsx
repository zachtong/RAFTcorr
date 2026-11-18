import { useCallback, useRef, useEffect } from "react";

interface ResizableSplitterProps {
  /** Callback with new top ratio (0-1) when user drags the splitter */
  onRatioChange: (ratio: number) => void;
  /** Reference to the parent container to compute relative positions */
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** Minimum top section height in pixels */
  minTopPx?: number;
  /** Minimum bottom section height in pixels */
  minBottomPx?: number;
}

export function ResizableSplitter({
  onRatioChange,
  containerRef,
  minTopPx = 120,
  minBottomPx = 120,
}: ResizableSplitterProps) {
  const dragging = useRef(false);

  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const totalH = rect.height;
      if (totalH <= 0) return;

      const y = e.clientY - rect.top;
      const minTop = minTopPx / totalH;
      const maxTop = 1 - minBottomPx / totalH;
      const ratio = Math.max(minTop, Math.min(maxTop, y / totalH));
      onRatioChange(ratio);
    },
    [containerRef, minTopPx, minBottomPx, onRatioChange]
  );

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
    return () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  }, []);

  return (
    <div
      onMouseDown={onMouseDown}
      className="h-[5px] cursor-row-resize flex-shrink-0 bg-[var(--border)] hover:bg-[var(--primary)] transition-colors duration-150"
    />
  );
}
