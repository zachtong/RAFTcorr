import { useState, useRef, useCallback, useEffect } from "react";
import { SmallInput } from "@/components/shared/SmallInput";

interface KeyFrameTimelineProps {
  totalFrames: number;
  keyFrames: number[]; // 1-indexed
  onChange: (kf: number[]) => void;
}

export function KeyFrameTimeline({
  totalFrames,
  keyFrames,
  onChange,
}: KeyFrameTimelineProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const [everyN, setEveryN] = useState("10");
  const [addFrame, setAddFrame] = useState("");

  // Hover state
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const [hoverX, setHoverX] = useState(0);
  const [hoverIsExisting, setHoverIsExisting] = useState(false);

  // Drag state — isDragging has both a ref (for sync event checks) and state (for rendering)
  const [dragFrame, setDragFrame] = useState<number | null>(null);
  const [dragTargetFrame, setDragTargetFrame] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragOriginalRef = useRef<number | null>(null);
  const dragStartXRef = useRef(0);
  const isDraggingRef = useRef(false);
  const dragTargetRef = useRef<number | null>(null);

  const sorted = [...keyFrames].sort((a, b) => a - b);

  // Tolerance for "near existing marker" detection
  const tolerance = Math.max(1, Math.round(totalFrames * 0.015));

  const frameFromClientX = useCallback(
    (clientX: number): number | null => {
      if (!barRef.current || totalFrames <= 1) return null;
      const rect = barRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const fraction = x / rect.width;
      return Math.max(
        1,
        Math.min(totalFrames, Math.round(fraction * (totalFrames - 1) + 1))
      );
    },
    [totalFrames]
  );

  const pctFromFrame = (frame: number) =>
    totalFrames <= 1 ? 0 : ((frame - 1) / (totalFrames - 1)) * 100;

  // --- Hover ---
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (isDraggingRef.current) return;
      if (!barRef.current) return;
      const rect = barRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const frame = frameFromClientX(e.clientX);
      if (frame === null) return;

      const existing = keyFrames.find(
        (kf) => Math.abs(kf - frame) <= tolerance
      );
      setHoverFrame(frame);
      setHoverX(x);
      setHoverIsExisting(existing !== undefined);
    },
    [frameFromClientX, keyFrames, tolerance]
  );

  const handleMouseLeave = useCallback(() => {
    if (!isDraggingRef.current) {
      setHoverFrame(null);
    }
  }, []);

  // --- Click (add / remove) ---
  const handleBarClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      // Ignore if we just finished a drag
      if (isDraggingRef.current) return;
      if (!barRef.current || totalFrames <= 1) return;

      const frame = frameFromClientX(e.clientX);
      if (frame === null) return;

      const existing = keyFrames.find(
        (kf) => Math.abs(kf - frame) <= tolerance
      );

      if (existing !== undefined && existing !== 1) {
        onChange(keyFrames.filter((kf) => kf !== existing));
      } else if (!keyFrames.includes(frame)) {
        onChange([...keyFrames, frame].sort((a, b) => a - b));
      }
    },
    [totalFrames, keyFrames, onChange, frameFromClientX, tolerance]
  );

  // --- Drag ---
  const handleMarkerMouseDown = useCallback(
    (e: React.MouseEvent, kf: number) => {
      if (kf === 1) return; // Frame 1 is locked
      e.stopPropagation();
      e.preventDefault();
      dragOriginalRef.current = kf;
      dragTargetRef.current = kf;
      dragStartXRef.current = e.clientX;
      isDraggingRef.current = false; // will become true once moved enough
      setDragFrame(kf);
      setDragTargetFrame(kf);
    },
    []
  );

  useEffect(() => {
    if (dragFrame === null) return;

    const handleDocMouseMove = (e: MouseEvent) => {
      const dx = Math.abs(e.clientX - dragStartXRef.current);
      if (!isDraggingRef.current && dx < 3) return; // threshold
      isDraggingRef.current = true;
      setIsDragging(true);

      const target = frameFromClientX(e.clientX);
      if (target === null || target === 1) return;
      dragTargetRef.current = target;
      setDragTargetFrame(target);
    };

    const handleDocMouseUp = () => {
      const original = dragOriginalRef.current;
      const wasDragging = isDraggingRef.current;

      if (wasDragging && original !== null) {
        // Commit the drag
        const target = dragTargetRef.current;
        if (
          target !== null &&
          target !== original &&
          target !== 1 &&
          !keyFrames.includes(target)
        ) {
          const next = keyFrames
            .map((kf) => (kf === original ? target : kf))
            .sort((a, b) => a - b);
          onChange(next);
        }
      }

      // Reset drag state
      setDragFrame(null);
      setDragTargetFrame(null);
      setIsDragging(false);
      dragOriginalRef.current = null;

      // Delay clearing isDragging ref so the click handler can check it
      requestAnimationFrame(() => {
        isDraggingRef.current = false;
      });
    };

    document.addEventListener("mousemove", handleDocMouseMove);
    document.addEventListener("mouseup", handleDocMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleDocMouseMove);
      document.removeEventListener("mouseup", handleDocMouseUp);
    };
  }, [dragFrame, keyFrames, onChange, frameFromClientX]);

  // --- Every N / Add / Clear ---
  const handleApplyEveryN = () => {
    const n = parseInt(everyN, 10);
    if (isNaN(n) || n < 1) return;
    const frames: number[] = [];
    for (let f = 1; f <= totalFrames; f += n) {
      frames.push(f);
    }
    onChange(frames);
  };

  const handleAddFrame = () => {
    const f = parseInt(addFrame, 10);
    if (isNaN(f) || f < 1 || f > totalFrames) return;
    if (!keyFrames.includes(f)) {
      onChange([...keyFrames, f].sort((a, b) => a - b));
    }
    setAddFrame("");
  };

  const handleClear = () => {
    onChange([1]);
  };

  // --- Tick marks ---
  const tickInterval = totalFrames <= 20 ? 1 : totalFrames <= 100 ? 10 : totalFrames <= 500 ? 50 : 100;
  const ticks: number[] = [];
  for (let t = 1; t <= totalFrames; t += tickInterval) {
    ticks.push(t);
  }
  // Always include last frame
  if (ticks[ticks.length - 1] !== totalFrames) ticks.push(totalFrames);

  // Build the set of visible markers (swap original for target during drag)
  const visibleMarkers = sorted.map((kf) => {
    if (dragFrame !== null && kf === dragFrame && isDragging) {
      return { frame: dragTargetFrame ?? kf, original: kf, isDragging: true };
    }
    return { frame: kf, original: kf, isDragging: false };
  });

  return (
    <div className="flex flex-col gap-1.5 w-full">
      {/* Timeline bar */}
      <div className="relative" style={{ paddingTop: 18 }}>
        {/* Tooltip for hover / drag */}
        {hoverFrame !== null && dragFrame === null && (
          <div
            className="absolute pointer-events-none select-none"
            style={{
              left: hoverX,
              top: 0,
              transform: "translateX(-50%)",
            }}
          >
            <span
              className="text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap"
              style={{
                backgroundColor: hoverIsExisting
                  ? "rgba(239,68,68,0.9)"
                  : "rgba(59,130,246,0.9)",
                color: "#fff",
              }}
            >
              {hoverIsExisting
                ? `Remove #${hoverFrame}`
                : `Add #${hoverFrame}`}
            </span>
          </div>
        )}

        {/* Tooltip for drag target */}
        {dragFrame !== null && dragTargetFrame !== null && isDragging && (
          <div
            className="absolute pointer-events-none select-none"
            style={{
              left: `${pctFromFrame(dragTargetFrame)}%`,
              top: 0,
              transform: "translateX(-50%)",
            }}
          >
            <span
              className="text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap"
              style={{
                backgroundColor: "rgba(234,179,8,0.9)",
                color: "#000",
              }}
            >
              Move → #{dragTargetFrame}
            </span>
          </div>
        )}

        {/* The bar itself */}
        <div
          ref={barRef}
          onClick={handleBarClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="relative h-6 bg-[var(--input)] border border-[#3a3d45] rounded overflow-visible"
          style={{ cursor: dragFrame !== null ? "grabbing" : "crosshair" }}
        >
          {/* Tick marks */}
          {totalFrames > 1 &&
            ticks.map((t) => {
              const pct = pctFromFrame(t);
              return (
                <div
                  key={`tick-${t}`}
                  className="absolute pointer-events-none"
                  style={{
                    left: `${pct}%`,
                    bottom: 0,
                    width: 1,
                    height: t === 1 || t === totalFrames || t % (tickInterval * 5) === 0 ? 8 : 4,
                    backgroundColor: "var(--muted-foreground)",
                    opacity: 0.25,
                  }}
                />
              );
            })}

          {/* Ghost marker (hover preview) */}
          {hoverFrame !== null && dragFrame === null && !hoverIsExisting && (
            <div
              className="absolute top-0 bottom-0 pointer-events-none"
              style={{
                left: `calc(${pctFromFrame(hoverFrame)}% - 1px)`,
                width: 3,
                backgroundColor: "rgba(59,130,246,0.5)",
                borderRadius: 1,
              }}
            />
          )}

          {/* Key frame markers */}
          {totalFrames > 1 &&
            visibleMarkers.map(({ frame, original, isDragging: markerDragging }) => {
              const pct = pctFromFrame(frame);
              const isFrame1 = original === 1;
              const isHovered =
                hoverFrame !== null &&
                !markerDragging &&
                Math.abs(frame - hoverFrame) <= tolerance;
              return (
                <div
                  key={original}
                  onMouseDown={(e) => handleMarkerMouseDown(e, original)}
                  className="absolute group"
                  style={{
                    left: `${pct}%`,
                    top: 0,
                    bottom: 0,
                    width: 16,
                    transform: "translateX(-8px)",
                    cursor: isFrame1
                      ? "default"
                      : markerDragging
                      ? "grabbing"
                      : "grab",
                    zIndex: markerDragging ? 20 : isHovered ? 10 : 1,
                  }}
                >
                  {/* Visual marker */}
                  <div
                    className="absolute left-1/2 top-0 bottom-0 rounded-sm transition-colors duration-100"
                    style={{
                      width: markerDragging ? 5 : isHovered ? 5 : 3,
                      transform: "translateX(-50%)",
                      backgroundColor: isFrame1
                        ? "var(--primary)"
                        : markerDragging
                        ? "rgb(234,179,8)"
                        : isHovered
                        ? "rgb(239,68,68)"
                        : "var(--foreground)",
                      opacity: isFrame1 ? 1 : markerDragging ? 1 : isHovered ? 1 : 0.8,
                      boxShadow: markerDragging ? "0 0 6px rgba(234,179,8,0.6)" : "none",
                    }}
                  />
                  {/* Diamond handle at top */}
                  <div
                    className="absolute left-1/2 transition-colors duration-100"
                    style={{
                      top: -2,
                      width: 8,
                      height: 8,
                      transform: "translateX(-50%) rotate(45deg)",
                      backgroundColor: isFrame1
                        ? "var(--primary)"
                        : markerDragging
                        ? "rgb(234,179,8)"
                        : isHovered
                        ? "rgb(239,68,68)"
                        : "var(--foreground)",
                      opacity: isFrame1 ? 1 : 0.9,
                      borderRadius: 1,
                      boxShadow: markerDragging ? "0 0 6px rgba(234,179,8,0.6)" : "none",
                    }}
                  />
                </div>
              );
            })}
        </div>
      </div>

      {/* Summary */}
      <span className="text-[10px] text-[var(--muted-foreground)]">
        {sorted.length} key frame{sorted.length !== 1 ? "s" : ""}
        {sorted.length <= 6
          ? `: ${sorted.join(", ")}`
          : `: ${sorted.slice(0, 5).join(", ")}...${sorted[sorted.length - 1]}`}
      </span>

      {/* Every N row */}
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)] shrink-0">
          Every
        </span>
        <SmallInput
          value={everyN}
          onChange={setEveryN}
          className="w-10"
        />
        <button
          onClick={handleApplyEveryN}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)]"
        >
          Apply
        </button>
      </div>

      {/* Add specific + Clear row */}
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)] shrink-0">
          Frame
        </span>
        <SmallInput
          value={addFrame}
          onChange={setAddFrame}
          className="w-10"
          placeholder="#"
        />
        <button
          onClick={handleAddFrame}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)]"
        >
          Add
        </button>
        <button
          onClick={handleClear}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)]"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
