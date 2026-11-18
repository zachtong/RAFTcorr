import { useRef, useState, useCallback, useEffect } from "react";
import { useRoiStore } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";
import { referenceImageUrl, frameImageUrl } from "@/api/images";
import { addPolygon, addRectangle, addCircle, confirmRoi } from "@/api/roi";
import { ImageIcon } from "lucide-react";

export function RoiCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageFiles = useAppStore((s) => s.imageFiles);
  const imageWidth = useAppStore((s) => s.imageWidth);
  const imageHeight = useAppStore((s) => s.imageHeight);
  const setRoiConfirmed = useAppStore((s) => s.setRoiConfirmed);

  const showTileGrid = useAppStore((s) => s.showTileGrid);
  const tilingPreview = useAppStore((s) => s.tilingPreview);

  const drawingMode = useRoiStore((s) => s.drawingMode);
  const cutMode = useRoiStore((s) => s.cutMode);
  const currentPoints = useRoiStore((s) => s.currentPoints);
  const addPoint = useRoiStore((s) => s.addPoint);
  const clearPoints = useRoiStore((s) => s.clearPoints);
  const maskUrl = useRoiStore((s) => s.maskUrl);
  const setMaskUrl = useRoiStore((s) => s.setMaskUrl);
  const editingFrameIdx = useRoiStore((s) => s.editingFrameIdx);
  const refreshMaskUrl = useRoiStore((s) => s.refreshMaskUrl);
  const snapshotBeforeEdit = useRoiStore((s) => s.snapshotBeforeEdit);

  // Drawing colors based on add/cut mode
  const strokeColor = cutMode ? "#ef4444" : "#3b82f6";
  const fillColor = cutMode
    ? "rgba(239,68,68,0.15)"
    : "rgba(59,130,246,0.15)";

  // Pan and zoom
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Temporary shape drawing state (for rect/circle second click)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Fit image to container on load
  useEffect(() => {
    if (!containerRef.current || !imageWidth || !imageHeight) return;
    const cw = containerRef.current.clientWidth;
    const ch = containerRef.current.clientHeight;
    const s = Math.min(cw / imageWidth, ch / imageHeight, 1) * 0.9;
    setScale(s);
    setOffset({
      x: (cw - imageWidth * s) / 2,
      y: (ch - imageHeight * s) / 2,
    });
  }, [imageWidth, imageHeight, imageFiles.length]);

  // Escape key cancels current drawing
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && drawingMode) {
        clearPoints();
        setMousePos(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [drawingMode, clearPoints]);

  // Convert screen coords to image pixel coords
  const screenToImage = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      if (!containerRef.current) return [0, 0];
      const rect = containerRef.current.getBoundingClientRect();
      const sx = clientX - rect.left;
      const sy = clientY - rect.top;
      const ix = (sx - offset.x) / scale;
      const iy = (sy - offset.y) / scale;
      return [Math.round(ix), Math.round(iy)];
    },
    [scale, offset]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Middle mouse = pan
      if (e.button === 1) {
        e.preventDefault();
        setIsPanning(true);
        setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
        return;
      }
    },
    [offset]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isPanning) {
        setOffset({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
        return;
      }
      if (drawingMode && currentPoints.length > 0) {
        const [ix, iy] = screenToImage(e.clientX, e.clientY);
        setMousePos({ x: ix, y: iy });
      }
    },
    [isPanning, panStart, drawingMode, currentPoints.length, screenToImage]
  );

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const rect = containerRef.current!.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const newScale = Math.min(Math.max(scale * factor, 0.1), 10);
      setScale(newScale);
      setOffset({
        x: mx - ((mx - offset.x) / scale) * newScale,
        y: my - ((my - offset.y) / scale) * newScale,
      });
    },
    [scale, offset]
  );

  // After drawing a shape, refresh mask and confirm ROI (for frame 0)
  const afterDraw = useCallback(async () => {
    refreshMaskUrl();
    if (editingFrameIdx === 0) {
      await confirmRoi();
      setRoiConfirmed(true);
    }
  }, [editingFrameIdx, refreshMaskUrl, setRoiConfirmed]);

  const handleClick = useCallback(
    async (e: React.MouseEvent) => {
      if (e.button !== 0 || !drawingMode || imageFiles.length === 0) return;
      const [ix, iy] = screenToImage(e.clientX, e.clientY);
      const mode = cutMode ? "cut" : "add";

      if (drawingMode === "polygon") {
        addPoint(ix, iy);
      } else if (drawingMode === "rectangle") {
        if (currentPoints.length === 0) {
          addPoint(ix, iy);
        } else {
          const [x0, y0] = currentPoints[0];
          try {
            await snapshotBeforeEdit(editingFrameIdx);
            await addRectangle(
              Math.min(x0, ix), Math.min(y0, iy),
              Math.max(x0, ix), Math.max(y0, iy),
              mode, editingFrameIdx,
            );
            await afterDraw();
          } catch (err) {
            console.error("Failed to add rectangle:", err);
          }
          clearPoints();
          setMousePos(null);
        }
      } else if (drawingMode === "circle") {
        if (currentPoints.length === 0) {
          addPoint(ix, iy);
        } else {
          const [cx, cy] = currentPoints[0];
          const r = Math.sqrt((ix - cx) ** 2 + (iy - cy) ** 2);
          try {
            await snapshotBeforeEdit(editingFrameIdx);
            await addCircle(cx, cy, r, mode, editingFrameIdx);
            await afterDraw();
          } catch (err) {
            console.error("Failed to add circle:", err);
          }
          clearPoints();
          setMousePos(null);
        }
      }
    },
    [drawingMode, cutMode, imageFiles.length, currentPoints, screenToImage,
     addPoint, clearPoints, editingFrameIdx, afterDraw, snapshotBeforeEdit]
  );

  const handleDoubleClick = useCallback(async () => {
    if (drawingMode === "polygon" && currentPoints.length >= 3) {
      try {
        const mode = cutMode ? "cut" : "add";
        await snapshotBeforeEdit(editingFrameIdx);
        await addPolygon(currentPoints, mode, editingFrameIdx);
        await afterDraw();
      } catch (err) {
        console.error("Failed to add polygon:", err);
      }
      clearPoints();
      setMousePos(null);
    }
  }, [drawingMode, cutMode, currentPoints, clearPoints, editingFrameIdx, afterDraw, snapshotBeforeEdit]);

  // Convert image coords to screen for SVG
  const imgToScreen = (ix: number, iy: number) => ({
    x: ix * scale + offset.x,
    y: iy * scale + offset.y,
  });

  const hasImage = imageFiles.length > 0;

  // Image source: frame 0 uses reference image, others use frame endpoint
  const imageSrc = editingFrameIdx === 0
    ? `${referenceImageUrl()}?v=${encodeURIComponent(imageFiles[0] ?? "")}`
    : `${frameImageUrl(editingFrameIdx)}?v=${encodeURIComponent(imageFiles[editingFrameIdx] ?? "")}`;

  return (
    <div
      ref={containerRef}
      className="flex-1 relative overflow-hidden bg-[var(--background)]"
      style={{ cursor: drawingMode ? "crosshair" : "default" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onWheel={handleWheel}
      onContextMenu={(e) => e.preventDefault()}
    >
      {hasImage ? (
        <>
          {/* Frame image */}
          <img
            key={`frame-${editingFrameIdx}-${imageFiles[editingFrameIdx] ?? "ref"}`}
            src={imageSrc}
            alt={editingFrameIdx === 0 ? "Reference" : `Frame ${editingFrameIdx + 1}`}
            draggable={false}
            className="absolute top-0 left-0 select-none max-w-none"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: "0 0",
              imageRendering: scale > 2 ? "pixelated" : "auto",
            }}
          />

          {/* ROI mask overlay */}
          {maskUrl && (
            <img
              src={maskUrl}
              alt="ROI Mask"
              draggable={false}
              className="absolute top-0 left-0 select-none pointer-events-none max-w-none"
              style={{
                transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
                transformOrigin: "0 0",
              }}
            />
          )}

          {/* Frame indicator badge */}
          {editingFrameIdx > 0 && (
            <div className="absolute top-2 left-2 px-2 py-1 rounded bg-[var(--accent)] text-[var(--accent-foreground)] text-[11px] font-medium z-10">
              Editing Frame {editingFrameIdx + 1} ROI
            </div>
          )}

          {/* SVG overlay for active drawing */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {/* In-progress polygon lines */}
            {drawingMode === "polygon" &&
              currentPoints.length > 0 && (
                <>
                  <polyline
                    points={[
                      ...currentPoints.map((p) => {
                        const s = imgToScreen(p[0], p[1]);
                        return `${s.x},${s.y}`;
                      }),
                      ...(mousePos
                        ? [
                            (() => {
                              const s = imgToScreen(mousePos.x, mousePos.y);
                              return `${s.x},${s.y}`;
                            })(),
                          ]
                        : []),
                    ].join(" ")}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                  />
                  {currentPoints.map((p, i) => {
                    const s = imgToScreen(p[0], p[1]);
                    return (
                      <circle
                        key={i}
                        cx={s.x}
                        cy={s.y}
                        r={3}
                        fill={strokeColor}
                      />
                    );
                  })}
                </>
              )}

            {/* In-progress rectangle preview */}
            {drawingMode === "rectangle" &&
              currentPoints.length === 1 &&
              mousePos && (() => {
                const s0 = imgToScreen(currentPoints[0][0], currentPoints[0][1]);
                const s1 = imgToScreen(mousePos.x, mousePos.y);
                return (
                  <rect
                    x={Math.min(s0.x, s1.x)}
                    y={Math.min(s0.y, s1.y)}
                    width={Math.abs(s1.x - s0.x)}
                    height={Math.abs(s1.y - s0.y)}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                  />
                );
              })()}

            {/* In-progress circle preview */}
            {drawingMode === "circle" &&
              currentPoints.length === 1 &&
              mousePos && (() => {
                const center = imgToScreen(currentPoints[0][0], currentPoints[0][1]);
                const edge = imgToScreen(mousePos.x, mousePos.y);
                const r = Math.sqrt(
                  (edge.x - center.x) ** 2 + (edge.y - center.y) ** 2
                );
                return (
                  <circle
                    cx={center.x}
                    cy={center.y}
                    r={r}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                  />
                );
              })()}

            {/* Tile Grid Overlay */}
            {showTileGrid && tilingPreview && (() => {
              const wa = tilingPreview.work_area;

              return (
                <>
                  {/* Work area boundary (orange dashed) */}
                  {(() => {
                    const tl = imgToScreen(wa.x, wa.y);
                    return (
                      <rect
                        x={tl.x}
                        y={tl.y}
                        width={wa.w * scale}
                        height={wa.h * scale}
                        fill="none"
                        stroke="#f97316"
                        strokeWidth="1"
                        strokeDasharray="6 3"
                        opacity={0.7}
                      />
                    );
                  })()}

                  {/* Individual tiles */}
                  {tilingPreview.tiles.map((tile, i) => {
                    const [tx, ty, tw, th] = tile;
                    const screenPos = imgToScreen(wa.x + tx, wa.y + ty);
                    const screenW = tw * scale;
                    const screenH = th * scale;

                    return (
                      <g key={i}>
                        <rect
                          x={screenPos.x}
                          y={screenPos.y}
                          width={screenW}
                          height={screenH}
                          fill="none"
                          stroke="#22d3ee"
                          strokeWidth="1.5"
                          strokeDasharray="6 3"
                          opacity={0.8}
                        />
                        {/* Tile number */}
                        {screenW > 30 && screenH > 20 && (
                          <text
                            x={screenPos.x + screenW / 2}
                            y={screenPos.y + screenH / 2}
                            textAnchor="middle"
                            dominantBaseline="central"
                            fill="#22d3ee"
                            fontSize={Math.min(13, screenW / 3)}
                            fontWeight="bold"
                            opacity={0.7}
                            stroke="#000"
                            strokeWidth="2.5"
                            paintOrder="stroke"
                          >
                            #{i + 1}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </>
              );
            })()}
          </svg>
        </>
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-[var(--muted-foreground)]">
          <ImageIcon className="w-12 h-12 mb-3 opacity-40" />
          <p className="text-[13px]">No images loaded</p>
          <p className="text-[11px] opacity-60">
            Set input directory and load images to begin
          </p>
        </div>
      )}
    </div>
  );
}
