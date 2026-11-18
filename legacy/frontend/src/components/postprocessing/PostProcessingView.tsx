import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useAppStore } from "@/stores/appStore";
import { renderUrl } from "@/api/displacement";
import { strainRenderUrl } from "@/api/strain";
import { arrowRenderUrl } from "@/api/arrows";
import { principalRenderUrl } from "@/api/principal";
import { addPoint, addLine, addArea, listProbes } from "@/api/probes";
import { ColorbarOverlay } from "@/components/shared/ColorbarOverlay";
import { useColorRange } from "@/hooks/useColorRange";
import { useSyncedImagePair } from "@/hooks/useSyncedImagePair";
import type { PreRenderState } from "@/hooks/usePreRenderCache";
import type { StrainComponent } from "@/types/api";
import { ImageIcon, Loader2, Download } from "lucide-react";
import { downloadSingleFrame } from "@/api/export";
import { useToast } from "@/components/shared/Toast";
import { getUnitInfo } from "@/utils/unitConversion";

const STRAIN_COMPONENTS: string[] = [
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
  "dexx_dt", "deyy_dt", "dexy_dt",
];

interface PostProcessingViewProps {
  cache?: PreRenderState;
}

export function PostProcessingView({ cache }: PostProcessingViewProps) {
  const hasResults = useAppStore((s) => s.hasResults);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const numFrames = useAppStore((s) => s.numFrames);
  const displayComponent = useAppStore((s) => s.displayComponent);
  const vis = useAppStore((s) => s.visSettings);
  const colorbarSettings = useAppStore((s) => s.colorbarSettings);
  const arrows = useAppStore((s) => s.arrowSettings);
  const probes = useAppStore((s) => s.probes);
  const viewZoom = useAppStore((s) => s.viewZoom);
  const viewOffset = useAppStore((s) => s.viewOffset);
  const setViewOffset = useAppStore((s) => s.setViewOffset);

  const referenceFrame = useAppStore((s) => s.referenceFrame);
  const resultVersion = useAppStore((s) => s.resultVersion);
  const hasStrain = useAppStore((s) => s.hasStrain);
  const imageWidth = useAppStore((s) => s.imageWidth);
  const imageHeight = useAppStore((s) => s.imageHeight);

  const placingMode = useAppStore((s) => s.probePlacingMode);
  const placingFirst = useAppStore((s) => s.probePlacingFirst);
  const setPlacingMode = useAppStore((s) => s.setProbePlacingMode);
  const setPlacingFirst = useAppStore((s) => s.setProbePlacingFirst);
  const setProbes = useAppStore((s) => s.setProbes);
  const updateVis = useAppStore((s) => s.updateVisSettings);
  const areaPolyPoints = useAppStore((s) => s.areaPolyPoints);
  const addAreaPolyPoint = useAppStore((s) => s.addAreaPolyPoint);
  const clearAreaPolyPoints = useAppStore((s) => s.clearAreaPolyPoints);

  const { toast } = useToast();
  const imgRef = useRef<HTMLImageElement>(null);
  const imageAreaRef = useRef<HTMLDivElement>(null);
  // Remember the background before entering placement mode so we can restore it
  const savedBgRef = useRef<"reference" | "deformed" | null>(null);

  // Mouse position for live preview (local state — 60fps updates shouldn't trigger global re-renders)
  const [mousePos, setMousePos] = useState<[number, number] | null>(null);

  // Track container size via ResizeObserver for explicit pixel constraints on the image.
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = imageAreaRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setContainerSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const isVectorComponent = ["u", "v", "magnitude", "velocity"].includes(displayComponent);
  const showArrows =
    isVectorComponent &&
    (arrows.showQuiver || arrows.showStreamlines);

  const autoRange = useColorRange(currentFrame, displayComponent, hasResults);

  // Convert screen click coordinates to full-image pixel coordinates
  const screenToImage = useCallback(
    (clientX: number, clientY: number): [number, number] | null => {
      const img = imgRef.current;
      if (!img || imageWidth === 0 || imageHeight === 0) return null;
      const rect = img.getBoundingClientRect();
      const sx = clientX - rect.left;
      const sy = clientY - rect.top;
      // Map to full-image coords (not naturalWidth which may be viewport-downsampled)
      const scaleX = imageWidth / rect.width;
      const scaleY = imageHeight / rect.height;
      const ix = Math.round(sx * scaleX);
      const iy = Math.round(sy * scaleY);
      if (ix < 0 || ix >= imageWidth || iy < 0 || iy >= imageHeight) return null;
      return [ix, iy];
    }, [imageWidth, imageHeight]
  );

  // Track mouse position for live preview during area/line placement
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!placingMode?.startsWith("area") && placingMode !== "line") return;
    if (placingMode === "area-rect" || placingMode === "area-circle" || placingMode === "line") {
      if (!placingFirst) return;
    } else if (placingMode === "area-poly") {
      if (areaPolyPoints.length === 0) return;
    }
    const pt = screenToImage(e.clientX, e.clientY);
    if (pt) setMousePos(pt);
  }, [placingMode, placingFirst, areaPolyPoints.length, screenToImage]);

  // Handle click on image area for probe placement
  const handleImageClick = useCallback(async (e: React.MouseEvent) => {
    if (!placingMode) return;
    const pt = screenToImage(e.clientX, e.clientY);
    if (!pt) return;

    try {
      if (placingMode === "point") {
        await addPoint(pt[0], pt[1]);
        const updated = await listProbes();
        setProbes(updated);
        setPlacingMode(null);
      } else if (placingMode === "line") {
        if (!placingFirst) {
          setPlacingFirst(pt);
        } else {
          await addLine(placingFirst, pt);
          const updated = await listProbes();
          setProbes(updated);
          setPlacingMode(null);
          setMousePos(null);
        }
      } else if (placingMode === "area-rect") {
        if (!placingFirst) {
          setPlacingFirst(pt);
        } else {
          const [x0, y0] = placingFirst;
          const [x1, y1] = pt;
          await addArea("rect", [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]);
          const updated = await listProbes();
          setProbes(updated);
          setPlacingMode(null);
          setMousePos(null);
        }
      } else if (placingMode === "area-circle") {
        if (!placingFirst) {
          setPlacingFirst(pt);
        } else {
          const [cx, cy] = placingFirst;
          const r = Math.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2);
          await addArea("circle", [cx, cy, Math.round(r)]);
          const updated = await listProbes();
          setProbes(updated);
          setPlacingMode(null);
          setMousePos(null);
        }
      } else if (placingMode === "area-poly") {
        addAreaPolyPoint(pt[0], pt[1]);
      }
    } catch (err) {
      console.error("Failed to place probe:", err);
    }
  }, [placingMode, placingFirst, screenToImage, setProbes, setPlacingMode, setPlacingFirst, addAreaPolyPoint]);

  // Double-click closes polygon
  const handleDoubleClick = useCallback(async (e: React.MouseEvent) => {
    if (placingMode !== "area-poly" || areaPolyPoints.length < 3) return;
    e.preventDefault();
    try {
      await addArea("poly", areaPolyPoints);
      const updated = await listProbes();
      setProbes(updated);
    } catch (err) {
      console.error("Failed to place area poly:", err);
    }
    setPlacingMode(null);
    clearAreaPolyPoints();
    setMousePos(null);
  }, [placingMode, areaPolyPoints, setProbes, setPlacingMode, clearAreaPolyPoints]);

  // Force reference background during probe placement
  useEffect(() => {
    if (placingMode) {
      if (vis.background !== "reference") {
        savedBgRef.current = vis.background;
        updateVis({ background: "reference" });
      }
    } else {
      if (savedBgRef.current) {
        updateVis({ background: savedBgRef.current });
        savedBgRef.current = null;
      }
    }
  }, [placingMode, updateVis]); // deliberately omit vis.background to avoid loop

  // Escape key cancels placement mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && placingMode) {
        setPlacingMode(null);
        clearAreaPolyPoints();
        setMousePos(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [placingMode, setPlacingMode, clearAreaPolyPoints]);

  // --- Pan & scroll-wheel zoom ---
  const panRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);

  const handlePanDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 1) { // middle mouse
      e.preventDefault();
      panRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        origX: viewOffset.x,
        origY: viewOffset.y,
      };
    }
  }, [viewOffset]);

  const handleMouseMoveAll = useCallback((e: React.MouseEvent) => {
    if (panRef.current) {
      setViewOffset({
        x: panRef.current.origX + (e.clientX - panRef.current.startX),
        y: panRef.current.origY + (e.clientY - panRef.current.startY),
      });
      return;
    }
    handleMouseMove(e);
  }, [handleMouseMove, setViewOffset]);

  const handlePanUp = useCallback(() => {
    panRef.current = null;
  }, []);

  // Attach native wheel listener with { passive: false } so preventDefault() works
  useEffect(() => {
    const el = imageAreaRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const state = useAppStore.getState();
      const newZoom = Math.max(0.2, Math.min(5, state.viewZoom * factor));
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const newX = mx - ((mx - state.viewOffset.x) / state.viewZoom) * newZoom;
      const newY = my - ((my - state.viewOffset.y) / state.viewZoom) * newZoom;
      useAppStore.setState({ viewZoom: newZoom, viewOffset: { x: newX, y: newY } });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const { unit: colorbarUnit, scale: colorbarScale } = useMemo(
    () =>
      getUnitInfo(
        displayComponent,
        vis.physicalEnabled,
        vis.physicalRatio,
        vis.physicalUnit,
        vis.fps
      ),
    [displayComponent, vis.physicalEnabled, vis.physicalUnit, vis.physicalRatio, vis.fps]
  );

  // Loading indicator
  const [imgLoading, setImgLoading] = useState(false);
  const loadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    loadTimerRef.current = setTimeout(() => setImgLoading(true), 150);
    return () => {
      if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
    };
  }, [currentFrame, displayComponent]);

  if (!hasResults) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-[var(--muted-foreground)]">
        <ImageIcon className="w-12 h-12 mb-3 opacity-40" />
        <p className="text-[13px]">No results available</p>
        <p className="text-[11px] opacity-60">
          Run processing and navigate here to analyze results
        </p>
      </div>
    );
  }

  const handleSaveFrame = useCallback(async () => {
    try {
      const strain = STRAIN_COMPONENTS.includes(displayComponent);
      const saveVmin = displayComponent === "v" ? vis.vminV : vis.vminU;
      const saveVmax = displayComponent === "v" ? vis.vmaxV : vis.vmaxU;
      await downloadSingleFrame({
        idx: currentFrame,
        component: displayComponent,
        colormap: vis.colormap,
        alpha: vis.alpha,
        background: vis.background,
        log_scale: vis.logScale,
        dpi: 300,
        isStrain: strain,
        ...(vis.fixedRange && saveVmin ? { vmin: parseFloat(saveVmin) } : {}),
        ...(vis.fixedRange && saveVmax ? { vmax: parseFloat(saveVmax) } : {}),
      });
      toast("success", "Frame saved");
    } catch {
      toast("error", "Failed to save frame");
    }
  }, [currentFrame, displayComponent, vis, toast]);

  const isStrain = STRAIN_COMPONENTS.includes(displayComponent);
  const vmin = displayComponent === "v" ? vis.vminV : vis.vminU;
  const vmax = displayComponent === "v" ? vis.vmaxV : vis.vmaxU;
  const renderParams = {
    component: displayComponent,
    colormap: vis.colormap,
    background: vis.background,
    overlay_only: "true",
    ...(containerSize.w > 0 ? { vw: containerSize.w } : {}),
    ...(containerSize.h > 0 ? { vh: containerSize.h } : {}),
    ...(vis.fixedRange && vmin ? { vmin } : {}),
    ...(vis.fixedRange && vmax ? { vmax } : {}),
    ...(vis.logScale ? { log_scale: "true" } : {}),
    ...(referenceFrame > 0 ? { ref_frame: referenceFrame } : {}),
    _v: resultVersion,
  };

  const liveSrc = isStrain
    ? strainRenderUrl(currentFrame, renderParams)
    : renderUrl(currentFrame, renderParams);
  const cachedSrc = cache?.getFrame(currentFrame);
  const overlaySrc = cachedSrc ?? liveSrc;

  // Background image URL (separate from overlay)
  const bgSrc =
    vis.background === "deformed"
      ? `/api/images/frame/${currentFrame + 1}`
      : "/api/images/reference";

  // Sync bg + overlay so both swap in the same paint (prevents deformed-mode desync)
  const synced = useSyncedImagePair(bgSrc, overlaySrc, vis.background === "deformed");

  // Placement banner text
  const getBannerText = () => {
    if (placingMode === "point") return "Click to place point probe";
    if (placingMode === "line") return placingFirst ? "Click to place second endpoint" : "Click to place first endpoint";
    if (placingMode === "area-rect") return placingFirst ? "Click second corner" : "Click first corner of rectangle";
    if (placingMode === "area-circle") return placingFirst ? "Click to set radius" : "Click center of circle";
    if (placingMode === "area-poly") {
      return areaPolyPoints.length < 3
        ? `Click to add points (${areaPolyPoints.length}/3 min)`
        : `${areaPolyPoints.length} points — double-click to close`;
    }
    return "";
  };

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-1.5 bg-[var(--card)] border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--primary)]" />
          <span className="text-[11px] text-[var(--foreground)]">
            {isStrain ? "Strain Field" : "Displacement Field"} — {displayComponent.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-[var(--muted-foreground)]">
            Frame: {currentFrame + 1} / {numFrames}
          </span>
          <button
            onClick={handleSaveFrame}
            className="p-1 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            title="Save current frame (300 DPI)"
          >
            <Download size={14} />
          </button>
        </div>
      </div>

      {/* Main visualization */}
      <div
        ref={imageAreaRef}
        className="flex-1 relative bg-[var(--background)] overflow-hidden min-h-0"
        onMouseDown={handlePanDown}
        onMouseUp={handlePanUp}
        onMouseLeave={handlePanUp}
      >
        {/* Probe placement status banner — floats over the canvas while a
            probe-placing mode is active so the user always knows what click
            will do. */}
        {placingMode && (
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-20 px-3 py-1.5 rounded-md bg-[var(--primary)] text-white text-[11px] shadow-lg pointer-events-none flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
            {(() => {
              switch (placingMode) {
                case "point":
                  return "Click to place a point probe · Esc to cancel";
                case "line":
                  return placingFirst
                    ? "Click to set the second endpoint · Esc to cancel"
                    : "Click to set the first endpoint · Esc to cancel";
                case "area-rect":
                  return placingFirst
                    ? "Click to set the opposite rectangle corner · Esc to cancel"
                    : "Click to set the first rectangle corner · Esc to cancel";
                case "area-circle":
                  return placingFirst
                    ? "Click to set the circle radius · Esc to cancel"
                    : "Click to set the circle centre · Esc to cancel";
                case "area-poly":
                  return areaPolyPoints.length >= 3
                    ? "Double-click or Enter to close polygon · Esc to cancel"
                    : `Click to add polygon vertex (${areaPolyPoints.length} so far) · Esc to cancel`;
                default:
                  return "Click to place probe · Esc to cancel";
              }
            })()}
          </div>
        )}
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className="relative"
            style={{
              maxWidth: containerSize.w > 0 ? `${containerSize.w}px` : "100%",
              maxHeight: containerSize.h > 0 ? `${containerSize.h}px` : "100%",
              transform: `translate(${viewOffset.x}px, ${viewOffset.y}px) scale(${viewZoom})`,
              transformOrigin: "0 0",
              cursor: placingMode ? "crosshair" : undefined,
            }}
            onClick={handleImageClick}
            onDoubleClick={handleDoubleClick}
            onMouseMove={handleMouseMoveAll}
          >
            {/* Background layer */}
            <img
              ref={imgRef}
              src={synced.bg}
              alt="background"
              className="block max-w-full"
              style={{ maxHeight: containerSize.h > 0 ? `${containerSize.h}px` : undefined }}
              draggable={false}
            />
            {/* Overlay layer — alpha applied via CSS opacity */}
            <img
              src={synced.overlay}
              alt={`${displayComponent} frame ${currentFrame}`}
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ opacity: vis.alpha }}
              draggable={false}
              onLoad={() => {
                if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
                setImgLoading(false);
              }}
            />
            {showArrows && (
              <img
                src={arrowRenderUrl(currentFrame, {
                  show_quiver: arrows.showQuiver,
                  show_streamlines: arrows.showStreamlines,
                  vector_source: displayComponent === "velocity" ? "velocity" : "displacement",
                  spacing: arrows.spacing,
                  scale: arrows.scale,
                  color: arrows.color,
                  line_width: arrows.lineWidth,
                  background: vis.background,
                  stream_ds: arrows.streamQuality,
                  ...(containerSize.w > 0 ? { vw: containerSize.w } : {}),
                  ...(containerSize.h > 0 ? { vh: containerSize.h } : {}),
                })}
                alt="vector field arrows"
                className="absolute inset-0 w-full h-full pointer-events-none"
                draggable={false}
              />
            )}
            {arrows.showPrincipalDirs && hasStrain && (
              <img
                src={principalRenderUrl(currentFrame, {
                  spacing: arrows.spacing,
                  scale: arrows.scale,
                  line_width: arrows.lineWidth,
                  ...(containerSize.w > 0 ? { vw: containerSize.w } : {}),
                  ...(containerSize.h > 0 ? { vh: containerSize.h } : {}),
                })}
                alt="principal strain directions"
                className="absolute inset-0 w-full h-full pointer-events-none"
                draggable={false}
              />
            )}

            {/* SVG overlay for probe markers — use full image coords, not naturalWidth */}
            {imgRef.current && imageWidth > 0 && imageHeight > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${imageWidth} ${imageHeight}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {probes.map((probe) => {
                  if (probe.type === "point") {
                    const [x, y] = probe.coords as [number, number];
                    return (
                      <circle
                        key={`p-${probe.id}`}
                        cx={x} cy={y} r={5}
                        fill={probe.color} stroke="white" strokeWidth={1.5} opacity={0.9}
                      />
                    );
                  }
                  if (probe.type === "line") {
                    const [[x1, y1], [x2, y2]] = probe.coords as [[number, number], [number, number]];
                    return (
                      <g key={`l-${probe.id}`}>
                        <line x1={x1} y1={y1} x2={x2} y2={y2}
                          stroke={probe.color} strokeWidth={2} opacity={0.9} />
                        <circle cx={x1} cy={y1} r={4} fill={probe.color} stroke="white" strokeWidth={1} />
                        <circle cx={x2} cy={y2} r={4} fill={probe.color} stroke="white" strokeWidth={1} />
                      </g>
                    );
                  }
                  if (probe.type === "area") {
                    const { shape, data: d } = probe.coords as { shape: string; data: unknown };
                    if (shape === "rect") {
                      const [x0, y0, x1, y1] = d as number[];
                      return (
                        <rect key={`a-${probe.id}`}
                          x={x0} y={y0} width={x1 - x0} height={y1 - y0}
                          fill={probe.color} fillOpacity={0.15}
                          stroke={probe.color} strokeWidth={2} strokeDasharray="6 3" opacity={0.9}
                        />
                      );
                    }
                    if (shape === "circle") {
                      const [cx, cy, r] = d as number[];
                      return (
                        <circle key={`a-${probe.id}`}
                          cx={cx} cy={cy} r={r}
                          fill={probe.color} fillOpacity={0.15}
                          stroke={probe.color} strokeWidth={2} strokeDasharray="6 3" opacity={0.9}
                        />
                      );
                    }
                    if (shape === "poly") {
                      const pts = (d as [number, number][]).map(p => `${p[0]},${p[1]}`).join(" ");
                      return (
                        <polygon key={`a-${probe.id}`}
                          points={pts}
                          fill={probe.color} fillOpacity={0.15}
                          stroke={probe.color} strokeWidth={2} strokeDasharray="6 3" opacity={0.9}
                        />
                      );
                    }
                  }
                  return null;
                })}

                {/* Pending first endpoint for line placement */}
                {placingMode === "line" && placingFirst && (
                  <>
                    <circle cx={placingFirst[0]} cy={placingFirst[1]} r={5}
                      fill="none" stroke="white" strokeWidth={2} strokeDasharray="3,3" />
                    {mousePos && (
                      <line x1={placingFirst[0]} y1={placingFirst[1]} x2={mousePos[0]} y2={mousePos[1]}
                        stroke="white" strokeWidth={1.5} strokeDasharray="6 3" opacity={0.7} />
                    )}
                  </>
                )}

                {/* Live preview: rect */}
                {placingMode === "area-rect" && placingFirst && mousePos && (
                  <rect
                    x={Math.min(placingFirst[0], mousePos[0])}
                    y={Math.min(placingFirst[1], mousePos[1])}
                    width={Math.abs(mousePos[0] - placingFirst[0])}
                    height={Math.abs(mousePos[1] - placingFirst[1])}
                    fill="white" fillOpacity={0.1}
                    stroke="white" strokeWidth={1.5} strokeDasharray="6 3"
                  />
                )}

                {/* Live preview: circle */}
                {placingMode === "area-circle" && placingFirst && mousePos && (
                  <circle
                    cx={placingFirst[0]} cy={placingFirst[1]}
                    r={Math.sqrt((mousePos[0] - placingFirst[0]) ** 2 + (mousePos[1] - placingFirst[1]) ** 2)}
                    fill="white" fillOpacity={0.1}
                    stroke="white" strokeWidth={1.5} strokeDasharray="6 3"
                  />
                )}

                {/* Live preview: polygon */}
                {placingMode === "area-poly" && areaPolyPoints.length > 0 && (
                  <>
                    <polyline
                      points={[...areaPolyPoints, ...(mousePos ? [mousePos] : [])]
                        .map(p => `${p[0]},${p[1]}`).join(" ")}
                      fill="none" stroke="white" strokeWidth={1.5} strokeDasharray="6 3"
                    />
                    {areaPolyPoints.map((p, i) => (
                      <circle key={i} cx={p[0]} cy={p[1]} r={3} fill="white" />
                    ))}
                  </>
                )}

                {/* First point marker for rect/circle */}
                {(placingMode === "area-rect" || placingMode === "area-circle") && placingFirst && (
                  <circle cx={placingFirst[0]} cy={placingFirst[1]} r={4}
                    fill="none" stroke="white" strokeWidth={2} strokeDasharray="3,3" />
                )}
              </svg>
            )}
          </div>
        </div>

        {/* Placement mode indicator */}
        {placingMode && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-[var(--primary)]/90 px-3 py-1 rounded text-[11px] text-white">
            {getBannerText()}
            <span className="ml-2 opacity-70">(Esc to cancel)</span>
          </div>
        )}

        {/* Loading overlay */}
        {imgLoading && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm text-white px-4 py-2 rounded-lg">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-[12px] font-medium">Rendering frame...</span>
            </div>
          </div>
        )}

        <ColorbarOverlay
          colormap={vis.colormap}
          vmin={vis.fixedRange && vmin ? parseFloat(vmin) : autoRange?.vmin}
          vmax={vis.fixedRange && vmax ? parseFloat(vmax) : autoRange?.vmax}
          unit={colorbarUnit}
          scaleFactor={colorbarScale}
          logScale={vis.logScale}
          component={displayComponent}
          colorbarSettings={colorbarSettings}
        />
      </div>
    </div>
  );
}
