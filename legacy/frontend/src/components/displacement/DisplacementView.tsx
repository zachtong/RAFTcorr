import { useState, useEffect, useRef, useCallback } from "react";
import { useAppStore } from "@/stores/appStore";
import { renderUrl } from "@/api/displacement";
import { ColorbarOverlay } from "@/components/shared/ColorbarOverlay";
import { useColorRange } from "@/hooks/useColorRange";
import { useSyncedImagePair } from "@/hooks/useSyncedImagePair";
import type { PreRenderState } from "@/hooks/usePreRenderCache";
import { ImageIcon, Loader2 } from "lucide-react";

function DisplacementPanel({
  label,
  component,
  cache,
}: {
  label: string;
  component: "u" | "v";
  cache?: PreRenderState;
}) {
  const currentFrame = useAppStore((s) => s.currentFrame);
  const hasResults = useAppStore((s) => s.hasResults);
  const vis = useAppStore((s) => s.visSettings);
  const colorbarSettings = useAppStore((s) => s.colorbarSettings);
  const viewZoom = useAppStore((s) => s.viewZoom);
  const viewOffset = useAppStore((s) => s.viewOffset);
  const referenceFrame = useAppStore((s) => s.referenceFrame);
  const resultVersion = useAppStore((s) => s.resultVersion);
  const imageWidth = useAppStore((s) => s.imageWidth);
  const imageHeight = useAppStore((s) => s.imageHeight);
  const numFrames = useAppStore((s) => s.numFrames);

  // Track container size for viewport downsampling
  const panelRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setContainerSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Loading indicator — only show spinner after 150ms delay
  const [imgLoading, setImgLoading] = useState(false);
  const loadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    loadTimerRef.current = setTimeout(() => setImgLoading(true), 150);
    return () => {
      if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
    };
  }, [currentFrame]);

  const vmin = component === "u" ? vis.vminU : vis.vminV;
  const vmax = component === "u" ? vis.vmaxU : vis.vmaxV;
  const autoRange = useColorRange(currentFrame, component, hasResults);

  // Overlay URL: overlay_only mode + viewport params, no alpha
  const overlaySrc = (() => {
    const cachedSrc = cache?.getFrame(currentFrame);
    if (cachedSrc) return cachedSrc;
    return renderUrl(currentFrame, {
      component,
      colormap: vis.colormap,
      background: vis.background,
      overlay_only: "true",
      ...(containerSize.w > 0 ? { vw: containerSize.w } : {}),
      ...(containerSize.h > 0 ? { vh: containerSize.h } : {}),
      ...(vis.fixedRange && vmin ? { vmin } : {}),
      ...(vis.fixedRange && vmax ? { vmax } : {}),
      ...(referenceFrame > 0 ? { ref_frame: referenceFrame } : {}),
      _v: resultVersion,
    });
  })();

  // Background URL
  const bgSrc =
    vis.background === "deformed"
      ? `/api/images/frame/${currentFrame + 1}`
      : "/api/images/reference";

  // Sync bg + overlay so both swap in the same paint (prevents deformed-mode desync)
  const synced = useSyncedImagePair(bgSrc, overlaySrc, vis.background === "deformed");

  return (
    <div
      ref={panelRef}
      className="relative flex-1 overflow-hidden flex items-center justify-center bg-[var(--background)]"
    >
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-[var(--card)]/80 px-3 py-1 rounded text-[11px] text-[var(--foreground)]">
        {label}
      </div>
      {/* Background layer */}
      <img
        src={synced.bg}
        alt="background"
        className="max-w-full max-h-full object-contain"
        style={{
          transform: `translate(${viewOffset.x}px, ${viewOffset.y}px) scale(${viewZoom})`,
          transformOrigin: "0 0",
        }}
        draggable={false}
      />
      {/* Overlay layer — alpha applied via CSS opacity */}
      <img
        src={synced.overlay}
        alt={label}
        className="absolute inset-0 max-w-full max-h-full object-contain m-auto"
        style={{
          transform: `translate(${viewOffset.x}px, ${viewOffset.y}px) scale(${viewZoom})`,
          transformOrigin: "0 0",
          opacity: vis.alpha,
        }}
        draggable={false}
        onLoad={() => {
          if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
          setImgLoading(false);
        }}
        onError={() => {
          if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
          setImgLoading(false);
        }}
      />
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
        unit={
          vis.physicalEnabled
            ? `[${vis.physicalUnit}]`
            : "[px]"
        }
        scaleFactor={vis.physicalEnabled ? vis.physicalRatio : 1}
        component={component}
        colorbarSettings={colorbarSettings}
      />
    </div>
  );
}

interface DisplacementViewProps {
  cacheU?: PreRenderState;
  cacheV?: PreRenderState;
}

export function DisplacementView({ cacheU, cacheV }: DisplacementViewProps) {
  const hasResults = useAppStore((s) => s.hasResults);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const numFrames = useAppStore((s) => s.numFrames);
  const viewOffset = useAppStore((s) => s.viewOffset);
  const setViewOffset = useAppStore((s) => s.setViewOffset);

  const containerRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);

  const handlePanDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 1) {
      e.preventDefault();
      panRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        origX: viewOffset.x,
        origY: viewOffset.y,
      };
    }
  }, [viewOffset]);

  const handlePanMove = useCallback((e: React.MouseEvent) => {
    if (!panRef.current) return;
    setViewOffset({
      x: panRef.current.origX + (e.clientX - panRef.current.startX),
      y: panRef.current.origY + (e.clientY - panRef.current.startY),
    });
  }, [setViewOffset]);

  const handlePanUp = useCallback(() => {
    panRef.current = null;
  }, []);

  // Attach native wheel listener with { passive: false } so preventDefault() works
  useEffect(() => {
    const el = containerRef.current;
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

  if (!hasResults) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-[var(--muted-foreground)]">
        <ImageIcon className="w-12 h-12 mb-3 opacity-40" />
        <p className="text-[13px]">No displacement results</p>
        <p className="text-[11px] opacity-60">
          Run processing from the ROI Selection page first
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Header info */}
      <div className="flex items-center justify-between px-4 py-1.5 bg-[var(--card)] border-b border-[var(--border)] shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--primary)]" />
          <span className="text-[11px] text-[var(--foreground)]">
            Displacement Field Overlay
          </span>
        </div>
        <span className="text-[11px] text-[var(--muted-foreground)]">
          Frame: {currentFrame + 1} / {numFrames}
        </span>
      </div>

      {/* Side-by-side U/V panels — pan & zoom handled at this level */}
      <div
        ref={containerRef}
        className="flex-1 flex min-h-0 overflow-hidden"
        onMouseDown={handlePanDown}
        onMouseMove={handlePanMove}
        onMouseUp={handlePanUp}
        onMouseLeave={handlePanUp}
      >
        <DisplacementPanel label="U COMPONENT" component="u" cache={cacheU} />
        <div className="w-px bg-[var(--border)]" />
        <DisplacementPanel label="V COMPONENT" component="v" cache={cacheV} />
      </div>
    </div>
  );
}
