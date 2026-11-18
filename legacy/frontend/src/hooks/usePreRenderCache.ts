import { useState, useEffect, useRef, useCallback } from "react";
import { useAppStore } from "@/stores/appStore.ts";
import type { DataTexture } from "@/lib/applyColormap";
import { decodeDataTexture } from "@/lib/applyColormap";
import { getColormapLUT, prefetchColormapLUT } from "@/lib/colormapLUT";
import { colorizeInWorker, terminatePool } from "@/workers/colorizePool";

export interface PreRenderState {
  /** Blob URL for a given frame, or undefined if not yet cached */
  getFrame: (idx: number) => string | undefined;
  /** Whether pre-rendering is in progress */
  isPreRendering: boolean;
  /** Progress: 0-100 */
  progress: number;
  /** Number of frames cached */
  cachedCount: number;
  /** Total frames to cache */
  totalFrames: number;
  /** Start pre-rendering with current vis settings */
  startPreRender: () => void;
  /** Cancel in-progress pre-render */
  cancelPreRender: () => void;
  /** Invalidate cache (e.g. when settings change) */
  invalidate: () => void;
}

const STRAIN_COMPONENTS = new Set([
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
  "dexx_dt", "deyy_dt", "dexy_dt",
]);

/**
 * Build the URL for fetching a data texture from the server.
 *
 * Data textures exclude vmin/vmax/colormap/log_scale since those are
 * applied client-side.
 */
function buildDataTextureUrl(
  idx: number,
  component: string,
  params: Record<string, string | number>,
  isStrain: boolean,
): string {
  const base = isStrain ? "/api/strain/render" : "/api/displacement/render";
  const qs = new URLSearchParams();
  qs.set("component", component);
  qs.set("overlay_only", "true");
  qs.set("render_mode", "data");
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  return `${base}/${idx}?${qs}`;
}

/**
 * Two-tier pre-render cache for overlay images.
 *
 * Tier 1 (data textures): Fetched from server, keyed by data params only.
 *   Survives colormap/vmin/vmax changes.
 *
 * Tier 2 (rendered blobs): Colorized blob URLs, keyed by all params.
 *   Instantly re-generated from Tier 1 when color params change.
 */
export function usePreRenderCache(
  componentOverride?: string,
  viewportSize?: { vw: number; vh: number },
): PreRenderState {
  const numFrames = useAppStore((s) => s.numFrames);
  const hasResults = useAppStore((s) => s.hasResults);
  const storeComponent = useAppStore((s) => s.displayComponent);
  const vis = useAppStore((s) => s.visSettings);
  const referenceFrame = useAppStore((s) => s.referenceFrame);
  const resultVersion = useAppStore((s) => s.resultVersion);

  const displayComponent = componentOverride ?? storeComponent;

  const [isPreRendering, setIsPreRendering] = useState(false);
  const [progress, setProgress] = useState(0);
  const [cachedCount, setCachedCount] = useState(0);

  // Tier 1: data textures (survive color changes)
  const dataCacheRef = useRef<Map<number, DataTexture>>(new Map());
  // Tier 2: rendered blob URLs (invalidated on color changes)
  const renderCacheRef = useRef<Map<number, string>>(new Map());
  // Track in-flight lazy colorizations to prevent duplicate work
  const colorizingRef = useRef<Set<number>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  const dataKeyRef = useRef<string>("");
  const colorKeyRef = useRef<string>("");

  // --- Data params: changes require server refetch ---
  const vw = viewportSize?.vw ?? 0;
  const vh = viewportSize?.vh ?? 0;
  const dataParams: Record<string, string | number> = {
    background: vis.background,
    ...(vw > 0 ? { vw } : {}),
    ...(vh > 0 ? { vh } : {}),
    ...(referenceFrame > 0 ? { ref_frame: referenceFrame } : {}),
    _v: resultVersion,
  };
  const dataKey = `${displayComponent}|${resultVersion}|${JSON.stringify(dataParams)}`;

  // --- Color params: changes only need client-side re-colorization ---
  const vmin = displayComponent === "v" ? vis.vminV : vis.vminU;
  const vmax = displayComponent === "v" ? vis.vmaxV : vis.vmaxU;
  const colorKey = `${vis.colormap}|${vis.fixedRange}|${vmin}|${vmax}|${vis.logScale}`;

  // Pre-warm colormap LUT
  useEffect(() => {
    prefetchColormapLUT(vis.colormap);
  }, [vis.colormap]);

  const revokeRendered = useCallback(() => {
    for (const url of renderCacheRef.current.values()) {
      URL.revokeObjectURL(url);
    }
    renderCacheRef.current.clear();
  }, []);

  const invalidateAll = useCallback(() => {
    revokeRendered();
    dataCacheRef.current.clear();
    colorizingRef.current.clear();
    setCachedCount(0);
    setProgress(0);
    dataKeyRef.current = "";
    colorKeyRef.current = "";
  }, [revokeRendered]);

  const cancelPreRender = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsPreRendering(false);
  }, []);

  /**
   * Colorize a single data texture with current color settings.
   *
   * Delegates pixel-level work to a Web Worker pool so the main
   * thread stays responsive during batch pre-rendering.
   */
  const colorizeFrame = useCallback(
    async (dt: DataTexture): Promise<string | null> => {
      try {
        const lut = await getColormapLUT(vis.colormap);
        const effectiveVmin =
          vis.fixedRange && vmin ? parseFloat(vmin) : dt.dataMin;
        const effectiveVmax =
          vis.fixedRange && vmax ? parseFloat(vmax) : dt.dataMax;
        return await colorizeInWorker(
          dt,
          lut,
          effectiveVmin,
          effectiveVmax,
          vis.logScale,
        );
      } catch {
        return null;
      }
    },
    [vis.colormap, vis.fixedRange, vmin, vmax, vis.logScale],
  );

  /**
   * Re-colorize all cached data textures with current color settings.
   * This runs entirely client-side — no server fetches.
   *
   * Atomically replaces the render cache to avoid race conditions
   * with getFrame() lazy-colorize calls.
   */
  const recolorizeAll = useCallback(async () => {
    // Atomically swap out old cache — getFrame() sees empty cache during rebuild
    const oldCache = renderCacheRef.current;
    renderCacheRef.current = new Map();
    colorizingRef.current.clear();
    for (const url of oldCache.values()) URL.revokeObjectURL(url);

    const entries = Array.from(dataCacheRef.current.entries());
    let count = 0;
    for (const [idx, dt] of entries) {
      const blobUrl = await colorizeFrame(dt);
      if (blobUrl) {
        renderCacheRef.current.set(idx, blobUrl);
        count++;
      }
    }
    setCachedCount(count);
    setProgress(entries.length > 0 ? 100 : 0);
    colorKeyRef.current = colorKey;
  }, [colorizeFrame, colorKey]);

  const getFrame = useCallback(
    (idx: number): string | undefined => {
      // Check Tier 2 (rendered)
      const rendered = renderCacheRef.current.get(idx);
      if (rendered) return rendered;

      // Check Tier 1 (data texture) — lazy colorize with deduplication
      const dt = dataCacheRef.current.get(idx);
      if (dt && !colorizingRef.current.has(idx)) {
        colorizingRef.current.add(idx);
        colorizeFrame(dt).then((url) => {
          colorizingRef.current.delete(idx);
          if (url) {
            renderCacheRef.current.set(idx, url);
            setCachedCount(renderCacheRef.current.size);
          }
        });
      }

      return undefined;
    },
    [colorizeFrame],
  );

  const startPreRender = useCallback(() => {
    if (!hasResults || numFrames === 0) return;

    const isStrain = STRAIN_COMPONENTS.has(displayComponent);
    const currentDataKey = dataKey;
    const currentColorKey = colorKey;

    // If data cache is complete and color key matches, nothing to do
    if (
      dataKeyRef.current === currentDataKey &&
      colorKeyRef.current === currentColorKey &&
      renderCacheRef.current.size >= numFrames
    ) {
      return;
    }

    // If only color params changed but data cache is complete, just re-colorize
    if (
      dataKeyRef.current === currentDataKey &&
      dataCacheRef.current.size >= numFrames &&
      colorKeyRef.current !== currentColorKey
    ) {
      recolorizeAll();
      return;
    }

    // Data params changed — need to refetch from server
    if (dataKeyRef.current !== currentDataKey) {
      invalidateAll();
    }
    dataKeyRef.current = currentDataKey;
    colorKeyRef.current = currentColorKey;

    // Abort any in-flight pre-render
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsPreRendering(true);
    setProgress(0);

    const CONCURRENCY = 3;
    let nextIdx = 0;

    const fetchFrame = async () => {
      while (nextIdx < numFrames) {
        if (controller.signal.aborted) return;
        const idx = nextIdx++;

        // Skip already-cached
        if (dataCacheRef.current.has(idx)) continue;

        const url = buildDataTextureUrl(
          idx,
          displayComponent,
          dataParams,
          isStrain,
        );
        try {
          const resp = await fetch(url, { signal: controller.signal });
          if (!resp.ok) continue;

          const dataMin = parseFloat(
            resp.headers.get("X-Data-Min") ?? "0",
          );
          const dataMax = parseFloat(
            resp.headers.get("X-Data-Max") ?? "1",
          );
          const blob = await resp.blob();
          if (controller.signal.aborted) return;

          const dt = await decodeDataTexture(blob, dataMin, dataMax);
          dataCacheRef.current.set(idx, dt);

          // Immediately colorize for display
          const blobUrl = await colorizeFrame(dt);
          if (blobUrl && !controller.signal.aborted) {
            renderCacheRef.current.set(idx, blobUrl);
          }

          const count = renderCacheRef.current.size;
          setCachedCount(count);
          setProgress(Math.round((count / numFrames) * 100));
        } catch {
          if (controller.signal.aborted) return;
        }
      }
    };

    const workers = Array.from(
      { length: Math.min(CONCURRENCY, numFrames) },
      fetchFrame,
    );
    Promise.all(workers).then(() => {
      if (!controller.signal.aborted) {
        setIsPreRendering(false);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    hasResults,
    numFrames,
    displayComponent,
    dataKey,
    colorKey,
    invalidateAll,
    recolorizeAll,
    colorizeFrame,
  ]);

  // When DATA params change → full refetch
  useEffect(() => {
    invalidateAll();
    cancelPreRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    displayComponent,
    vis.background,
    referenceFrame,
    resultVersion,
    vw,
    vh,
  ]);

  // When COLOR params change → re-colorize from data cache (instant)
  useEffect(() => {
    if (dataCacheRef.current.size > 0 && colorKeyRef.current !== colorKey) {
      recolorizeAll();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vis.colormap, vis.fixedRange, vmin, vmax, vis.logScale]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelPreRender();
      revokeRendered();
      terminatePool();
    };
  }, [cancelPreRender, revokeRendered]);

  return {
    getFrame,
    isPreRendering,
    progress,
    cachedCount,
    totalFrames: numFrames,
    startPreRender,
    cancelPreRender,
    invalidate: invalidateAll,
  };
}
