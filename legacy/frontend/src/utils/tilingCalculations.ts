/**
 * Pure tile grid computation — mirrors raft_dic_gui/processing.py logic.
 * Used for client-side tile grid preview overlay on ROI canvas.
 */

export interface TileRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WorkArea {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface TilingResult {
  workArea: WorkArea;
  tiles: TileRect[];
  tileSize: number;
  isSingleShot: boolean;
  overlapRatio: number;
  effectiveOverlap: number;
}

/**
 * Generate start positions for sliding window with fixed overlap.
 * Mirrors _generate_tile_starts in processing.py.
 */
export function generateTileStarts(totalLen: number, tileLen: number, overlap: number): number[] {
  if (totalLen <= tileLen) return [0];

  const stride = tileLen - overlap;
  if (stride <= 0) {
    throw new Error(
      `Tile overlap (${overlap}) must be smaller than tile size (${tileLen}). ` +
      `Reduce overlap or increase memory budget.`
    );
  }

  const starts: number[] = [];
  let pos = 0;
  while (pos + tileLen < totalLen) {
    starts.push(pos);
    pos += stride;
  }

  // Ensure last tile covers the end
  const lastStart = starts[starts.length - 1];
  if (lastStart + tileLen < totalLen) {
    starts.push(totalLen - tileLen);
  }

  // Deduplicate and sort
  return [...new Set(starts)].sort((a, b) => a - b);
}

/**
 * Compute the work area (ROI bbox + context padding, clamped to image bounds).
 */
export function computeWorkArea(
  roiBbox: { x: number; y: number; w: number; h: number },
  contextPadding: number,
  imageWidth: number,
  imageHeight: number,
): WorkArea {
  const xC = Math.max(0, roiBbox.x - contextPadding);
  const yC = Math.max(0, roiBbox.y - contextPadding);
  const xR = Math.min(imageWidth, roiBbox.x + roiBbox.w + contextPadding);
  const yB = Math.min(imageHeight, roiBbox.y + roiBbox.h + contextPadding);
  return { x: xC, y: yC, w: xR - xC, h: yB - yC };
}

/**
 * Compute the full tiling grid from parameters.
 * This is the main function used by the UI for real-time preview.
 */
export function computeTilingGrid(
  roiBbox: { x: number; y: number; w: number; h: number },
  contextPadding: number,
  imageWidth: number,
  imageHeight: number,
  pMaxPixels: number,
  tileOverlap: number,
): TilingResult {
  const workArea = computeWorkArea(roiBbox, contextPadding, imageWidth, imageHeight);
  const { w: wC, h: hC } = workArea;

  const maxT = Math.floor(Math.sqrt(pMaxPixels));

  if (wC * hC <= pMaxPixels) {
    // Single shot — no tiling needed
    return {
      workArea,
      tiles: [{ x: 0, y: 0, w: wC, h: hC }],
      tileSize: Math.max(wC, hC),
      isSingleShot: true,
      overlapRatio: 0,
      effectiveOverlap: 0,
    };
  }

  // Tiling needed
  const T = maxT;
  let effectiveOverlap = tileOverlap;

  // If tile is too small for meaningful overlap, reduce overlap
  if (T < effectiveOverlap * 2 + 16) {
    effectiveOverlap = Math.max(4, Math.floor(T / 4));
  }

  const startsX = generateTileStarts(wC, T, effectiveOverlap);
  const startsY = generateTileStarts(hC, T, effectiveOverlap);

  const tiles: TileRect[] = [];
  for (const sy of startsY) {
    for (const sx of startsX) {
      tiles.push({
        x: sx,
        y: sy,
        w: Math.min(T, wC - sx),
        h: Math.min(T, hC - sy),
      });
    }
  }

  const overlapRatio = T > 0 ? effectiveOverlap / T : 0;

  return {
    workArea,
    tiles,
    tileSize: T,
    isSingleShot: false,
    overlapRatio,
    effectiveOverlap,
  };
}

/**
 * Compute default parameters from slider positions.
 */

/** Memory Budget slider → p_max_pixels */
export function memoryBudgetToPixels(sliderValue: number, maxPmax: number): number {
  // Slider is 0..1, exponential scale from 4096 to maxPmax
  const minPixels = 4096;
  const logMin = Math.log(minPixels);
  const logMax = Math.log(maxPmax);
  return Math.round(Math.exp(logMin + sliderValue * (logMax - logMin)));
}

export function pixelsToMemoryBudget(pMaxPixels: number, maxPmax: number): number {
  const minPixels = 4096;
  const logMin = Math.log(minPixels);
  const logMax = Math.log(maxPmax);
  return (Math.log(pMaxPixels) - logMin) / (logMax - logMin);
}

/** Blend Quality slider → (overlap, sigma) */
export function blendQualityToParams(sliderValue: number): { overlap: number; sigma: number } {
  // Slider 0..1: Fast(0) → Balanced(0.5) → High(1.0)
  // overlap: 16 → 64 → 128
  // sigma: 1.0 → 1.5 → 2.0
  const overlap = Math.round(16 + sliderValue * (128 - 16));
  const sigma = +(1.0 + sliderValue * (2.0 - 1.0)).toFixed(1);
  return { overlap, sigma };
}

export function paramsToBlendQuality(overlap: number): number {
  return Math.max(0, Math.min(1, (overlap - 16) / (128 - 16)));
}

/**
 * Format tile size for display.
 */
export function formatTileSize(pMaxPixels: number): string {
  const side = Math.floor(Math.sqrt(pMaxPixels));
  return `${side} × ${side}`;
}

/**
 * Format bytes to human-readable.
 */
export function formatMemory(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}
