/* ─── Backend API types ─── */

export interface DICConfig {
  imgDir: string;
  modelPath: string;
  mode: "accumulative" | "incremental";
  contextPadding: number;
  tileOverlap: number;
  useSmooth: boolean;
  sigma: number;
  pMaxPixels: number;
  device: string;
  projectRoot: string;
}

export interface ModelEntry {
  path: string;
  label: string;
}

export interface ModelMetadata {
  path: string;
  label: string;
  small: boolean;
  mixed_precision: boolean;
  alternate_corr: boolean;
  corr_levels: number | null;
  corr_radius: number | null;
  variant: string;
  full_resolution: boolean;
  summary: string;
  safe_pmax: number;
}

export interface ImageLoadResult {
  files: string[];
  count: number;
  width: number;
  height: number;
  sort_suggestion?: "natural";
}

export interface RoiResult {
  rect: [number, number, number, number] | null;
  area_px: number;
}

export interface ProcessingProgress {
  percent: number;
  current: number;
  total: number;
  tile_current?: number;
  tile_total?: number;
  vram_used_mb?: number;
  vram_total_mb?: number;
  elapsed_seconds?: number;
  est_remaining_seconds?: number;
}

export interface DisplacementInfo {
  num_frames: number;
  roi_rect: [number, number, number, number];
  roi_shape: [number, number];
}

export interface FrameData {
  data: number[][];
  shape: [number, number];
  vmin: number;
  vmax: number;
  mean: number;
}

export interface StrainStatus {
  computed: boolean;
  computing: boolean;
  components: string[];
  num_frames: number;
}

export interface ProbeData {
  id: number;
  type: "point" | "line" | "area";
  coords: unknown;
  color: string;
  label: string;
}

export interface TimeSeriesResult {
  series: Record<string, (number | null)[]>;
}

export interface KymographResult {
  data: number[][];
  dist_axis: number[];
  shape: [number, number];
}

export interface ExportProgress {
  current: number;
  total: number;
  percent: number;
  message: string;
}

export interface TilingPreview {
  image_size: [number, number];
  roi_bbox: [number, number, number, number];
  work_area: { x: number; y: number; w: number; h: number };
  tile_size: number;
  tiles: [number, number, number, number][];
  tile_count: number;
  overlap_ratio: number;
  gpu: { name: string; total_mb: number; free_mb: number };
  est_memory_per_tile_mb: number;
  est_time_seconds: number;
  is_single_shot: boolean;
}

export type StrainComponent =
  | "exx" | "eyy" | "exy"
  | "e1" | "e2"
  | "max_shear" | "von_mises" | "rotation"
  | "dexx_dt" | "deyy_dt" | "dexy_dt";

export type DisplacementComponent = "u" | "v" | "magnitude" | "velocity";

export type DisplayComponent = DisplacementComponent | StrainComponent;
