import { create } from "zustand";
import type { ModelMetadata, ProbeData, DisplayComponent, TilingPreview } from "@/types/api";

interface ArrowSettings {
  showQuiver: boolean;
  showStreamlines: boolean;
  showPrincipalDirs: boolean;
  spacing: number;
  scale: number;
  color: string;
  lineWidth: number;
  streamQuality: number; // downsample factor: 1=full, 2=fine, 4=balanced, 8=fast
}

interface VisSettings {
  colormap: string;
  alpha: number;
  background: "reference" | "deformed";
  logScale: boolean;
  physicalEnabled: boolean;
  physicalRatio: number;
  physicalUnit: string;
  fps: number;
  fixedRange: boolean;
  vminU: string;
  vmaxU: string;
  vminV: string;
  vmaxV: string;
}

interface ColorbarSettings {
  labelText: string;        // "" = auto from component name
  labelFontSize: number;    // default 12
  tickCount: number;        // 0 = auto
  tickFontSize: number;     // default 10
  barThickness: number;     // 0.5-3.0, default 1.0 — controls bar width (fatness)
  hideOutline: boolean;     // default false
  discreteLevels: number;   // 0 = continuous, >0 = discrete color bins
}

interface AppState {
  // Image state
  imageDir: string;
  imageFiles: string[];
  imageWidth: number;
  imageHeight: number;

  // Model state
  selectedModel: string;
  modelMetadata: ModelMetadata | null;

  // Config
  mode: "accumulative" | "incremental";

  // ROI
  roiConfirmed: boolean;
  roiVersion: number;

  // Processing
  processingActive: boolean;
  processingPaused: boolean;
  processingProgress: number;
  processingCurrent: number;
  processingTotal: number;

  // Displacement
  hasResults: boolean;
  numFrames: number;
  currentFrame: number;
  referenceFrame: number;

  // Result version — monotonically increases on each recomputation (busts caches)
  resultVersion: number;

  // Provenance of the currently displayed results — used to detect staleness
  // when the user edits the ROI or changes the model after a run.
  lastRunRoiVersion: number;
  lastRunModel: string;
  lastRunAt: number | null;

  // Strain
  hasStrain: boolean;
  strainComputing: boolean;
  strainProgress: number;
  strainComponents: string[];

  // Display
  displayComponent: DisplayComponent;
  visSettings: VisSettings;
  arrowSettings: ArrowSettings;

  // Probes
  probes: ProbeData[];
  activeProbeTab: "point" | "line" | "area";
  probePlacingMode: "point" | "line" | "area-rect" | "area-circle" | "area-poly" | null;
  probePlacingFirst: [number, number] | null;
  areaPolyPoints: [number, number][];

  // Incremental settings
  keyFrames: number[];
  keyFrameMode: "every_frame" | "every_n" | "custom";
  keyFrameInterval: number;
  useMedianFilter: boolean;
  maskSource: "auto" | "folder";
  maskDir: string;
  maskValidation: { matched_count: number; total_frames: number; masks_needed: number; matched_frames: number[]; has_frame_1: boolean } | null;

  // Tiling state
  tilingPreview: TilingPreview | null;
  showTileGrid: boolean;

  // View zoom & pan (shared by displacement + postprocessing)
  viewZoom: number;
  viewOffset: { x: number; y: number };

  // Colorbar
  colorbarSettings: ColorbarSettings;

  // Export
  exportActive: boolean;
  exportProgress: number;

  // Actions
  resetSession: () => void;
  setImageDir: (dir: string) => void;
  setImageFiles: (files: string[], w: number, h: number) => void;
  setModel: (path: string, meta: ModelMetadata | null) => void;
  setMode: (mode: "accumulative" | "incremental") => void;
  setRoiConfirmed: (v: boolean) => void;
  bumpRoiVersion: () => void;
  setProcessing: (active: boolean, progress?: number, current?: number, total?: number) => void;
  setProcessingPaused: (paused: boolean) => void;
  setResults: (numFrames: number) => void;
  setCurrentFrame: (frame: number) => void;
  setReferenceFrame: (frame: number) => void;
  setStrain: (has: boolean, components?: string[]) => void;
  setStrainComputing: (v: boolean, progress?: number) => void;
  setDisplayComponent: (c: DisplayComponent) => void;
  updateVisSettings: (partial: Partial<VisSettings>) => void;
  updateArrowSettings: (partial: Partial<ArrowSettings>) => void;
  setProbes: (probes: ProbeData[]) => void;
  setActiveProbeTab: (tab: "point" | "line" | "area") => void;
  setProbePlacingMode: (mode: "point" | "line" | "area-rect" | "area-circle" | "area-poly" | null) => void;
  setProbePlacingFirst: (pt: [number, number] | null) => void;
  addAreaPolyPoint: (x: number, y: number) => void;
  clearAreaPolyPoints: () => void;
  setKeyFrames: (kf: number[]) => void;
  setKeyFrameMode: (mode: "every_frame" | "every_n" | "custom") => void;
  setKeyFrameInterval: (n: number) => void;
  setUseMedianFilter: (v: boolean) => void;
  setMaskSource: (source: "auto" | "folder") => void;
  setMaskDir: (dir: string) => void;
  setMaskValidation: (result: { matched_count: number; total_frames: number; masks_needed: number; matched_frames: number[]; has_frame_1: boolean } | null) => void;
  setTilingPreview: (preview: TilingPreview | null) => void;
  setShowTileGrid: (show: boolean) => void;
  updateColorbarSettings: (partial: Partial<ColorbarSettings>) => void;
  setExport: (active: boolean, progress?: number) => void;
  setViewOffset: (offset: { x: number; y: number }) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  zoomReset: () => void;
  restoreFromSession: (summary: {
    num_displacement_frames: number;
    num_strain_frames: number;
    strain_components: string[];
    has_roi: boolean;
    roi_confirmed: boolean;
    image_dir: string;
    image_dir_exists: boolean;
    image_width: number;
    image_height: number;
  }) => void;
}

export type { ColorbarSettings };

export const useAppStore = create<AppState>((set) => ({
  imageDir: "",
  imageFiles: [],
  imageWidth: 0,
  imageHeight: 0,
  selectedModel: "",
  modelMetadata: null,
  mode: "accumulative",
  roiConfirmed: false,
  roiVersion: 0,
  processingActive: false,
  processingPaused: false,
  processingProgress: 0,
  processingCurrent: 0,
  processingTotal: 0,
  hasResults: false,
  numFrames: 0,
  currentFrame: 0,
  referenceFrame: 0,
  resultVersion: 0,
  lastRunRoiVersion: 0,
  lastRunModel: "",
  lastRunAt: null,
  hasStrain: false,
  strainComputing: false,
  strainProgress: 0,
  strainComponents: [],
  displayComponent: "u",
  visSettings: {
    colormap: "turbo",
    alpha: 0.7,
    background: "reference",
    logScale: false,
    physicalEnabled: false,
    physicalRatio: 1.0,
    physicalUnit: "px",
    fps: 1.0,
    fixedRange: false,
    vminU: "",
    vmaxU: "",
    vminV: "",
    vmaxV: "",
  },
  arrowSettings: {
    showQuiver: false,
    showStreamlines: false,
    showPrincipalDirs: false,
    spacing: 30,
    scale: 20,
    color: "white",
    lineWidth: 1.5,
    streamQuality: 4,
  },
  probes: [],
  activeProbeTab: "point",
  probePlacingMode: null,
  probePlacingFirst: null,
  areaPolyPoints: [],
  keyFrames: [1],
  keyFrameMode: "every_frame",
  keyFrameInterval: 10,
  useMedianFilter: false,
  maskSource: "auto",
  maskDir: "",
  maskValidation: null,
  tilingPreview: null,
  showTileGrid: false,
  colorbarSettings: {
    labelText: "",
    labelFontSize: 12,
    tickCount: 0,
    tickFontSize: 10,
    barThickness: 1.0,
    hideOutline: false,
    discreteLevels: 0,
  },
  viewZoom: 1,
  viewOffset: { x: 0, y: 0 },
  exportActive: false,
  exportProgress: 0,

  resetSession: () =>
    set({
      imageWidth: 0,
      imageHeight: 0,
      roiConfirmed: false,
  roiVersion: 0,
      processingActive: false,
      processingPaused: false,
      processingProgress: 0,
      processingCurrent: 0,
      processingTotal: 0,
      hasResults: false,
      numFrames: 0,
      currentFrame: 0,
      referenceFrame: 0,
      lastRunRoiVersion: 0,
      lastRunModel: "",
      lastRunAt: null,
      hasStrain: false,
      strainComputing: false,
      strainProgress: 0,
      strainComponents: [],
      displayComponent: "u",
      probes: [],
      probePlacingMode: null,
      probePlacingFirst: null,
      areaPolyPoints: [],
      keyFrames: [1],
      keyFrameMode: "every_frame",
      keyFrameInterval: 10,
      useMedianFilter: false,
      maskSource: "auto",
      maskDir: "",
      maskValidation: null,
      tilingPreview: null,
      showTileGrid: false,
      viewZoom: 1,
      viewOffset: { x: 0, y: 0 },
      exportActive: false,
      exportProgress: 0,
    }),
  setImageDir: (dir) => set({ imageDir: dir }),
  setImageFiles: (files, w, h) => {
    set({ imageFiles: files, imageWidth: w, imageHeight: h });
  },
  setModel: (path, meta) => set({ selectedModel: path, modelMetadata: meta }),
  setMode: (mode) => set({ mode }),
  // setRoiConfirmed only toggles the "confirmed" gate for Run.  Version
  // bumping is handled by bumpRoiVersion, which every ROI mutation call
  // site triggers — otherwise per-frame edits would silently leave the
  // stale-result banner untriggered.
  setRoiConfirmed: (v) => set({ roiConfirmed: v }),
  bumpRoiVersion: () => set((s) => ({ roiVersion: s.roiVersion + 1 })),
  setProcessing: (active, progress = 0, current = 0, total = 0) =>
    set({
      processingActive: active,
      processingPaused: false,
      processingProgress: progress,
      processingCurrent: current,
      processingTotal: total,
    }),
  setProcessingPaused: (paused) => set({ processingPaused: paused }),
  setResults: (numFrames) =>
    set((s) => ({
      hasResults: numFrames > 0,
      numFrames,
      currentFrame: 0,
      resultVersion: s.resultVersion + 1,
      lastRunRoiVersion: s.roiVersion,
      lastRunModel: s.selectedModel,
      lastRunAt: Date.now(),
    })),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setReferenceFrame: (frame) => set({ referenceFrame: frame }),
  setStrain: (has, components = []) =>
    set((s) => ({ hasStrain: has, strainComponents: components, strainComputing: false, resultVersion: s.resultVersion + 1 })),
  setStrainComputing: (v, progress = 0) => set({ strainComputing: v, strainProgress: progress }),
  setDisplayComponent: (c) => set({ displayComponent: c }),
  updateVisSettings: (partial) =>
    set((s) => ({ visSettings: { ...s.visSettings, ...partial } })),
  updateArrowSettings: (partial) =>
    set((s) => ({ arrowSettings: { ...s.arrowSettings, ...partial } })),
  setProbes: (probes) => set({ probes }),
  setActiveProbeTab: (tab) => set({ activeProbeTab: tab }),
  setProbePlacingMode: (mode) => set({ probePlacingMode: mode, probePlacingFirst: null, areaPolyPoints: [] }),
  setProbePlacingFirst: (pt) => set({ probePlacingFirst: pt }),
  addAreaPolyPoint: (x, y) => set((s) => ({ areaPolyPoints: [...s.areaPolyPoints, [x, y]] })),
  clearAreaPolyPoints: () => set({ areaPolyPoints: [] }),
  setKeyFrames: (kf) => set({ keyFrames: kf }),
  setKeyFrameMode: (mode) => set({ keyFrameMode: mode }),
  setKeyFrameInterval: (n) => set({ keyFrameInterval: n }),
  setUseMedianFilter: (v) => set({ useMedianFilter: v }),
  setMaskSource: (source) => set({ maskSource: source }),
  setMaskDir: (dir) => set({ maskDir: dir }),
  setMaskValidation: (result) => set({ maskValidation: result }),
  setTilingPreview: (preview) => set({ tilingPreview: preview }),
  setShowTileGrid: (show) => set({ showTileGrid: show }),
  updateColorbarSettings: (partial) =>
    set((s) => ({ colorbarSettings: { ...s.colorbarSettings, ...partial } })),
  setExport: (active, progress = 0) =>
    set({ exportActive: active, exportProgress: progress }),
  setViewOffset: (offset) => set({ viewOffset: offset }),
  zoomIn: () => set((s) => ({ viewZoom: Math.min(s.viewZoom * 1.25, 5) })),
  zoomOut: () => set((s) => ({ viewZoom: Math.max(s.viewZoom / 1.25, 0.2) })),
  zoomReset: () => set({ viewZoom: 1, viewOffset: { x: 0, y: 0 } }),
  restoreFromSession: (summary) =>
    set((s) => ({
      imageDir: summary.image_dir,
      imageWidth: summary.image_width,
      imageHeight: summary.image_height,
      hasResults: summary.num_displacement_frames > 0,
      numFrames: summary.num_displacement_frames,
      currentFrame: 0,
      referenceFrame: 0,
      resultVersion: s.resultVersion + 1,
      roiConfirmed: summary.roi_confirmed,
      hasStrain: summary.num_strain_frames > 0,
      strainComponents: summary.strain_components,
      displayComponent: "u",
      // Treat a restored session as fresh: whatever ROI/model we have now
      // are the ones those results were computed from.
      lastRunRoiVersion: s.roiVersion,
      lastRunModel: s.selectedModel,
      lastRunAt: Date.now(),
    })),
}));
