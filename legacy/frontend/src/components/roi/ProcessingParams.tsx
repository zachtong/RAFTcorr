import { useState, useCallback, useEffect, useRef } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { SegmentedControl } from "@/components/shared/SegmentedControl";
import { SelectField } from "@/components/shared/SelectField";
import { Toggle } from "@/components/shared/Toggle";
import { SmallInput } from "@/components/shared/SmallInput";
import { useAppStore } from "@/stores/appStore";
import { configureProcessing, fetchTilingPreview } from "@/api/processing";
import { KeyFrameTimeline } from "./KeyFrameTimeline";
import type { TilingPreview } from "@/types/api";
import { ParamTooltip } from "@/components/shared/ParamTooltip";
import {
  memoryBudgetToPixels,
  pixelsToMemoryBudget,
  blendQualityToParams,
  formatTileSize,
  formatMemory,
} from "@/utils/tilingCalculations";

const keyFrameModeOptions = [
  { value: "every_frame", label: "Every Frame" },
  { value: "every_n", label: "Every N" },
  { value: "custom", label: "Custom" },
];


// Tooltip content for computed parameters (uses PARAM_DESCRIPTIONS via ParamTooltip)
import { PARAM_DESCRIPTIONS } from "@/lib/paramDescriptions";

export function ProcessingParams() {
  // Existing store selectors
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const totalFrames = useAppStore((s) => s.imageFiles.length);
  const keyFrames = useAppStore((s) => s.keyFrames);
  const keyFrameMode = useAppStore((s) => s.keyFrameMode);
  const keyFrameInterval = useAppStore((s) => s.keyFrameInterval);
  const setKeyFrames = useAppStore((s) => s.setKeyFrames);
  const setKeyFrameMode = useAppStore((s) => s.setKeyFrameMode);
  const setKeyFrameInterval = useAppStore((s) => s.setKeyFrameInterval);
  const useMedianFilter = useAppStore((s) => s.useMedianFilter);
  const setUseMedianFilter = useAppStore((s) => s.setUseMedianFilter);

  // New tiling-related selectors
  const modelMetadata = useAppStore((s) => s.modelMetadata);
  const roiConfirmed = useAppStore((s) => s.roiConfirmed);
  const roiVersion = useAppStore((s) => s.roiVersion);
  const tilingPreview = useAppStore((s) => s.tilingPreview);
  const showTileGrid = useAppStore((s) => s.showTileGrid);
  const setTilingPreview = useAppStore((s) => s.setTilingPreview);
  const setShowTileGrid = useAppStore((s) => s.setShowTileGrid);
  const processingActive = useAppStore((s) => s.processingActive);

  // Maximum p_max_pixels from model (or default)
  const maxPmax = modelMetadata?.safe_pmax ?? 4_000_000;
  // Recommended position (from safe_pmax)
  const recommendedPmax = maxPmax;
  // Allow going up to 2x recommended (with warning)
  const sliderMaxPmax = Math.min(maxPmax * 2, 16_000_000);

  // Slider states (0-1)
  const [memoryBudget, setMemoryBudget] = useState(() =>
    pixelsToMemoryBudget(maxPmax, sliderMaxPmax)
  );
  const [blendQuality, setBlendQuality] = useState(0.5); // Balanced default

  // Override states: null means auto-computed, number means user override
  const [overrides, setOverrides] = useState<{
    tileOverlap: number | null;
    roiMargin: number | null;
    smoothingRadius: number | null;
  }>({
    tileOverlap: null,
    roiMargin: null,
    smoothingRadius: null,
  });

  // Editing states for click-to-edit
  const [editing, setEditing] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  // Local state for interval input (synced on blur / Enter)
  const [intervalInput, setIntervalInput] = useState(String(keyFrameInterval));

  // Computed values
  const pMaxPixels = memoryBudgetToPixels(memoryBudget, sliderMaxPmax);
  const blendParams = blendQualityToParams(blendQuality);

  const effectiveTileOverlap = overrides.tileOverlap ?? blendParams.overlap;
  const effectiveRoiMargin = overrides.roiMargin ?? 64;
  const effectiveSmoothingRadius = overrides.smoothingRadius ?? blendParams.sigma;

  // Sync to backend when computed values change
  const syncRef = useRef(false);
  useEffect(() => {
    if (!syncRef.current) {
      syncRef.current = true;
      return;
    }
    configureProcessing({
      p_max_pixels: pMaxPixels,
      tile_overlap: effectiveTileOverlap,
      context_padding: effectiveRoiMargin,
      sigma: effectiveSmoothingRadius,
      use_smooth: effectiveSmoothingRadius > 0,
    }).catch(() => {});
  }, [pMaxPixels, effectiveTileOverlap, effectiveRoiMargin, effectiveSmoothingRadius]);

  // Fetch tiling preview when relevant state changes
  useEffect(() => {
    if (!roiConfirmed || !modelMetadata) {
      setTilingPreview(null);
      return;
    }
    const timer = setTimeout(() => {
      fetchTilingPreview()
        .then(setTilingPreview)
        .catch((err) => {
          console.error("[TilingPreview] fetch failed:", err?.response?.status, err?.message);
          setTilingPreview(null);
        });
    }, 300); // debounce 300ms
    return () => clearTimeout(timer);
  }, [roiConfirmed, roiVersion, modelMetadata, pMaxPixels, effectiveTileOverlap, effectiveRoiMargin, setTilingPreview]);

  // Update memory budget slider when model changes
  useEffect(() => {
    if (modelMetadata?.safe_pmax) {
      const newBudget = pixelsToMemoryBudget(modelMetadata.safe_pmax, sliderMaxPmax);
      setMemoryBudget(newBudget);
    }
  }, [modelMetadata?.safe_pmax, sliderMaxPmax]);

  // --- Mode handler ---
  const handleModeChange = (val: string) => {
    const m = val.toLowerCase() as "accumulative" | "incremental";
    setMode(m);
    configureProcessing({ mode: m }).catch(() => {});
  };

  // --- Incremental settings handlers ---

  const syncKeyFrameConfig = useCallback(
    (
      newMode: "every_frame" | "every_n" | "custom",
      interval?: number,
      frames?: number[]
    ) => {
      if (newMode === "every_frame") {
        configureProcessing({
          key_frames: null,
          key_frame_interval: 1,
        }).catch(() => {});
      } else if (newMode === "every_n") {
        configureProcessing({
          key_frame_interval: interval ?? keyFrameInterval,
          key_frames: null,
        }).catch(() => {});
      } else {
        // custom
        configureProcessing({
          key_frames: frames ?? keyFrames,
          key_frame_interval: null,
        }).catch(() => {});
      }
    },
    [keyFrameInterval, keyFrames]
  );

  const handleKeyFrameModeChange = (val: string) => {
    const m = val as "every_frame" | "every_n" | "custom";
    setKeyFrameMode(m);
    syncKeyFrameConfig(m);
  };

  const handleIntervalCommit = (val: string) => {
    const n = parseInt(val, 10);
    if (!isNaN(n) && n >= 1) {
      setKeyFrameInterval(n);
      syncKeyFrameConfig("every_n", n);
    }
  };

  const handleKeyFramesChange = (kf: number[]) => {
    setKeyFrames(kf);
    syncKeyFrameConfig("custom", undefined, kf);
  };

  const handleMedianFilterChange = (checked: boolean) => {
    setUseMedianFilter(checked);
    configureProcessing({ use_median_filter: checked }).catch(() => {});
  };

  // Click-to-edit handlers
  const startEdit = (key: string, currentValue: number) => {
    setEditing(key);
    setEditValue(String(currentValue));
  };

  const commitEdit = (key: string) => {
    const num = parseFloat(editValue);
    if (!isNaN(num) && num >= 0) {
      setOverrides((prev) => ({ ...prev, [key]: num }));
    }
    setEditing(null);
  };

  const resetOverride = (key: string) => {
    setOverrides((prev) => ({ ...prev, [key]: null }));
  };

  // Determine if memory budget exceeds recommendation
  const isOverBudget = pMaxPixels > recommendedPmax;
  const recommendedSliderPos = pixelsToMemoryBudget(recommendedPmax, sliderMaxPmax);

  // Helper to render a computed parameter row
  function renderComputedParam(key: string, label: string, value: number, displayValue: string) {
    const isOverridden = overrides[key as keyof typeof overrides] !== null;
    const isEditingThis = editing === key;

    return (
      <div key={key} className="flex items-center justify-between gap-2 group">
        <div className="flex items-center gap-1">
          <span
            className={`text-[11px] ${isOverridden ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"}`}
          >
            {label}
          </span>
          <span className="opacity-0 group-hover:opacity-100">
            <ParamTooltip paramKey={key} />
          </span>
        </div>
        <div className="flex items-center gap-1">
          {isEditingThis ? (
            <input
              type="text"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onBlur={() => commitEdit(key)}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitEdit(key);
                if (e.key === "Escape") setEditing(null);
              }}
              className="w-16 h-5 px-1 text-[11px] bg-[var(--input)] border border-[var(--primary)] rounded text-right text-[var(--foreground)] outline-none"
              autoFocus
            />
          ) : (
            <span
              className={`text-[11px] cursor-pointer hover:underline ${
                isOverridden ? "text-[var(--foreground)] font-medium" : "text-[var(--muted-foreground)]"
              }`}
              onClick={() => !processingActive && startEdit(key, value)}
            >
              {displayValue}
            </span>
          )}
          {isOverridden && !isEditingThis && (
            <button
              onClick={() => resetOverride(key)}
              className="text-[10px] text-[var(--muted-foreground)] hover:text-[var(--foreground)] opacity-60 hover:opacity-100"
              title="Reset to auto"
            >
              &#8634;
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <>
      <CollapsibleSection title="Processing Mode">
        {/* Mode selector */}
        <SegmentedControl
          options={["Accumulative", "Incremental"]}
          value={mode === "accumulative" ? "Accumulative" : "Incremental"}
          onChange={handleModeChange}
        />

        {/* Tiling controls — only shown after ROI is confirmed */}
        {roiConfirmed && (
          <>
            {/* Memory Budget Slider */}
            <div className="mt-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-[var(--muted-foreground)] flex items-center gap-1">Memory Budget <ParamTooltip paramKey="memoryBudget" /></span>
                <span className="text-[11px] text-[var(--muted-foreground)]">
                  {formatTileSize(pMaxPixels)}
                </span>
              </div>
              <div className="relative">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={memoryBudget}
                  onChange={(e) => setMemoryBudget(parseFloat(e.target.value))}
                  className="w-full h-1 rounded appearance-none cursor-pointer"
                  style={{
                    background: isOverBudget
                      ? `linear-gradient(to right, var(--primary) 0%, var(--primary) ${recommendedSliderPos * 100}%, #f59e0b ${recommendedSliderPos * 100}%, #f59e0b 100%)`
                      : "var(--secondary)",
                    accentColor: isOverBudget ? "#f59e0b" : "var(--primary)",
                  }}
                  disabled={processingActive}
                />
                {isOverBudget && (
                  <span className="text-[9px] text-yellow-400 mt-0.5 block">
                    Exceeds recommended — may cause OOM
                  </span>
                )}
              </div>
            </div>

            {/* Blend Quality Slider */}
            <div className="mt-2 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-[var(--muted-foreground)] flex items-center gap-1">Blend Quality <ParamTooltip paramKey="blendQuality" /></span>
                <span className="text-[11px] text-[var(--muted-foreground)]">
                  {blendQuality < 0.25 ? "Fast" : blendQuality < 0.75 ? "Balanced" : "High"}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={blendQuality}
                onChange={(e) => setBlendQuality(parseFloat(e.target.value))}
                className="w-full h-1 bg-[var(--secondary)] rounded appearance-none cursor-pointer accent-[var(--primary)]"
                disabled={processingActive}
              />
              <div className="flex justify-between text-[9px] text-[var(--muted-foreground)] opacity-60">
                <span>Fast</span>
                <span>Balanced</span>
                <span>High</span>
              </div>
            </div>

            {/* Computed Parameters */}
            <div className="mt-3 space-y-1 border-t border-[var(--border)] pt-2">
              <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider">
                Parameters
              </span>
              {renderComputedParam("tileOverlap", "Tile Overlap", effectiveTileOverlap, `${effectiveTileOverlap} px`)}
              {renderComputedParam("roiMargin", "ROI Margin", effectiveRoiMargin, `${effectiveRoiMargin} px`)}
              {renderComputedParam("smoothingRadius", "Smoothing Radius", effectiveSmoothingRadius, `${effectiveSmoothingRadius}`)}
            </div>

            {/* Show Tile Grid toggle */}
            <FieldRow label="Show Tile Grid">
              <Toggle checked={showTileGrid} onChange={setShowTileGrid} />
            </FieldRow>

            {/* Tiling Status Panel */}
            {tilingPreview && (
              <TilingStatusPanel preview={tilingPreview} />
            )}
          </>
        )}
      </CollapsibleSection>

      {mode === "incremental" && (
        <CollapsibleSection title="Incremental Settings" defaultOpen>
          <FieldRow label={<span className="flex items-center gap-1">Ref. Update <ParamTooltip paramKey="keyFrameMode" /></span>}>
            <SelectField
              value={keyFrameMode}
              options={keyFrameModeOptions}
              onChange={handleKeyFrameModeChange}
            />
          </FieldRow>

          {keyFrameMode === "every_n" && (
            <FieldRow label="Interval">
              <SmallInput
                value={intervalInput}
                onChange={(v) => setIntervalInput(v)}
                className="w-14"
              />
              <button
                onClick={() => handleIntervalCommit(intervalInput)}
                className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)]"
              >
                Set
              </button>
            </FieldRow>
          )}

          {keyFrameMode === "custom" && (
            <KeyFrameTimeline
              totalFrames={totalFrames}
              keyFrames={keyFrames}
              onChange={handleKeyFramesChange}
            />
          )}

          <FieldRow label={<span className="flex items-center gap-1">Median Filter <ParamTooltip paramKey="useMedianFilter" /></span>}>
            <Toggle
              checked={useMedianFilter}
              onChange={handleMedianFilterChange}
            />
          </FieldRow>
          {useMedianFilter && (
            <p className="text-[10px] text-yellow-400/80 leading-tight px-1 -mt-0.5">
              Applies median filter to accumulated displacement.
              Reduces error buildup but modifies raw output data.
            </p>
          )}
        </CollapsibleSection>
      )}
    </>
  );
}

// Tiling Status Panel sub-component
function TilingStatusPanel({ preview }: { preview: TilingPreview }) {
  return (
    <div className="mt-2 space-y-0.5 border-t border-[var(--border)] pt-2">
      <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider">
        Status
      </span>
      <StatusRow label="Image" value={`${preview.image_size[0]} \u00d7 ${preview.image_size[1]} px`} />
      <StatusRow label="ROI Area" value={`${preview.roi_bbox[2]} \u00d7 ${preview.roi_bbox[3]} px`} />
      <StatusRow label="Work Area" value={`${preview.work_area.w} \u00d7 ${preview.work_area.h} px`} />
      <StatusRow
        label="Tiling"
        value={preview.is_single_shot
          ? "Single shot"
          : `${preview.tile_count} tiles`}
      />
      {!preview.is_single_shot && (
        <StatusRow label="Overlap" value={`${(preview.overlap_ratio * 100).toFixed(1)}%`} />
      )}
      <div className="border-t border-[var(--border)] my-1" />
      <StatusRow label="GPU" value={preview.gpu.name} />
      <StatusRow
        label="VRAM Free"
        value={`${formatMemory(preview.gpu.free_mb)} / ${formatMemory(preview.gpu.total_mb)}`}
      />
      <StatusRow
        label="Est. Memory"
        value={`~${formatMemory(preview.est_memory_per_tile_mb)} / tile`}
      />
      <StatusRow
        label="Est. Time"
        value={`~${preview.est_time_seconds.toFixed(1)}s`}
      />
    </div>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] text-[var(--muted-foreground)]">{label}</span>
      <span className="text-[10px] text-[var(--foreground)]">{value}</span>
    </div>
  );
}
