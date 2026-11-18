import { SmallInput } from "@/components/shared/SmallInput";
import { applyFrame1Mask } from "@/api/processing";
import { useAppStore } from "@/stores/appStore";
import { useRoiStore } from "@/stores/roiStore";

interface MaskSourceSelectorProps {
  maskSource: "auto" | "folder";
  maskDir: string;
  onSourceChange: (source: "auto" | "folder") => void;
  onDirChange: (dir: string) => void;
  validationResult?: {
    matched_count: number;
    total_frames: number;
    masks_needed: number;
    matched_frames: number[];
    has_frame_1: boolean;
  } | null;
  onValidate: () => void;
  keyFrames: number[];
  keyFrameMode: "every_frame" | "every_n" | "custom";
}

export function MaskSourceSelector({
  maskDir,
  onDirChange,
  validationResult,
  onValidate,
  keyFrames,
  keyFrameMode,
}: MaskSourceSelectorProps) {
  // Compute unmatched key frames when in custom mode
  const unmatchedKeyFrames =
    validationResult && keyFrameMode === "custom" && keyFrames.length > 0
      ? keyFrames.filter((kf) => !validationResult.matched_frames.includes(kf))
      : [];

  const handleApplyFrame1 = async () => {
    if (!maskDir) return;
    try {
      await applyFrame1Mask(maskDir);
      useRoiStore.getState().setMaskUrl(`/api/roi/mask?t=${Date.now()}`);
      useAppStore.getState().setRoiConfirmed(true);
      useAppStore.getState().bumpRoiVersion();
    } catch (err: any) {
      console.error("Apply frame-1 mask failed:", err);
    }
  };

  return (
    <div className="flex flex-col gap-1.5 w-full">
      {/* Folder path input + Validate */}
      <div className="flex items-center gap-1">
        <SmallInput
          value={maskDir}
          onChange={onDirChange}
          className="flex-1 min-w-0 text-left"
          placeholder="Mask folder path"
        />
        <button
          onClick={onValidate}
          disabled={!maskDir}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)] disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
        >
          Validate
        </button>
      </div>

      {/* Validation result */}
      {validationResult && (
        <>
          <span
            className={`text-[10px] ${
              validationResult.matched_count >= validationResult.masks_needed
                ? "text-green-400"
                : validationResult.matched_count > 0
                ? "text-yellow-400"
                : "text-red-400"
            }`}
          >
            Loaded {validationResult.matched_count} masks ({validationResult.masks_needed} needed for {validationResult.total_frames} images)
          </span>

          {/* Frame 1 status */}
          {validationResult.has_frame_1 ? (
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-green-400">
                Frame 1 mask found
              </span>
              <button
                onClick={handleApplyFrame1}
                className="h-5 px-1.5 text-[10px] bg-blue-600/30 border border-blue-500/40 rounded hover:bg-blue-600/50 text-blue-300"
              >
                Apply as ROI
              </button>
            </div>
          ) : (
            <span className="text-[10px] text-yellow-400/80">
              No Frame 1 mask — draw ROI manually
            </span>
          )}

          {/* Unmatched key frames warning */}
          {unmatchedKeyFrames.length > 0 && (
            <p className="text-[10px] text-yellow-400/80 leading-tight">
              Key frames without masks:{" "}
              {unmatchedKeyFrames.join(", ")}
              {" "}— auto-warp will be used for these frames
            </p>
          )}
        </>
      )}
    </div>
  );
}
