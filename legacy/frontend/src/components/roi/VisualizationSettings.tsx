import { useEffect, useCallback, useState, useRef } from "react";
import { Settings } from "lucide-react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { ColorbarSettingsPanel } from "@/components/shared/ColorbarSettingsPanel";
import { FieldRow } from "@/components/shared/FieldRow";
import { Toggle } from "@/components/shared/Toggle";
import { SmallInput } from "@/components/shared/SmallInput";
import { SelectField } from "@/components/shared/SelectField";
import { SliderField } from "@/components/shared/SliderField";
import { ColormapBar } from "@/components/shared/ColormapBar";
import { useAppStore } from "@/stores/appStore";
import { getFrameData } from "@/api/displacement";

const COLORMAP_OPTIONS = [
  { value: "turbo", label: "Turbo" },
  { value: "viridis", label: "Viridis" },
  { value: "jet", label: "Jet" },
  { value: "coolwarm", label: "Coolwarm" },
  { value: "plasma", label: "Plasma" },
  { value: "inferno", label: "Inferno" },
];

const UNIT_OPTIONS = [
  { value: "px", label: "px" },
  { value: "mm", label: "mm" },
  { value: "µm", label: "µm" },
  { value: "m", label: "m" },
];

/** Input that edits locally and commits on blur/Enter */
function RangeInput({
  storeValue,
  onCommit,
  placeholder,
}: {
  storeValue: string;
  onCommit: (v: string) => void;
  placeholder: string;
}) {
  const [local, setLocal] = useState(storeValue);
  const [focused, setFocused] = useState(false);

  // Sync from store when not focused
  useEffect(() => {
    if (!focused) setLocal(storeValue);
  }, [storeValue, focused]);

  const commit = useCallback(() => {
    onCommit(local);
    setFocused(false);
  }, [local, onCommit]);

  return (
    <input
      type="text"
      value={local}
      placeholder={placeholder}
      onChange={(e) => setLocal(e.target.value)}
      onFocus={() => setFocused(true)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
      className="w-14 h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[11px] text-[var(--foreground)] text-center focus:border-[var(--primary)] focus:outline-none"
    />
  );
}

export function VisualizationSettings() {
  const vis = useAppStore((s) => s.visSettings);
  const update = useAppStore((s) => s.updateVisSettings);
  const hasResults = useAppStore((s) => s.hasResults);
  const currentFrame = useAppStore((s) => s.currentFrame);

  const [showCbPanel, setShowCbPanel] = useState(false);
  const cbBtnRef = useRef<HTMLButtonElement>(null);
  const physicalEnabled = vis.physicalEnabled;

  // Local string state for numeric fields that need decimal input
  const [ratioText, setRatioText] = useState(String(vis.physicalRatio));
  const [fpsText, setFpsText] = useState(String(vis.fps));

  // Auto-populate from current frame data when Fixed Range is toggled on
  const handleFixedRangeToggle = useCallback(
    async (enabled: boolean) => {
      update({ fixedRange: enabled });
      if (enabled && hasResults && !vis.vminU && !vis.vmaxU) {
        try {
          const [uData, vData] = await Promise.all([
            getFrameData(currentFrame, "u"),
            getFrameData(currentFrame, "v"),
          ]);
          update({
            vminU: String(uData.vmin.toFixed(3)),
            vmaxU: String(uData.vmax.toFixed(3)),
            vminV: String(vData.vmin.toFixed(3)),
            vmaxV: String(vData.vmax.toFixed(3)),
          });
        } catch {
          // No data yet — leave empty
        }
      }
    },
    [update, hasResults, currentFrame, vis.vminU, vis.vmaxU]
  );

  // Validate min < max and show warning
  const rangeWarning = (min: string, max: string): boolean => {
    const a = parseFloat(min);
    const b = parseFloat(max);
    return !isNaN(a) && !isNaN(b) && a >= b;
  };

  const uWarning = vis.fixedRange && rangeWarning(vis.vminU, vis.vmaxU);
  const vWarning = vis.fixedRange && rangeWarning(vis.vminV, vis.vmaxV);

  return (
    <>
      <CollapsibleSection title="Visualization">
        <FieldRow label="Fixed Range">
          <Toggle
            checked={vis.fixedRange}
            onChange={handleFixedRangeToggle}
          />
        </FieldRow>

        {vis.fixedRange && (
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-[var(--muted-foreground)] w-3">U</span>
              <RangeInput
                storeValue={vis.vminU}
                onCommit={(v) => update({ vminU: v })}
                placeholder="min"
              />
              <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
              <RangeInput
                storeValue={vis.vmaxU}
                onCommit={(v) => update({ vmaxU: v })}
                placeholder="max"
              />
            </div>
            {uWarning && (
              <p className="text-[9px] text-amber-400 pl-4">min must be less than max</p>
            )}
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-[var(--muted-foreground)] w-3">V</span>
              <RangeInput
                storeValue={vis.vminV}
                onCommit={(v) => update({ vminV: v })}
                placeholder="min"
              />
              <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
              <RangeInput
                storeValue={vis.vmaxV}
                onCommit={(v) => update({ vmaxV: v })}
                placeholder="max"
              />
            </div>
            {vWarning && (
              <p className="text-[9px] text-amber-400 pl-4">min must be less than max</p>
            )}
          </div>
        )}

        <FieldRow label="Colormap">
          <SelectField
            value={vis.colormap}
            options={COLORMAP_OPTIONS}
            onChange={(v) => update({ colormap: v })}
          />
        </FieldRow>
        <ColormapBar colormap={vis.colormap} />

        {/* Colorbar settings gear button */}
        <div>
          <button
            ref={cbBtnRef}
            onClick={() => setShowCbPanel((v) => !v)}
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px]
              bg-[var(--secondary)] hover:bg-[var(--secondary)]/80
              text-[var(--muted-foreground)] hover:text-[var(--foreground)]
              transition-colors w-full"
          >
            <Settings size={11} />
            Colorbar Settings
          </button>
          {showCbPanel && (
            <ColorbarSettingsPanel
              onClose={() => setShowCbPanel(false)}
              anchorRef={cbBtnRef}
            />
          )}
        </div>

        <FieldRow label="Opacity">
          <SliderField
            value={vis.alpha}
            onChange={(v) => update({ alpha: v })}
            min={0}
            max={1}
            step={0.05}
          />
        </FieldRow>
      </CollapsibleSection>

      <CollapsibleSection title="Physical Units">
        <FieldRow label="Enable">
          <Toggle
            checked={physicalEnabled}
            onChange={(v) => update({ physicalEnabled: v })}
          />
        </FieldRow>
        {physicalEnabled && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1 text-[10px] text-[var(--muted-foreground)]">
              <SmallInput
                value={ratioText}
                onChange={(v) => {
                  setRatioText(v);
                  const n = parseFloat(v);
                  if (!isNaN(n) && n > 0) update({ physicalRatio: n });
                }}
                onBlur={() => {
                  const n = parseFloat(ratioText);
                  setRatioText((!isNaN(n) && n > 0) ? String(n) : String(vis.physicalRatio));
                }}
                className="w-12"
              />
              <SelectField
                value={vis.physicalUnit}
                options={UNIT_OPTIONS}
                onChange={(v) => update({ physicalUnit: v })}
              />
              <span>= 1 px</span>
            </div>
            <FieldRow label="FPS">
              <SmallInput
                value={fpsText}
                onChange={(v) => {
                  setFpsText(v);
                  const n = parseFloat(v);
                  if (!isNaN(n) && n > 0) update({ fps: n });
                }}
                onBlur={() => {
                  const n = parseFloat(fpsText);
                  setFpsText((!isNaN(n) && n > 0) ? String(n) : String(vis.fps));
                }}
              />
            </FieldRow>
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection title="Background">
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="bg"
              checked={vis.background === "reference"}
              onChange={() => update({ background: "reference" })}
              className="accent-[var(--primary)] w-3 h-3"
            />
            <span className={`text-[11px] ${vis.background === "reference" ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"}`}>Reference Frame</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="bg"
              checked={vis.background === "deformed"}
              onChange={() => update({ background: "deformed" })}
              className="accent-[var(--primary)] w-3 h-3"
            />
            <span className={`text-[11px] ${vis.background === "deformed" ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"}`}>Deformed Frame</span>
          </label>
        </div>
      </CollapsibleSection>
    </>
  );
}
