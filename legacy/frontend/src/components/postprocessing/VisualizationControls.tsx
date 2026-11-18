import { useState, useRef } from "react";
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
import type { DisplayComponent } from "@/types/api";

const DISPLACEMENT_COMPONENTS: { value: DisplayComponent; label: string }[] = [
  { value: "u", label: "U (horizontal)" },
  { value: "v", label: "V (vertical)" },
  { value: "magnitude", label: "Magnitude" },
  { value: "velocity", label: "Velocity" },
];

const STRAIN_COMPONENTS: { value: DisplayComponent; label: string }[] = [
  { value: "exx", label: "εxx" },
  { value: "eyy", label: "εyy" },
  { value: "exy", label: "εxy" },
  { value: "e1", label: "ε1 (principal)" },
  { value: "e2", label: "ε2 (principal)" },
  { value: "max_shear", label: "Max Shear" },
  { value: "von_mises", label: "Von Mises" },
  { value: "rotation", label: "Rotation" },
  { value: "dexx_dt", label: "d\u03B5xx/dt" },
  { value: "deyy_dt", label: "d\u03B5yy/dt" },
  { value: "dexy_dt", label: "d\u03B5xy/dt" },
];

const UNIT_OPTIONS = [
  { value: "px", label: "px" },
  { value: "mm", label: "mm" },
  { value: "µm", label: "µm" },
  { value: "m", label: "m" },
];

const COLORMAP_OPTIONS = [
  { value: "turbo", label: "Turbo" },
  { value: "viridis", label: "Viridis" },
  { value: "jet", label: "Jet" },
  { value: "coolwarm", label: "Coolwarm" },
  { value: "plasma", label: "Plasma" },
  { value: "inferno", label: "Inferno" },
];

export function VisualizationControls() {
  const displayComponent = useAppStore((s) => s.displayComponent);
  const setDisplayComponent = useAppStore((s) => s.setDisplayComponent);
  const hasStrain = useAppStore((s) => s.hasStrain);
  const vis = useAppStore((s) => s.visSettings);
  const update = useAppStore((s) => s.updateVisSettings);
  const referenceFrame = useAppStore((s) => s.referenceFrame);
  const setReferenceFrame = useAppStore((s) => s.setReferenceFrame);
  const numFrames = useAppStore((s) => s.numFrames);

  const [showCbPanel, setShowCbPanel] = useState(false);
  const cbBtnRef = useRef<HTMLButtonElement>(null);
  const physicalEnabled = vis.physicalEnabled;

  // Local string state for numeric fields that need decimal input
  const [ratioText, setRatioText] = useState(String(vis.physicalRatio));
  const [fpsText, setFpsText] = useState(String(vis.fps));

  const allComponents = [
    ...DISPLACEMENT_COMPONENTS,
    ...(hasStrain ? STRAIN_COMPONENTS : []),
  ];

  return (
    <>
      <CollapsibleSection title="Visualization Controls">
        <FieldRow label="Display">
          <SelectField
            value={displayComponent}
            options={allComponents}
            onChange={(v) => setDisplayComponent(v as DisplayComponent)}
          />
        </FieldRow>
        {displayComponent === "von_mises" && (
          <div className="text-[9px] text-yellow-400/70 mt-0.5">
            * 2D plane-stress assumption (σ_z = 0)
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

        <div className="h-px bg-[var(--border)] my-1" />

        <FieldRow label="Fixed Color Range">
          <Toggle
            checked={vis.fixedRange}
            onChange={(v) => update({ fixedRange: v })}
          />
        </FieldRow>
        {vis.fixedRange && (
          <div className="flex items-center gap-1">
            <SmallInput
              value={vis.vminU}
              onChange={(v) => update({ vminU: v })}
              placeholder="min"
            />
            <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
            <SmallInput
              value={vis.vmaxU}
              onChange={(v) => update({ vmaxU: v })}
              placeholder="max"
            />
          </div>
        )}

        <FieldRow label="Log Scale">
          <Toggle
            checked={vis.logScale}
            onChange={(v) => update({ logScale: v })}
          />
        </FieldRow>

        <div className="h-px bg-[var(--border)] my-1" />

        <FieldRow label="Zero Frame">
          <SmallInput
            value={String(referenceFrame + 1)}
            onChange={(v) => {
              if (v === "" || v === "1") { setReferenceFrame(0); return; }
              const n = parseInt(v);
              if (!isNaN(n) && n >= 1 && n <= numFrames) setReferenceFrame(n - 1);
            }}
            placeholder="1"
          />
        </FieldRow>
        {referenceFrame > 0 && (
          <div className="text-[9px] text-yellow-400/70 mt-0.5">
            Displacement = frame[i] − frame[{referenceFrame + 1}]. Strain unaffected.
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection title="Physical Units" defaultOpen={false}>
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
            <div className="text-[9px] text-[var(--muted-foreground)] mt-0.5">
              Used for velocity display and strain rate (dε/dt)
            </div>
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection title="Background" defaultOpen={false}>
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="pp-bg"
              checked={vis.background === "reference"}
              onChange={() => update({ background: "reference" })}
              className="accent-[var(--primary)] w-3 h-3"
            />
            <span className={`text-[11px] ${vis.background === "reference" ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"}`}>Reference Frame</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="pp-bg"
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
