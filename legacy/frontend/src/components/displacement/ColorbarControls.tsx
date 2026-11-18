import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { Toggle } from "@/components/shared/Toggle";
import { SmallInput } from "@/components/shared/SmallInput";
import { SelectField } from "@/components/shared/SelectField";
import { SliderField } from "@/components/shared/SliderField";
import { ColormapBar } from "@/components/shared/ColormapBar";
import { useAppStore } from "@/stores/appStore";

const COLORMAP_OPTIONS = [
  { value: "turbo", label: "Turbo" },
  { value: "viridis", label: "Viridis" },
  { value: "jet", label: "Jet" },
  { value: "coolwarm", label: "Coolwarm" },
  { value: "plasma", label: "Plasma" },
  { value: "inferno", label: "Inferno" },
];

export function ColorbarControls() {
  const vis = useAppStore((s) => s.visSettings);
  const update = useAppStore((s) => s.updateVisSettings);

  return (
    <>
      <CollapsibleSection title="Display Controls">
        <FieldRow label="Colormap">
          <SelectField
            value={vis.colormap}
            options={COLORMAP_OPTIONS}
            onChange={(v) => update({ colormap: v })}
          />
        </FieldRow>
        <ColormapBar colormap={vis.colormap} />

        <FieldRow label="Opacity">
          <SliderField
            value={vis.alpha}
            onChange={(v) => update({ alpha: v })}
            min={0}
            max={1}
            step={0.05}
          />
        </FieldRow>

        <FieldRow label="Fixed Range">
          <Toggle
            checked={vis.fixedRange}
            onChange={(v) => update({ fixedRange: v })}
          />
        </FieldRow>

        {vis.fixedRange && (
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-[var(--muted-foreground)] w-3">U</span>
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
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-[var(--muted-foreground)] w-3">V</span>
              <SmallInput
                value={vis.vminV}
                onChange={(v) => update({ vminV: v })}
                placeholder="min"
              />
              <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
              <SmallInput
                value={vis.vmaxV}
                onChange={(v) => update({ vmaxV: v })}
                placeholder="max"
              />
            </div>
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection title="Background" defaultOpen={false}>
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="disp-bg"
              checked={vis.background === "reference"}
              onChange={() => update({ background: "reference" })}
              className="accent-[var(--primary)] w-3 h-3"
            />
            <span className="text-[11px] text-[var(--foreground)]">Reference Frame</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="disp-bg"
              checked={vis.background === "deformed"}
              onChange={() => update({ background: "deformed" })}
              className="accent-[var(--primary)] w-3 h-3"
            />
            <span className="text-[11px] text-[var(--muted-foreground)]">Deformed Frame</span>
          </label>
        </div>
      </CollapsibleSection>
    </>
  );
}
