import { useState, useEffect } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { Toggle } from "@/components/shared/Toggle";
import { SelectField } from "@/components/shared/SelectField";
import { useAppStore } from "@/stores/appStore";

const COLOR_OPTIONS = [
  { value: "white", label: "White" },
  { value: "black", label: "Black" },
  { value: "red", label: "Red" },
  { value: "cyan", label: "Cyan" },
  { value: "yellow", label: "Yellow" },
];

const STREAM_QUALITY_OPTIONS = [
  { value: "1", label: "Full" },
  { value: "2", label: "Fine" },
  { value: "4", label: "Balanced" },
  { value: "8", label: "Fast" },
];

export function VelocityArrowControls() {
  const displayComponent = useAppStore((s) => s.displayComponent);
  const hasStrain = useAppStore((s) => s.hasStrain);
  const arrows = useAppStore((s) => s.arrowSettings);
  const update = useAppStore((s) => s.updateArrowSettings);

  const isVectorComponent = ["u", "v", "magnitude", "velocity"].includes(displayComponent);
  const sectionTitle = displayComponent === "velocity" ? "Velocity Arrows" : "Displacement Arrows";

  // Nothing to show if neither vector arrows nor principal dirs are relevant
  if (!isVectorComponent && !hasStrain) return null;

  return (
    <>
      {isVectorComponent && (
        <CollapsibleSection title={sectionTitle} defaultOpen={true}>
          <FieldRow label="Show Quiver">
            <Toggle
              checked={arrows.showQuiver}
              onChange={(v) => update({ showQuiver: v })}
            />
          </FieldRow>

          <FieldRow label="Show Streamlines">
            <Toggle
              checked={arrows.showStreamlines}
              onChange={(v) => update({ showStreamlines: v })}
            />
          </FieldRow>

          {arrows.showStreamlines && (
            <FieldRow label="Stream Quality">
              <SelectField
                value={String(arrows.streamQuality)}
                options={STREAM_QUALITY_OPTIONS}
                onChange={(v) => update({ streamQuality: parseInt(v, 10) })}
              />
            </FieldRow>
          )}

          <div className="h-px bg-[var(--border)] my-1" />

          <FieldRow label="Spacing (px)">
            <DebouncedNumericInput
              value={arrows.spacing}
              min={10}
              max={500}
              onChange={(v) => update({ spacing: v })}
            />
          </FieldRow>

          <FieldRow label="Scale">
            <DebouncedNumericInput
              value={arrows.scale}
              min={1}
              max={1000}
              onChange={(v) => update({ scale: v })}
            />
          </FieldRow>

          <FieldRow label="Color">
            <SelectField
              value={arrows.color}
              options={COLOR_OPTIONS}
              onChange={(v) => update({ color: v })}
            />
          </FieldRow>

          <FieldRow label="Line Width">
            <DebouncedNumericInput
              value={arrows.lineWidth}
              min={0.1}
              max={10}
              onChange={(v) => update({ lineWidth: v })}
            />
          </FieldRow>
        </CollapsibleSection>
      )}

      {hasStrain && (
        <CollapsibleSection title="Principal Directions" defaultOpen={false}>
          <FieldRow label="Show Crosses">
            <Toggle
              checked={arrows.showPrincipalDirs}
              onChange={(v) => update({ showPrincipalDirs: v })}
            />
          </FieldRow>

          {arrows.showPrincipalDirs && (
            <>
              <FieldRow label="Spacing (px)">
                <DebouncedNumericInput
                  value={arrows.spacing}
                  min={10}
                  max={500}
                  onChange={(v) => update({ spacing: v })}
                />
              </FieldRow>

              <FieldRow label="Scale">
                <DebouncedNumericInput
                  value={arrows.scale}
                  min={1}
                  max={1000}
                  onChange={(v) => update({ scale: v })}
                />
              </FieldRow>

              <FieldRow label="Line Width">
                <DebouncedNumericInput
                  value={arrows.lineWidth}
                  min={0.1}
                  max={10}
                  onChange={(v) => update({ lineWidth: v })}
                />
              </FieldRow>

              <div className="mt-1 px-2">
                <div className="flex items-center gap-3 text-[10px] text-[var(--muted-foreground)]">
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-3 h-0.5 bg-red-500" /> e1
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-3 h-0.5 bg-blue-500" /> e2
                  </span>
                </div>
              </div>
            </>
          )}
        </CollapsibleSection>
      )}
    </>
  );
}

/** Input that only commits to the store on blur or Enter key. */
function DebouncedNumericInput({
  value,
  onChange,
  min,
  max,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
}) {
  const [local, setLocal] = useState(String(value));

  useEffect(() => {
    setLocal(String(value));
  }, [value]);

  const commit = () => {
    const n = parseFloat(local);
    if (!isNaN(n)) {
      onChange(Math.max(min, Math.min(max, n)));
    } else {
      setLocal(String(value));
    }
  };

  return (
    <input
      type="text"
      value={local}
      onChange={(e) => setLocal(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => { if (e.key === "Enter") commit(); }}
      className="w-14 h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[11px] text-[var(--foreground)] text-center focus:border-[var(--primary)] focus:outline-none"
    />
  );
}
