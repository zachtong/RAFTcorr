import { useState } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { SmallInput } from "@/components/shared/SmallInput";
import { SelectField } from "@/components/shared/SelectField";
import { ParamTooltip } from "@/components/shared/ParamTooltip";
import { useAppStore } from "@/stores/appStore";
import { calculateStrain } from "@/api/strain";

const METHOD_OPTIONS = [
  { value: "green_lagrange", label: "Green-Lagrange" },
  { value: "engineering", label: "Engineering" },
];

const WEIGHTING_OPTIONS = [
  { value: "gaussian", label: "Gaussian" },
  { value: "uniform", label: "Uniform" },
];

export function StrainControls() {
  const hasStrain = useAppStore((s) => s.hasStrain);
  const strainComputing = useAppStore((s) => s.strainComputing);
  const strainProgress = useAppStore((s) => s.strainProgress);
  const setStrainComputing = useAppStore((s) => s.setStrainComputing);

  const [method, setMethod] = useState("green_lagrange");
  const [vsgSize, setVsgSize] = useState("31");
  const [step, setStep] = useState("15");
  const [polyOrder, setPolyOrder] = useState("1");
  const [weighting, setWeighting] = useState("uniform");
  const [spatialSigma, setSpatialSigma] = useState("0");
  const [temporalSigma, setTemporalSigma] = useState("0");
  const [gaussianSigma, setGaussianSigma] = useState("");
  const [condThreshold, setCondThreshold] = useState("1000");
  const [erosionAuto, setErosionAuto] = useState(true);
  const [erosionPx, setErosionPx] = useState("");
  const fps = useAppStore((s) => s.visSettings.fps);

  const handleCalculate = async () => {
    if (strainComputing) return;

    // Sanitize VSG size: must be odd integer >= 9
    let vsg = parseInt(vsgSize);
    if (isNaN(vsg) || vsg < 9) vsg = 9;
    if (vsg % 2 === 0) vsg = vsg + 1;
    setVsgSize(String(vsg));

    setStrainComputing(true);
    try {
      await calculateStrain({
        method,
        vsg_size: vsg,
        step: parseInt(step),
        poly_order: parseInt(polyOrder),
        weighting,
        spatial_sigma: parseFloat(spatialSigma) || 0,
        temporal_sigma: parseFloat(temporalSigma) || 0,
        gaussian_sigma: gaussianSigma ? parseFloat(gaussianSigma) : undefined,
        fps: fps || 1,
        cond_threshold: parseFloat(condThreshold) || 1000,
        boundary_erosion: erosionAuto ? -1 : (parseInt(erosionPx) || 0),
      });
    } catch (e) {
      console.error("Failed to start strain calculation:", e);
      setStrainComputing(false);
    }
  };

  return (
    <CollapsibleSection title="Strain Calculation">
      <FieldRow label={<span className="flex items-center gap-1">Method <ParamTooltip paramKey="strainMethod" /></span>}>
        <SelectField
          value={method}
          options={METHOD_OPTIONS}
          onChange={setMethod}
        />
      </FieldRow>
      <FieldRow label={<span className="flex items-center gap-1">VSG Size <ParamTooltip paramKey="vsgSize" /></span>}>
        <SmallInput
          value={vsgSize}
          onChange={setVsgSize}
          onBlur={() => {
            let n = parseInt(vsgSize);
            if (isNaN(n) || n < 9) n = 9;
            if (n % 2 === 0) n = n + 1;               // even → next odd
            setVsgSize(String(n));
          }}
          placeholder="odd ≥ 9"
        />
      </FieldRow>
      <FieldRow label={<span className="flex items-center gap-1">Step <ParamTooltip paramKey="step" /></span>}>
        <SmallInput value={step} onChange={setStep} placeholder="e.g. 15" />
      </FieldRow>
      <FieldRow label={<span className="flex items-center gap-1">Poly Order <ParamTooltip paramKey="polyOrder" /></span>}>
        <SmallInput
          value={polyOrder}
          onChange={(v) => {
            const n = parseInt(v);
            if (v === "" || (!isNaN(n) && n >= 1 && n <= 2)) setPolyOrder(v);
          }}
          placeholder="1 or 2"
        />
      </FieldRow>
      <FieldRow label={<span className="flex items-center gap-1">Weighting <ParamTooltip paramKey="weighting" /></span>}>
        <SelectField
          value={weighting}
          options={WEIGHTING_OPTIONS}
          onChange={setWeighting}
        />
      </FieldRow>
      {weighting === "gaussian" && (
        <FieldRow label="Gauss σ">
          <SmallInput value={gaussianSigma} onChange={setGaussianSigma} placeholder="auto" />
        </FieldRow>
      )}
      <div className="h-px bg-[var(--border)] my-1.5" />
      <div className="text-[9px] text-[var(--muted-foreground)] mb-1 uppercase tracking-wider">Smoothing</div>
      <FieldRow label={<span className="flex items-center gap-1">Spatial <ParamTooltip paramKey="spatialSigma" /></span>}>
        <div className="flex items-center gap-1">
          <SmallInput value={spatialSigma} onChange={setSpatialSigma} placeholder="0 = off" />
          <span className="text-[9px] text-[var(--muted-foreground)] shrink-0">px</span>
        </div>
      </FieldRow>
      {parseFloat(spatialSigma) > 0 && (
        <div className="text-[9px] text-[var(--muted-foreground)] mt-0.5">
          Gaussian blur on displacement before strain calc
        </div>
      )}
      <FieldRow label={<span className="flex items-center gap-1">Temporal <ParamTooltip paramKey="temporalSigma" /></span>}>
        <div className="flex items-center gap-1">
          <SmallInput value={temporalSigma} onChange={setTemporalSigma} placeholder="0 = off" />
          <span className="text-[9px] text-[var(--muted-foreground)] shrink-0">frames</span>
        </div>
      </FieldRow>
      {parseFloat(temporalSigma) > 0 && (
        <div className="text-[9px] text-[var(--muted-foreground)] mt-0.5">
          Gaussian smoothing of strain across time
        </div>
      )}
      <div className="h-px bg-[var(--border)] my-1.5" />
      <div className="text-[9px] text-[var(--muted-foreground)] mb-1 uppercase tracking-wider">Quality Filtering</div>
      <FieldRow label={<span className="flex items-center gap-1">Cond. # <ParamTooltip paramKey="condThreshold" /></span>}>
        <div className="flex items-center gap-1">
          <SmallInput
            value={condThreshold}
            onChange={setCondThreshold}
            placeholder="1000"
          />
          <span className="text-[9px] text-[var(--muted-foreground)] shrink-0">max</span>
        </div>
      </FieldRow>
      <FieldRow label={<span className="flex items-center gap-1">Erosion <ParamTooltip paramKey="erosion" /></span>}>
        <div className="flex items-center gap-1.5">
          <label className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer shrink-0">
            <input
              type="checkbox"
              checked={erosionAuto}
              onChange={(e) => setErosionAuto(e.target.checked)}
              className="w-3 h-3 accent-[var(--primary)]"
            />
            Auto
          </label>
          {!erosionAuto && (
            <div className="flex items-center gap-1">
              <SmallInput
                value={erosionPx}
                onChange={setErosionPx}
                placeholder="0"
              />
              <span className="text-[9px] text-[var(--muted-foreground)] shrink-0">px</span>
            </div>
          )}
        </div>
      </FieldRow>
      {erosionAuto && (
        <div className="text-[9px] text-[var(--muted-foreground)] mt-0.5">
          Auto = VSG / 2 ({Math.floor(parseInt(vsgSize) / 2) || "?"} px)
        </div>
      )}
      <button
        onClick={handleCalculate}
        disabled={strainComputing}
        className={`w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-[11px] mt-1 ${
          strainComputing
            ? "bg-[var(--secondary)] text-[var(--muted-foreground)] animate-pulse"
            : hasStrain
            ? "bg-green-600/20 text-green-400 border border-green-600/40 hover:bg-green-600/30 cursor-pointer"
            : "bg-[var(--primary)] hover:bg-[var(--primary)]/90 text-white"
        }`}
      >
        {strainComputing
          ? `Computing... ${strainProgress.toFixed(0)}%`
          : hasStrain
          ? "Recalculate Strain"
          : "Calculate Strain"}
      </button>
    </CollapsibleSection>
  );
}
