import type { DisplayComponent } from "@/types/api";

export interface UnitInfo {
  scale: number;
  unit: string;
}

const STRAIN_COMPONENTS = new Set([
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
]);
const STRAIN_RATE_COMPONENTS = new Set(["dexx_dt", "deyy_dt", "dexy_dt"]);

/** Compute the colorbar unit string and display scale factor. */
export function getUnitInfo(
  component: DisplayComponent,
  physicalEnabled: boolean,
  physicalRatio: number,
  physicalUnit: string,
  fps: number,
): UnitInfo {
  if (component === "rotation") {
    return { scale: 1, unit: "[deg]" };
  }
  if (STRAIN_COMPONENTS.has(component)) {
    return { scale: 1, unit: "[-]" };
  }
  if (STRAIN_RATE_COMPONENTS.has(component)) {
    return { scale: fps, unit: "[1/s]" };
  }
  if (component === "velocity") {
    if (physicalEnabled) {
      return { scale: fps * physicalRatio, unit: `[${physicalUnit}/s]` };
    }
    return { scale: fps, unit: "[px/s]" };
  }
  // displacement: u, v, magnitude
  if (physicalEnabled) {
    return { scale: physicalRatio, unit: `[${physicalUnit}]` };
  }
  return { scale: 1, unit: "[px]" };
}
