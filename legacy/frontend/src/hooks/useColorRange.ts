import { useState, useEffect } from "react";
import client from "@/api/client";
import { useAppStore } from "@/stores/appStore";

const STRAIN_COMPONENTS = new Set([
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
  "dexx_dt", "deyy_dt", "dexy_dt",
]);

export interface ColorRange {
  vmin: number;
  vmax: number;
}

/**
 * Fetch the auto vmin/vmax for a given frame and display component.
 * Hits /displacement/range or /strain/range depending on the component.
 */
export function useColorRange(
  frameIdx: number,
  component: string,
  active: boolean
): ColorRange | null {
  const [range, setRange] = useState<ColorRange | null>(null);
  const referenceFrame = useAppStore((s) => s.referenceFrame);

  useEffect(() => {
    if (!active || frameIdx < 0) {
      setRange(null);
      return;
    }

    const isStrain = STRAIN_COMPONENTS.has(component);
    const refParam = !isStrain && referenceFrame > 0 ? `&ref_frame=${referenceFrame}` : "";
    const url = isStrain
      ? `/strain/range/${frameIdx}?component=${component}`
      : `/displacement/range/${frameIdx}?component=${component}${refParam}`;

    let cancelled = false;
    client
      .get<ColorRange>(url)
      .then(({ data }) => {
        if (!cancelled) setRange(data);
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [frameIdx, component, active, referenceFrame]);

  return range;
}
