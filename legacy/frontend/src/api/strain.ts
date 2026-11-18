import client from "./client";
import type { StrainStatus, FrameData } from "@/types/api";

export async function calculateStrain(params: {
  method?: string;
  vsg_size?: number;
  poly_order?: number;
  weighting?: string;
  step?: number;
  temporal_sigma?: number;
  spatial_sigma?: number;
  gaussian_sigma?: number;
  fps?: number;
  cond_threshold?: number;
  boundary_erosion?: number;
}): Promise<void> {
  await client.post("/strain/calculate", params);
}

export async function getStrainStatus(): Promise<StrainStatus> {
  const { data } = await client.get<StrainStatus>("/strain/status");
  return data;
}

export async function getStrainFrame(
  idx: number,
  component: string
): Promise<FrameData> {
  const { data } = await client.get<FrameData>(`/strain/frame/${idx}`, {
    params: { component },
  });
  return data;
}

export function strainRenderUrl(
  idx: number,
  params: Record<string, string | number>
): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  return `/api/strain/render/${idx}?${qs}`;
}
