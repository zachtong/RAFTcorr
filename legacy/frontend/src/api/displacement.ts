import client from "./client";
import type { DisplacementInfo, FrameData } from "@/types/api";

export async function getDisplacementInfo(): Promise<DisplacementInfo> {
  const { data } = await client.get<DisplacementInfo>("/displacement/info");
  return data;
}

export async function getFrameData(
  idx: number,
  component: string
): Promise<FrameData> {
  const { data } = await client.get<FrameData>(
    `/displacement/frame/${idx}`,
    { params: { component } }
  );
  return data;
}

export function renderUrl(
  idx: number,
  params: Record<string, string | number>
): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  return `/api/displacement/render/${idx}?${qs}`;
}
