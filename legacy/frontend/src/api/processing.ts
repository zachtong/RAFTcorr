import client from "./client";
import type { TilingPreview } from "@/types/api";

export async function configureProcessing(
  config: Record<string, unknown>
): Promise<void> {
  await client.post("/processing/configure", config);
}

export async function runProcessing(): Promise<void> {
  await client.post("/processing/run");
}

export async function pauseProcessing(): Promise<void> {
  await client.post("/processing/pause");
}

export async function resumeProcessing(): Promise<void> {
  await client.post("/processing/resume");
}

export async function stopProcessing(): Promise<void> {
  await client.post("/processing/stop");
}

export async function getProcessingStatus(): Promise<{
  active: boolean;
  paused: boolean;
  has_results: boolean;
  num_frames: number;
}> {
  const { data } = await client.get("/processing/status");
  return data;
}

export async function validateMasks(maskDir: string): Promise<{
  matched_count: number;
  total_frames: number;
  masks_needed: number;
  matched_frames: number[];
  has_frame_1: boolean;
}> {
  const { data } = await client.post("/processing/validate-masks", { mask_dir: maskDir });
  return data;
}

export async function applyFrame1Mask(maskDir: string): Promise<{
  rect: [number, number, number, number];
  area_px: number;
}> {
  const { data } = await client.post("/processing/apply-frame1-mask", { mask_dir: maskDir });
  return data;
}

export async function fetchTilingPreview(): Promise<TilingPreview> {
  const { data } = await client.get<TilingPreview>("/tiling/preview");
  return data;
}
