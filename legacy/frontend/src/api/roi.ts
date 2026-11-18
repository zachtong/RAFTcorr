import client from "./client";
import type { RoiResult } from "@/types/api";

export async function addPolygon(
  points: [number, number][],
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/polygon", {
    points, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function addRectangle(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/rectangle", {
    x0, y0, x1, y1, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function addCircle(
  cx: number,
  cy: number,
  r: number,
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/circle", {
    cx, cy, r, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function importMask(
  path: string,
  minArea: number = 0,
  smoothRadius: number = 0,
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/import", {
    path, min_area: minArea, smooth_radius: smoothRadius,
    frame_idx: frameIdx,
  });
  return data;
}

export async function invertMask(frameIdx: number = 0): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/invert", {
    frame_idx: frameIdx,
  });
  return data;
}

export function maskImageUrl(frameIdx: number = 0): string {
  return `/api/roi/mask?frame_idx=${frameIdx}`;
}

export async function confirmRoi(): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/confirm");
  return data;
}

export async function clearRoi(): Promise<void> {
  await client.delete("/roi");
}

export async function clearFrameRoi(frameIdx: number): Promise<void> {
  await client.delete(`/roi/frame/${frameIdx}`);
}

export async function clearAllFrameRois(): Promise<void> {
  await client.delete("/roi/frames/clear-all");
}

export interface FramesRoiStatus {
  total_frames: number;
  frames_with_roi: number[];
  frame_0_confirmed: boolean;
}

export async function getFramesRoiStatus(): Promise<FramesRoiStatus> {
  const { data } = await client.get<FramesRoiStatus>("/roi/frames/status");
  return data;
}

export interface BatchImportResult {
  imported: number;
  total_files: number;
  assignments: Record<string, string>;
}

export async function batchImportRois(
  folder: string,
  strategy: "auto_match" | "sequential" = "sequential",
): Promise<BatchImportResult> {
  const { data } = await client.post<BatchImportResult>("/roi/frames/batch-import", {
    folder, strategy,
  });
  return data;
}

export async function exportMask(): Promise<void> {
  const resp = await client.get("/roi/mask/binary", { responseType: "blob" });
  const url = URL.createObjectURL(resp.data);
  const a = document.createElement("a");
  a.href = url;
  a.download = "roi_mask.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Fetch the current ROI mask for a frame as a Blob (used by undo history). */
export async function fetchMaskBlob(frameIdx: number): Promise<Blob | null> {
  try {
    const resp = await client.get("/roi/mask/binary", {
      params: { frame_idx: frameIdx },
      responseType: "blob",
    });
    return resp.data as Blob;
  } catch {
    return null;
  }
}

/** Replace a frame's ROI mask with an uploaded PNG (empty=true clears it). */
export async function uploadMask(
  frameIdx: number,
  blob: Blob | null,
): Promise<void> {
  const form = new FormData();
  form.append("frame_idx", String(frameIdx));
  if (blob == null) {
    form.append("empty", "1");
  } else {
    form.append("mask", blob, "mask.png");
  }
  await client.post("/roi/mask/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}
