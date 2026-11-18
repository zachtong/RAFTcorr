import client from "./client";

export interface SessionSummary {
  num_displacement_frames: number;
  num_strain_frames: number;
  strain_components: string[];
  has_roi: boolean;
  roi_confirmed: boolean;
  num_probes: number;
  image_dir: string;
  image_dir_exists: boolean;
  image_width: number;
  image_height: number;
}

export async function saveSession(params: {
  file_path: string;
  overwrite?: boolean;
}): Promise<{ ok: boolean; path: string; size_mb: number }> {
  const { data } = await client.post("/session/save", params);
  return data;
}

export async function loadSession(params: {
  file_path: string;
}): Promise<{ ok: boolean; warnings: string[]; summary: SessionSummary }> {
  const { data } = await client.post("/session/load", params);
  return data;
}

export async function getSessionInfo(): Promise<{
  has_results: boolean;
  num_displacement_frames: number;
  num_strain_frames: number;
  strain_components: string[];
  has_roi: boolean;
  roi_confirmed: boolean;
  num_probes: number;
  image_dir: string;
  image_width: number;
  image_height: number;
  config: Record<string, unknown>;
}> {
  const { data } = await client.get("/session/info");
  return data;
}
