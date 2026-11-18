import client from "./client";

export async function exportScientific(params: {
  file_path: string;
  upsample_strain?: boolean;
  metadata?: Record<string, unknown>;
  overwrite?: boolean;
  /** Subset of variable names to export. Omit/empty = export everything. */
  variables?: string[];
}): Promise<{ ok: boolean; path: string }> {
  const { data } = await client.post("/export/scientific", params);
  return data;
}

export async function exportImages(params: {
  output_dir: string;
  components: Record<string, { vmin?: number; vmax?: number }>;
  frame_range: [number, number];
  settings: Record<string, unknown>;
}): Promise<void> {
  await client.post("/export/images", params);
}

export async function getExportStatus(): Promise<{
  active: boolean;
  progress: number;
  total: number;
  percent: number;
}> {
  const { data } = await client.get("/export/images/status");
  return data;
}

export async function cancelExport(): Promise<void> {
  await client.post("/export/images/cancel");
}

export async function downloadSingleFrame(params: {
  idx: number;
  component: string;
  colormap: string;
  alpha: number;
  vmin?: number;
  vmax?: number;
  background: string;
  log_scale: boolean;
  dpi: number;
  isStrain?: boolean;
}): Promise<void> {
  const { idx, isStrain, ...rest } = params;
  const base = isStrain ? "/strain/download" : "/displacement/download";
  const queryParams = new URLSearchParams();
  for (const [k, v] of Object.entries(rest)) {
    if (v !== undefined && v !== null) queryParams.set(k, String(v));
  }
  const response = await client.get(`${base}/${idx}?${queryParams}`, {
    responseType: "blob",
  });
  const blob = new Blob([response.data], { type: "image/png" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${rest.component}_frame_${idx + 1}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function exportAnimation(params: {
  output_path: string;
  format: "gif" | "mp4";
  component: string;
  frame_range: [number, number];
  fps: number;
  loop?: boolean;
  timestamp_overlay?: boolean;
  include_colorbar?: boolean;
  resize_factor?: number;
  settings: Record<string, unknown>;
}): Promise<void> {
  await client.post("/export/animation", params);
}

export async function exportReport(params: {
  output_path: string;
  sections?: string[];
  format?: "html" | "pdf" | "both";
  theme?: "light" | "dark" | "academic" | "minimal";
  custom_title?: string;
  author?: string;
  notes?: string;
  key_frame_components?: string[];
}): Promise<{ ok: boolean; path: string; pdf_path?: string }> {
  const { data } = await client.post("/export/report", params);
  return data;
}
