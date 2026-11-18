import client from "./client";
import type { ProbeData, TimeSeriesResult, KymographResult } from "@/types/api";

export async function listProbes(): Promise<ProbeData[]> {
  const { data } = await client.get<{ probes: ProbeData[] }>("/probes");
  return data.probes;
}

export async function addPoint(x: number, y: number): Promise<ProbeData> {
  const { data } = await client.post<ProbeData>("/probes/point", { x, y });
  return data;
}

export async function addLine(
  p1: [number, number],
  p2: [number, number]
): Promise<ProbeData> {
  const { data } = await client.post<ProbeData>("/probes/line", { p1, p2 });
  return data;
}

export async function addArea(
  shape: string,
  coords: unknown
): Promise<ProbeData> {
  const { data } = await client.post<ProbeData>("/probes/area", {
    shape,
    data: coords,
  });
  return data;
}

export async function removeProbe(
  id: number,
  type: string
): Promise<void> {
  await client.delete(`/probes/${id}`, { params: { type } });
}

export async function clearProbes(type?: string): Promise<void> {
  await client.delete("/probes", { params: type ? { type } : {} });
}

export async function getTimeSeries(
  component: string,
  type: string = "point",
  metric: string = "avg"
): Promise<TimeSeriesResult> {
  const { data } = await client.get<TimeSeriesResult>("/probes/timeseries", {
    params: { component, type, metric },
  });
  return data;
}

export interface ExtensometerResult {
  initial_length: number;
  lengths: number[];
  strains: number[];
  num_frames: number;
}

export async function getExtensometer(lineId: number): Promise<ExtensometerResult> {
  const { data } = await client.get<ExtensometerResult>(
    `/probes/extensometer/${lineId}`
  );
  return data;
}

export async function getKymograph(
  lineId: number,
  component: string
): Promise<KymographResult> {
  const { data } = await client.get<KymographResult>(
    `/probes/kymograph/${lineId}`,
    { params: { component } }
  );
  return data;
}

export async function downloadProbeCSV(
  component: string,
  type: string,
  metric: string,
  includeExtensometer: boolean = false
): Promise<void> {
  const response = await client.get("/probes/export/csv", {
    params: {
      component,
      type,
      metric,
      extensometer: includeExtensometer ? "true" : "false",
    },
    responseType: "blob",
  });
  const blob = new Blob([response.data], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `probes_${type}_${component}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
