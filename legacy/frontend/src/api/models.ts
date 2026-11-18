import client from "./client";
import type { ModelEntry, ModelMetadata } from "@/types/api";

export async function listModels(): Promise<ModelEntry[]> {
  const { data } = await client.get<{ models: ModelEntry[] }>("/models");
  return data.models;
}

export async function describeModel(path: string): Promise<ModelMetadata> {
  const { data } = await client.post<{ metadata: ModelMetadata }>(
    "/models/describe",
    { path }
  );
  return data.metadata;
}

export async function selectModel(
  path: string
): Promise<{ ok: boolean; label: string; summary: string; safe_pmax: number }> {
  const { data } = await client.post("/models/select", { path });
  return data;
}
