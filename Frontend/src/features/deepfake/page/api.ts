import { API_BASE } from "@/lib/api";
import type { CustomDataset, RecordingInfo, UserClip } from "../types";

/** POST to one of the deepfake task's endpoints and unwrap FastAPI's `detail`. */
export async function postDeepfake<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}/tasks/deepfake/${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload as T;
}

/** The visitor's own clips carry an `up_` id; everything else is a demo clip. */
export const isUserClip = (recordingId: string) => recordingId.startsWith("up_");

/** Clips in one of the visitor's custom datasets carry a `cd_` id. */
export const isCustomDatasetClip = (recordingId: string) => recordingId.startsWith("cd_");

export const audioUrlFor = (recordingId: string) =>
  isUserClip(recordingId)
    ? `${API_BASE}/tasks/deepfake/uploads/${encodeURIComponent(recordingId)}/audio`
    : isCustomDatasetClip(recordingId)
      ? `${API_BASE}/tasks/deepfake/datasets/clips/${encodeURIComponent(recordingId)}/audio`
      : `${API_BASE}/tasks/deepfake/dataset/recordings/${encodeURIComponent(recordingId)}/audio`;

async function unwrap<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload as T;
}

export async function listUserClips(): Promise<UserClip[]> {
  const response = await fetch(`${API_BASE}/tasks/deepfake/uploads`, { credentials: "include" });
  return (await unwrap<{ recordings: UserClip[] }>(response)).recordings ?? [];
}

export async function uploadUserClip(file: Blob, filename: string, source: UserClip["source"]): Promise<UserClip> {
  const form = new FormData();
  form.append("file", file, filename);
  form.append("source", source);
  const response = await fetch(`${API_BASE}/tasks/deepfake/uploads`, {
    method: "POST",
    credentials: "include",
    body: form,
  });
  return unwrap<UserClip>(response);
}

export async function deleteUserClip(recordingId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/tasks/deepfake/uploads/${encodeURIComponent(recordingId)}`, {
    method: "DELETE",
    credentials: "include",
  });
  await unwrap(response);
}

export const errorMessage = (caught: unknown, fallback: string) =>
  caught instanceof Error ? caught.message : fallback;

export const formatSeconds = (seconds: number | null | undefined) =>
  seconds === null || seconds === undefined ? "n/a" : `${seconds.toFixed(2)}s`;

export const formatBytes = (bytes: number) =>
  bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(2)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

// ── Custom datasets (Manage Datasets) ───────────────────────────────────────

const datasetUrl = (name?: string, rest = "") =>
  `${API_BASE}/tasks/deepfake/datasets${name !== undefined ? `/${encodeURIComponent(name)}` : ""}${rest}`;

export interface DatasetLimits {
  max_datasets: number;
  max_files_per_dataset: number;
  max_file_mb: number;
  ttl_days: number;
}

export async function listDatasets(): Promise<{ datasets: CustomDataset[]; limits: DatasetLimits }> {
  return unwrap(await fetch(datasetUrl(), { credentials: "include" }));
}

export async function createDataset(name: string): Promise<CustomDataset> {
  const form = new FormData();
  form.append("dataset_name", name);
  return unwrap(await fetch(datasetUrl(), { method: "POST", credentials: "include", body: form }));
}

export async function deleteDataset(name: string): Promise<void> {
  await unwrap(await fetch(datasetUrl(name), { method: "DELETE", credentials: "include" }));
}

export async function listDatasetRecordings(name: string): Promise<RecordingInfo[]> {
  return (await unwrap<{ recordings: RecordingInfo[] }>(await fetch(datasetUrl(name, "/recordings"), { credentials: "include" })))
    .recordings;
}

export async function uploadDatasetFiles(
  name: string,
  files: File[],
): Promise<{ uploaded_files: RecordingInfo[]; errors: { filename: string; error: string }[] }> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  return unwrap(await fetch(datasetUrl(name, "/files"), { method: "POST", credentials: "include", body: form }));
}

export async function uploadDatasetLabels(name: string, file: File): Promise<CustomDataset> {
  const form = new FormData();
  form.append("file", file, file.name);
  return unwrap(await fetch(datasetUrl(name, "/labels"), { method: "POST", credentials: "include", body: form }));
}
