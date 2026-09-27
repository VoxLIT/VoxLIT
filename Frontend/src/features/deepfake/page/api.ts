import { API_BASE } from "@/lib/api";
import type { UserClip } from "../types";

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

export const audioUrlFor = (recordingId: string) =>
  isUserClip(recordingId)
    ? `${API_BASE}/tasks/deepfake/uploads/${encodeURIComponent(recordingId)}/audio`
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
