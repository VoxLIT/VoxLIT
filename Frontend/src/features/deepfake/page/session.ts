import { type Dispatch, type SetStateAction, useCallback, useState } from "react";

/**
 * Page state that survives a refresh: the chosen detector, the clip under
 * study, the map settings and every result already computed. Kept in
 * sessionStorage, so it belongs to this browser tab and ends with it.
 * Storage can be unavailable (private mode, full quota), so every access is
 * guarded and the page simply starts fresh when it fails.
 */
const PREFIX = "voxlit.deepfake.";

export function readSession<T>(key: string): T | null {
  try {
    const raw = window.sessionStorage.getItem(PREFIX + key);
    return raw === null ? null : (JSON.parse(raw) as T);
  } catch {
    return null;
  }
}

export function writeSession(key: string, value: unknown): void {
  try {
    if (value === null || value === undefined) window.sessionStorage.removeItem(PREFIX + key);
    else window.sessionStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // Quota or privacy settings: the result still shows, it just will not survive a refresh.
  }
}

/** A result's storage key: which analysis, by which detector, on which clip. */
export const resultKey = (kind: string, ...parts: string[]) => `result.${kind}.${parts.join(".")}`;

/** useState that is restored from, and written back to, this tab's session. */
export function useSessionState<T>(key: string, initial: T | (() => T)): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    const stored = readSession<T>(key);
    if (stored !== null) return stored;
    return typeof initial === "function" ? (initial as () => T)() : initial;
  });
  const set = useCallback<Dispatch<SetStateAction<T>>>(
    (next) =>
      setValue((current) => {
        const resolved = typeof next === "function" ? (next as (previous: T) => T)(current) : next;
        writeSession(key, resolved);
        return resolved;
      }),
    [key],
  );
  return [value, set];
}
