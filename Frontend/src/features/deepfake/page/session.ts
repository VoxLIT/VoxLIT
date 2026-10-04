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

/**
 * Which stored results are on screen. A result stays in the session once
 * computed, but it is only shown again after its button is clicked: leaving a
 * clip forgets that clip's results here (not in storage), so coming back to it
 * shows the buttons, and a click brings the stored result back without asking
 * the server. Kept in the session too, so a refresh shows what was on screen.
 */
const REVEALED = "revealed";

const revealedKeys = () => readSession<string[]>(REVEALED) ?? [];

export function markRevealed(key: string): void {
  const keys = revealedKeys();
  if (!keys.includes(key)) writeSession(REVEALED, [...keys, key]);
}

/** A stored result, but only if it is currently revealed. */
export function readRevealed<T>(key: string): T | null {
  return revealedKeys().includes(key) ? readSession<T>(key) : null;
}

/** Hide every per-clip result of `recordingId` until its button is clicked again. */
export function forgetRevealedClip(recordingId: string): void {
  if (!recordingId) return;
  const keys = revealedKeys();
  const kept = keys.filter((key) => !key.endsWith(`.${recordingId}`));
  if (kept.length !== keys.length) writeSession(REVEALED, kept.length ? kept : null);
}

/**
 * A button's result: the stored one if this tab already computed it, else a
 * fresh request (which the server may also answer from its own cache, and
 * which is stored here). The caller marks it revealed once it shows it, so a
 * reply that lands after the visitor moved on stays hidden.
 */
export async function loadResult<T>(key: string, fetchResult: () => Promise<T>): Promise<{ payload: T; stored: boolean }> {
  const stored = readSession<T>(key);
  if (stored !== null) return { payload: stored, stored: true };
  const payload = await fetchResult();
  writeSession(key, payload);
  return { payload, stored: false };
}
