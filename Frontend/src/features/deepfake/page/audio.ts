import { useEffect, useState, useSyncExternalStore } from "react";

export interface Waveform {
  /** Peak envelope, normalised to the clip's loudest bucket. */
  peaks: number[];
  duration: number;
}

const cache = new Map<string, Promise<Waveform>>();

async function decode(url: string, buckets: number): Promise<Waveform> {
  const response = await fetch(url, { credentials: "include" });
  const buffer = await response.arrayBuffer();
  const Context =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const context = new Context();
  try {
    const audio = await context.decodeAudioData(buffer);
    const channel = audio.getChannelData(0);
    const size = Math.max(1, Math.floor(channel.length / buckets));
    const peaks: number[] = [];
    for (let index = 0; index < buckets; index += 1) {
      let peak = 0;
      const start = index * size;
      for (let offset = 0; offset < size; offset += 1) {
        const value = Math.abs(channel[start + offset] ?? 0);
        if (value > peak) peak = value;
      }
      peaks.push(peak);
    }
    const loudest = Math.max(...peaks, 1e-6);
    return { peaks: peaks.map((value) => value / loudest), duration: audio.duration };
  } finally {
    context.close().catch(() => undefined);
  }
}

/** Decode a clip once (per url and resolution) and share the envelope. */
export function useWaveform(url: string | undefined, buckets = 160): Waveform | null {
  const [waveform, setWaveform] = useState<Waveform | null>(null);

  useEffect(() => {
    setWaveform(null);
    if (!url) return;
    let cancelled = false;
    const key = `${url}#${buckets}`;
    if (!cache.has(key)) {
      const pending = decode(url, buckets);
      pending.catch(() => cache.delete(key));
      cache.set(key, pending);
    }
    cache
      .get(key)!
      .then((result) => {
        if (!cancelled) setWaveform(result);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [url, buckets]);

  return waveform;
}

/** play() returns a promise in browsers but not in every environment. */
export const safePlay = (element: HTMLMediaElement) => {
  try {
    const pending = element.play() as Promise<void> | undefined;
    pending?.catch(() => undefined);
  } catch {
    // playback refused (autoplay policy, no media support) — stay silent
  }
};

/**
 * One shared element for quick previews (the hover popup and the library
 * cards), so starting one preview always stops the previous one.
 */
let previewElement: HTMLAudioElement | null = null;
let previewUrl: string | null = null;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((listener) => listener());

export const preview = {
  toggle(url: string) {
    if (typeof Audio === "undefined") return;
    if (!previewElement) {
      previewElement = new Audio();
      previewElement.addEventListener("ended", () => {
        previewUrl = null;
        notify();
      });
    }
    if (previewUrl === url && !previewElement.paused) {
      previewElement.pause();
      previewUrl = null;
    } else {
      previewElement.src = url;
      previewElement.currentTime = 0;
      safePlay(previewElement);
      previewUrl = url;
    }
    notify();
  },
  stop() {
    previewElement?.pause();
    previewUrl = null;
    notify();
  },
};

export function usePreviewUrl(): string | null {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => previewUrl,
    () => null,
  );
}
