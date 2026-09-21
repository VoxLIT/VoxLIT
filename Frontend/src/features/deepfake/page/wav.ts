/**
 * Turn whatever the browser can decode — a MediaRecorder webm/opus blob, an
 * m4a, an mp3 — into 16 kHz mono 16-bit PCM WAV before upload.
 *
 * The backend decodes with libsndfile, which cannot read webm or m4a, and
 * every detector resamples to 16 kHz mono anyway, so doing it here costs no
 * information and keeps the upload small (~32 KB per second).
 */

export const TARGET_RATE = 16_000;

type ContextCtor = typeof AudioContext;

function audioContextCtor(): ContextCtor | null {
  if (typeof window === "undefined") return null;
  return window.AudioContext || (window as unknown as { webkitAudioContext?: ContextCtor }).webkitAudioContext || null;
}

/** Mix to mono and resample with an OfflineAudioContext. */
async function toMono16k(buffer: AudioBuffer): Promise<Float32Array> {
  const length = Math.max(1, Math.ceil(buffer.duration * TARGET_RATE));
  const Offline =
    window.OfflineAudioContext ||
    (window as unknown as { webkitOfflineAudioContext?: typeof OfflineAudioContext }).webkitOfflineAudioContext;
  const offline = new Offline(1, length, TARGET_RATE);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

export function encodeWav(samples: Float32Array, sampleRate = TARGET_RATE): Blob {
  const bytesPerSample = 2;
  const dataBytes = samples.length * bytesPerSample;
  const view = new DataView(new ArrayBuffer(44 + dataBytes));
  const text = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, dataBytes, true);
  let offset = 44;
  for (let index = 0; index < samples.length; index += 1, offset += 2) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

export interface ConvertedClip {
  wav: Blob;
  duration: number;
}

/** Decode any browser-playable audio blob and re-encode it as 16 kHz mono WAV. */
export async function convertToWav(input: Blob): Promise<ConvertedClip> {
  const Context = audioContextCtor();
  if (!Context) throw new Error("This browser cannot decode audio.");
  const context = new Context();
  try {
    const decoded = await context.decodeAudioData(await input.arrayBuffer());
    const samples = await toMono16k(decoded);
    return { wav: encodeWav(samples), duration: decoded.duration };
  } catch (caught) {
    if (caught instanceof Error && caught.message === "This browser cannot decode audio.") throw caught;
    throw new Error("Could not read this file as audio. Try WAV, MP3, FLAC, OGG or M4A.");
  } finally {
    context.close().catch(() => undefined);
  }
}

/** `voice note.m4a` -> `voice note.wav` */
export const wavName = (filename: string) => `${filename.replace(/\.[^./\\]+$/, "") || "clip"}.wav`;
