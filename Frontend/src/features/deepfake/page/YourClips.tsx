import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { AnimatePresence, motion } from "motion/react";
import { FileAudio, Mic, Pause, Play, RotateCcw, Send, Square, Trash2, Upload } from "lucide-react";
import type { UserClip } from "../types";
import { audioUrlFor, deleteUserClip, errorMessage, formatBytes, formatSeconds, uploadUserClip } from "./api";
import { preview, usePreviewUrl } from "./audio";
import { ErrorNote, PrimaryButton } from "./ui";
import { convertToWav, wavName } from "./wav";

/** Every detector analyses at most ~30 s, so a longer take would only be cut. */
export const MAX_RECORDING_SECONDS = 30;
const MIN_SECONDS = 0.5;
const ACCEPT = "audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm";
const MAX_FILE_BYTES = 25 * 1024 * 1024;

interface YourClipsProps {
  clips: UserClip[];
  selectedId: string;
  onAdded: (clip: UserClip) => void;
  onRemoved: (recordingId: string) => void;
  onSelect: (recordingId: string) => void;
}

type Take = { blob: Blob; url: string; seconds: number };

/**
 * Bring your own voice: drop a file or record one in the browser. Either way
 * the clip is converted to 16 kHz mono WAV here, stored for this session on
 * the server, and then behaves exactly like a dataset clip — every detector
 * and every explanation on the page accepts it.
 */
export const YourClips = ({ clips, selectedId, onAdded, onRemoved, onSelect }: YourClipsProps) => {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const playingUrl = usePreviewUrl();

  const send = useCallback(
    async (blob: Blob, filename: string, source: UserClip["source"]) => {
      setError(null);
      setBusy(source === "recording" ? "Uploading your recording" : `Converting ${filename}`);
      try {
        const { wav, duration } = await convertToWav(blob);
        if (duration < MIN_SECONDS) throw new Error(`That clip is only ${duration.toFixed(1)}s. Give it at least ${MIN_SECONDS}s.`);
        setBusy("Uploading");
        const clip = await uploadUserClip(wav, wavName(filename), source);
        onAdded(clip);
      } catch (caught) {
        setError(errorMessage(caught, "Upload failed."));
      } finally {
        setBusy(null);
      }
    },
    [onAdded],
  );

  const takeFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      setError(`${file.name} is larger than 25 MB.`);
      return;
    }
    send(file, file.name, "upload");
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    takeFiles(event.dataTransfer.files);
  };

  const remove = async (recordingId: string) => {
    try {
      await deleteUserClip(recordingId);
      onRemoved(recordingId);
    } catch (caught) {
      setError(errorMessage(caught, "Could not delete the clip."));
    }
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.1fr)]">
      {/* upload */}
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`df-glass flex flex-col items-center justify-center gap-3 border-2 border-dashed p-4 text-center transition-colors ${
          dragging ? "border-cyan-300/70 bg-cyan-300/5" : "border-white/10"
        }`}
      >
        <div className="grid h-14 w-14 place-items-center rounded-2xl bg-cyan-400/10 ring-1 ring-cyan-300/30">
          <Upload className="h-7 w-7 text-cyan-200" />
        </div>
        <div>
          <div className="font-display text-lg font-semibold text-white">Upload a clip</div>
          <p className="mt-1 text-sm text-slate-400">Drop an audio file here: WAV, MP3, FLAC, OGG or M4A, up to 60 s.</p>
        </div>
        <PrimaryButton onClick={() => fileInput.current?.click()} disabled={!!busy}>
          <FileAudio className="h-4 w-4" /> Choose a file
        </PrimaryButton>
        <input
          ref={fileInput}
          type="file"
          accept={ACCEPT}
          className="hidden"
          aria-label="Upload an audio clip"
          onChange={(event) => {
            takeFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </div>

      {/* record */}
      <Recorder disabled={!!busy} onUse={(take) => send(take.blob, "recording.webm", "recording")} onError={setError} />

      {/* your clips */}
      <div className="df-glass flex flex-col gap-3 p-3">
        <div className="flex items-baseline justify-between">
          <div className="font-display text-lg font-semibold text-white">Your clips</div>
          <span className="text-xs text-slate-500">Kept for 24 h. Only you can see them.</span>
        </div>
        <AnimatePresence>{busy && <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-sm text-cyan-200">{busy}</motion.p>}</AnimatePresence>
        {error && <ErrorNote>{error}</ErrorNote>}
        {clips.length === 0 && !busy ? (
          <p className="text-sm text-slate-400">Nothing yet. Upload or record a clip and it will appear here, ready to score.</p>
        ) : (
          <ul className="space-y-2" aria-label="Your clips">
            <AnimatePresence initial={false}>
              {clips.map((clip) => {
                const url = audioUrlFor(clip.recording_id);
                const playing = playingUrl === url;
                const selected = clip.recording_id === selectedId;
                return (
                  <motion.li
                    key={clip.recording_id}
                    layout
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className={`flex items-center gap-3 rounded-2xl p-2.5 ${selected ? "bg-white/10 ring-2 ring-violet-300/70" : "bg-white/[0.035] ring-1 ring-white/10"}`}
                  >
                    <button
                      type="button"
                      onClick={() => preview.toggle(url)}
                      aria-label={`${playing ? "Stop" : "Preview"} ${clip.display_filename}`}
                      className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-white/10 text-white transition hover:bg-white/20"
                    >
                      {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
                    </button>
                    <button type="button" onClick={() => onSelect(clip.recording_id)} className="min-w-0 flex-1 text-left" aria-pressed={selected}>
                      <div className="flex items-center gap-1.5 truncate text-sm font-semibold text-white">
                        {clip.source === "recording" ? <Mic className="h-3.5 w-3.5 shrink-0 text-rose-300" /> : <FileAudio className="h-3.5 w-3.5 shrink-0 text-cyan-300" />}
                        <span className="truncate">{clip.display_filename}</span>
                      </div>
                      <div className="text-[11px] text-slate-400">
                        {formatSeconds(clip.duration_seconds)}, {formatBytes(clip.size_bytes)}, {selected ? "being studied" : "click to study"}
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={() => remove(clip.recording_id)}
                      aria-label={`Delete ${clip.display_filename}`}
                      className="rounded-full p-2 text-slate-400 transition hover:bg-rose-400/10 hover:text-rose-300"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ul>
        )}
      </div>
    </div>
  );
};

/** Microphone capture with a live level meter, a countdown and a review step. */
const Recorder = ({ disabled, onUse, onError }: { disabled: boolean; onUse: (take: Take) => void; onError: (message: string | null) => void }) => {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [take, setTake] = useState<Take | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const cleanup = useRef<() => void>(() => undefined);
  const supported = typeof window !== "undefined" && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined";

  useEffect(() => () => cleanup.current(), []);
  useEffect(() => () => {
    if (take) URL.revokeObjectURL(take.url);
  }, [take]);

  const stop = useCallback(() => {
    if (recorder.current?.state === "recording") recorder.current.stop();
  }, []);

  const start = async () => {
    onError(null);
    setTake(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch {
      onError("Microphone access was blocked. Allow it in the browser's address bar and try again.");
      return;
    }

    const chunks: Blob[] = [];
    const media = new MediaRecorder(stream);
    recorder.current = media;

    // Level meter.
    const Context = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const context = new Context();
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(stream).connect(analyser);
    const buffer = new Float32Array(analyser.fftSize);
    const began = performance.now();
    let frame = 0;
    const tick = () => {
      analyser.getFloatTimeDomainData(buffer);
      let peak = 0;
      for (const value of buffer) peak = Math.max(peak, Math.abs(value));
      setLevel(peak);
      const seconds = (performance.now() - began) / 1000;
      setElapsed(seconds);
      if (seconds >= MAX_RECORDING_SECONDS) stop();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);

    cleanup.current = () => {
      cancelAnimationFrame(frame);
      stream.getTracks().forEach((track) => track.stop());
      context.close().catch(() => undefined);
      if (media.state === "recording") media.stop();
    };

    media.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    media.onstop = () => {
      const seconds = (performance.now() - began) / 1000;
      cleanup.current();
      cleanup.current = () => undefined;
      setRecording(false);
      setLevel(0);
      const blob = new Blob(chunks, { type: media.mimeType || "audio/webm" });
      if (seconds < MIN_SECONDS || blob.size === 0) {
        onError("That take was too short. Hold the button a little longer.");
        return;
      }
      setTake({ blob, url: URL.createObjectURL(blob), seconds });
    };

    media.start();
    setElapsed(0);
    setRecording(true);
  };

  const remaining = Math.max(0, MAX_RECORDING_SECONDS - elapsed);

  return (
    <div className="df-glass flex flex-col items-center justify-center gap-3 p-4 text-center">
      <div className="relative grid h-14 w-14 place-items-center">
        {recording && (
          <motion.span
            className="absolute inset-0 rounded-2xl bg-rose-400/30"
            animate={{ scale: 1 + Math.min(level * 3, 1.2), opacity: 0.35 + Math.min(level * 2, 0.5) }}
            transition={{ duration: 0.08 }}
            aria-hidden
          />
        )}
        <div className="relative grid h-14 w-14 place-items-center rounded-2xl bg-rose-400/10 ring-1 ring-rose-300/30">
          <Mic className="h-7 w-7 text-rose-200" />
        </div>
      </div>
      <div>
        <div className="font-display text-lg font-semibold text-white">Record your voice</div>
        <p className="mt-1 text-sm text-slate-400">
          {supported
            ? recording
              ? `Recording ${elapsed.toFixed(1)}s (stops by itself in ${remaining.toFixed(0)}s)`
              : `Say a sentence or two, up to ${MAX_RECORDING_SECONDS}s. Then see if the detectors believe you're real.`
            : "This browser cannot record audio. Upload a file instead."}
        </p>
      </div>

      {recording && (
        <div className="h-1.5 w-full max-w-[240px] overflow-hidden rounded-full bg-white/10" aria-hidden>
          <motion.div className="h-full df-accent-bar" animate={{ width: `${Math.min(100, level * 140)}%` }} transition={{ duration: 0.05 }} />
        </div>
      )}

      {!take && (
        <PrimaryButton onClick={recording ? stop : start} disabled={!supported || disabled}>
          {recording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
          {recording ? "Stop" : "Start recording"}
        </PrimaryButton>
      )}

      {take && (
        <div className="w-full space-y-2">
          <audio src={take.url} controls className="w-full" aria-label="Your recording" />
          <div className="flex flex-wrap justify-center gap-2">
            <PrimaryButton
              onClick={() => {
                onUse(take);
                setTake(null);
              }}
              disabled={disabled}
            >
              <Send className="h-4 w-4" /> Use this take ({take.seconds.toFixed(1)}s)
            </PrimaryButton>
            <button
              type="button"
              onClick={() => setTake(null)}
              className="inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold text-white ring-1 ring-white/20 transition hover:bg-white/10"
            >
              <RotateCcw className="h-4 w-4" /> Record again
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
