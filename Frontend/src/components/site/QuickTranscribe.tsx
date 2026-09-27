import { useCallback, useEffect, useRef, useState } from "react";
import { FileRejection, useDropzone } from "react-dropzone";
import WaveSurfer from "wavesurfer.js";
import TimelinePlugin from "wavesurfer.js/dist/plugins/timeline.esm.js";
import HoverPlugin from "wavesurfer.js/dist/plugins/hover.esm.js";
import {
  AlertCircle,
  ArrowDown,
  AudioLines,
  Check,
  Copy,
  Loader2,
  Mic,
  Pause,
  Play,
  RotateCcw,
  Square,
  UploadCloud,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { recordingToWavFile } from "./recordToWav";
import { createPeakRmsRenderer } from "./peakRmsRenderer";
import { clearClip, loadClip, saveClip, saveTranscript } from "./clipStore";
import { pillPath } from "./canvasShapes";

const ACCEPT = { "audio/*": [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"] };
const MAX_BYTES = 25 * 1024 * 1024;
const MAX_SECONDS = 120;
const WAVE_HEIGHT = 168;
const ANALYSIS_RATE = 16000;

type TranscriptState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; text: string; seconds: number }
  | { status: "error"; message: string };

const formatTime = (s: number) => {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec.toFixed(2).padStart(5, "0")}`;
};

/** Live level bars drawn from the microphone while recording. */
const LiveMeter = ({ stream }: { stream: MediaStream }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx2d = canvas.getContext("2d");
    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    const data = new Float32Array(analyser.fftSize);
    const history: number[] = [];
    let raf = 0;

    const draw = () => {
      analyser.getFloatTimeDomainData(data);
      let peak = 0;
      for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
      history.push(Math.min(1, peak * 1.6));

      const dpr = window.devicePixelRatio;
      const w = canvas.clientWidth * dpr;
      const h = canvas.clientHeight * dpr;
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      const bar = 2 * dpr;
      const gap = 2 * dpr;
      const maxBars = Math.floor(w / (bar + gap));
      while (history.length > maxBars) history.shift();

      if (ctx2d) {
        ctx2d.clearRect(0, 0, w, h);
        history.forEach((v, i) => {
          const bh = Math.max(2 * dpr, v * h * 0.9);
          const x = w - (history.length - i) * (bar + gap);
          ctx2d.fillStyle = i === history.length - 1 ? "#F06638" : "#2E8CFF";
          pillPath(ctx2d, x, (h - bh) / 2, bar, bh, bar / 2);
          ctx2d.fill();
        });
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      audioCtx.close();
    };
  }, [stream]);

  return <canvas ref={canvasRef} className="block h-8 w-full" />;
};

/** Home-page "try it" panel: upload or record a clip, see it immediately,
 *  get a Whisper Small transcript from the backend, then pick a workbench. */
const QuickTranscribe = ({ onContinue }: { onContinue: () => void }) => {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [transcript, setTranscript] = useState<TranscriptState>({ status: "idle" });
  const [notice, setNotice] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [recStream, setRecStream] = useState<MediaStream | null>(null);
  const [recSeconds, setRecSeconds] = useState(0);
  const [converting, setConverting] = useState(false);

  const waveRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);

  const transcribe = useCallback(async (f: File) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setTranscript({ status: "loading" });
    const started = performance.now();
    try {
      const body = new FormData();
      body.append("file", f);
      const res = await fetch(`${API_BASE}/quick-transcribe`, {
        method: "POST",
        body,
        credentials: "include",
        signal: controller.signal,
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.detail || `Request failed (${res.status})`);
      const done = { text: json.transcript || "", seconds: (performance.now() - started) / 1000 };
      setTranscript({ status: "done", ...done });
      saveTranscript(done);
    } catch (e) {
      if (controller.signal.aborted) return;
      const message =
        e instanceof TypeError
          ? "Could not reach the VoxLIT backend. Is it running?"
          : e instanceof Error
            ? e.message
            : "Transcription failed.";
      setTranscript({ status: "error", message });
    }
  }, []);

  const showFile = useCallback((f: File) => {
    setNotice(null);
    setDuration(null);
    setCurrentTime(0);
    setPlaying(false);
    setCopied(false);
    setFile(f);
  }, []);

  // Set once the user picks a clip, so a late restore never overwrites it
  const userPickedRef = useRef(false);

  const loadFile = useCallback(
    (f: File) => {
      userPickedRef.current = true;
      showFile(f);
      saveClip(f);
      transcribe(f);
    },
    [showFile, transcribe]
  );

  // Restore the last clip (and its transcript) after a page refresh
  useEffect(() => {
    let cancelled = false;
    loadClip().then((saved) => {
      if (cancelled || !saved || userPickedRef.current) return;
      const f = new File([saved.blob], saved.name, { type: saved.type });
      showFile(f);
      if (saved.transcript) setTranscript({ status: "done", ...saved.transcript });
      else transcribe(f);
    });
    return () => {
      cancelled = true;
    };
  }, [showFile, transcribe]);

  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      if (rejected.length) {
        setNotice(rejected[0].errors[0]?.message ?? "Unsupported file.");
        return;
      }
      if (accepted[0]) loadFile(accepted[0]);
    },
    [loadFile]
  );

  const busy = !!recStream || converting;
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxSize: MAX_BYTES,
    multiple: false,
    noClick: true,
    noKeyboard: true,
    disabled: busy,
  });

  // ---- Recording -------------------------------------------------------
  const stopRecording = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
  }, []);

  const startRecording = async () => {
    setNotice(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setNotice("Recording isn't supported in this browser.");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setNotice("Microphone access was blocked. Allow it in the browser to record.");
      return;
    }
    const chunks: Blob[] = [];
    const rec = new MediaRecorder(stream);
    recorderRef.current = rec;
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setRecStream(null);
      recorderRef.current = null;
      if (!chunks.length) return;
      setConverting(true);
      try {
        loadFile(await recordingToWavFile(new Blob(chunks, { type: rec.mimeType })));
      } catch {
        setNotice("Could not process the recording. Try again or upload a file.");
      } finally {
        setConverting(false);
      }
    };
    rec.start(250);
    setRecSeconds(0);
    setRecStream(stream);
  };

  // Recording timer + hard stop at the backend's length limit
  useEffect(() => {
    if (!recStream) return;
    const started = performance.now();
    const id = window.setInterval(() => {
      const s = (performance.now() - started) / 1000;
      setRecSeconds(s);
      if (s >= MAX_SECONDS) stopRecording();
    }, 100);
    return () => window.clearInterval(id);
  }, [recStream, stopRecording]);

  // ---- Waveform --------------------------------------------------------
  useEffect(() => {
    if (!file || !waveRef.current || !timelineRef.current) return;
    const url = URL.createObjectURL(file);
    // Clip-wide peak so every canvas chunk shares one amplitude scale
    let globalPeak = 0;
    const ws = WaveSurfer.create({
      container: waveRef.current,
      height: WAVE_HEIGHT,
      waveColor: "#5AA9F5",
      progressColor: "#0B4FB3",
      cursorColor: "#0B4FB3",
      cursorWidth: 2,
      dragToSeek: true,
      // Decode at Whisper's input rate so the view shows what the model hears
      sampleRate: ANALYSIS_RATE,
      renderFunction: createPeakRmsRenderer(() => {
        if (!globalPeak) {
          const data = ws.getDecodedData()?.getChannelData(0);
          if (data) for (let i = 0; i < data.length; i++) globalPeak = Math.max(globalPeak, Math.abs(data[i]));
        }
        return globalPeak;
      }),
      url,
      plugins: [
        TimelinePlugin.create({
          container: timelineRef.current,
          height: 20,
          timeInterval: 0.1,
          primaryLabelInterval: 1,
          secondaryLabelInterval: 0,
          style: { color: "hsl(213 10% 50%)", fontSize: "10px" },
        }),
        HoverPlugin.create({
          lineColor: "rgba(15,23,42,0.45)",
          lineWidth: 1,
          labelBackground: "#0f172a",
          labelColor: "#fff",
          labelSize: "11px",
        }),
      ],
    });
    wsRef.current = ws;

    ws.on("decode", (d) => setDuration(d));
    ws.on("timeupdate", (t) => setCurrentTime(t));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    // Rewind when playback ends so the next play starts from the beginning
    ws.on("finish", () => {
      setPlaying(false);
      ws.setTime(0);
    });

    return () => {
      ws.destroy();
      wsRef.current = null;
      URL.revokeObjectURL(url);
    };
  }, [file]);

  useEffect(
    () => () => {
      requestRef.current?.abort();
      const rec = recorderRef.current;
      if (rec && rec.state !== "inactive") {
        rec.onstop = null;
        rec.stop();
        rec.stream.getTracks().forEach((t) => t.stop());
      }
    },
    []
  );

  const reset = () => {
    requestRef.current?.abort();
    setFile(null);
    setDuration(null);
    setNotice(null);
    setTranscript({ status: "idle" });
    clearClip();
  };

  const copyTranscript = async () => {
    if (transcript.status !== "done") return;
    try {
      await navigator.clipboard.writeText(transcript.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  const iconBtn =
    "flex h-9 w-9 items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40";

  return (
    <div className="space-y-6">
      {/* Composer bar */}
      <div {...getRootProps()} className="mx-auto max-w-2xl">
        <input {...getInputProps()} />
        <div
          className={`flex h-14 items-center gap-2 rounded-full border bg-white pl-5 pr-2 shadow-aws-md transition-all ${
            isDragActive
              ? "border-primary ring-4 ring-primary/15"
              : recStream
                ? "border-red-200 ring-4 ring-red-100"
                : "border-border hover:border-primary/30"
          }`}
        >
          {recStream ? (
            <>
              <span className="relative flex h-2.5 w-2.5 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500 opacity-75" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-red-500" />
              </span>
              <span className="shrink-0 font-mono text-[13px] tabular-nums text-foreground">{formatTime(recSeconds)}</span>
              <div className="min-w-0 flex-1">
                <LiveMeter stream={recStream} />
              </div>
              <button
                type="button"
                onClick={stopRecording}
                title="Stop and transcribe"
                aria-label="Stop and transcribe"
                className={`${iconBtn} bg-red-500 text-white hover:bg-red-600`}
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            </>
          ) : converting ? (
            <div className="flex flex-1 items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Preparing your recording…
            </div>
          ) : (
            <>
              <AudioLines className="h-4 w-4 shrink-0 text-primary" />
              {file ? (
                <span className="min-w-0 flex-1 truncate text-sm text-foreground">
                  {file.name}
                  <span className="ml-2 text-muted-foreground">loaded</span>
                </span>
              ) : (
                <button
                  type="button"
                  onClick={open}
                  className="min-w-0 flex-1 truncate text-left text-sm text-muted-foreground"
                >
                  {isDragActive ? "Drop to transcribe" : "Drop a speech clip, or record one…"}
                </button>
              )}
              {file && (
                <button type="button" onClick={reset} title="Clear" aria-label="Clear clip" className={`${iconBtn} text-muted-foreground hover:bg-slate-100 hover:text-foreground`}>
                  <RotateCcw className="h-4 w-4" />
                </button>
              )}
              <button type="button" onClick={open} title="Upload audio" aria-label="Upload audio" className={`${iconBtn} text-slate-600 hover:bg-slate-100 hover:text-primary`}>
                <UploadCloud className="h-[18px] w-[18px]" />
              </button>
              <button
                type="button"
                onClick={startRecording}
                title="Record from microphone"
                aria-label="Record from microphone"
                className={`${iconBtn} h-10 w-10 bg-primary text-primary-foreground shadow-aws-sm hover:bg-primary-hover`}
              >
                <Mic className="h-[18px] w-[18px]" />
              </button>
            </>
          )}
        </div>
        <div className="mt-2.5 text-center text-xs text-muted-foreground">
          {notice ? (
            <span className="inline-flex items-center gap-1.5 text-red-600">
              <AlertCircle className="h-3.5 w-3.5" /> {notice}
            </span>
          ) : recStream ? (
            "Recording from your microphone · stops automatically at 2:00"
          ) : (
            "WAV, MP3, M4A, FLAC, OGG up to 2 min · transcribed by Whisper Small"
          )}
        </div>
      </div>

      {file && (
        <div className="overflow-hidden rounded-3xl border border-border bg-white text-left shadow-aws-lg animate-in fade-in slide-in-from-bottom-2 duration-500">
        <div className="grid gap-6 p-5 sm:p-6 lg:grid-cols-[1fr_20rem]">
          {/* Signal view */}
          <div className="min-w-0 space-y-3">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => wsRef.current?.playPause()}
                disabled={duration === null}
                aria-label={playing ? "Pause" : "Play"}
                className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-aws-sm transition-transform hover:scale-105 disabled:opacity-40"
              >
                {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 translate-x-px" />}
              </button>
              <div className="font-mono text-[13px] tabular-nums text-foreground">
                {formatTime(currentTime)}
                <span className="text-muted-foreground"> / {duration !== null ? formatTime(duration) : "–:––.––"}</span>
              </div>
            </div>

            <div className="overflow-hidden rounded-xl border border-border bg-white">
              <div className="flex items-center justify-end gap-4 px-3 pt-2.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2 w-3 rounded-sm bg-[#5AA9F5]/40" /> Peak
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2 w-3 rounded-sm bg-[#5AA9F5]" /> RMS
                </span>
              </div>
              {/* Faint amplitude guides at ±50% and the zero line, behind the canvas */}
              <div className="relative px-3">
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-x-3 inset-y-0 [background-image:linear-gradient(to_bottom,transparent_calc(25%-0.5px),#EEF2F7_calc(25%-0.5px),#EEF2F7_calc(25%+0.5px),transparent_calc(25%+0.5px),transparent_calc(50%-0.5px),#E2E8F0_calc(50%-0.5px),#E2E8F0_calc(50%+0.5px),transparent_calc(50%+0.5px),transparent_calc(75%-0.5px),#EEF2F7_calc(75%-0.5px),#EEF2F7_calc(75%+0.5px),transparent_calc(75%+0.5px))]"
                />
                <div ref={waveRef} className="relative cursor-pointer" />
              </div>
              <div ref={timelineRef} className="mx-3 border-t border-border pb-1" />
            </div>
          </div>

          {/* Transcript */}
          <div className="flex min-h-[12rem] flex-col rounded-xl border border-border bg-slate-50/60">
            <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Transcript</div>
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                openai/whisper-small
              </span>
            </div>
            <div className="flex-1 px-4 py-3 text-sm leading-relaxed">
              {transcript.status === "loading" && (
                <div className="space-y-2" aria-live="polite">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> Transcribing… the first run also downloads the model.
                  </div>
                  {[92, 78, 85, 60].map((w, i) => (
                    <div key={i} className="h-3 animate-pulse rounded bg-slate-200" style={{ width: `${w}%` }} />
                  ))}
                </div>
              )}
              {transcript.status === "done" &&
                (transcript.text ? (
                  <p className="text-foreground">“{transcript.text}”</p>
                ) : (
                  <p className="text-muted-foreground">No speech was detected in this clip.</p>
                ))}
              {transcript.status === "error" && (
                <div className="space-y-2 text-xs">
                  <div className="flex items-start gap-1.5 text-red-600">
                    <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {transcript.message}
                  </div>
                  <button
                    type="button"
                    onClick={() => transcribe(file)}
                    className="rounded-full border border-border bg-white px-3 py-1 text-foreground hover:border-primary/40"
                  >
                    Retry
                  </button>
                </div>
              )}
            </div>
            {transcript.status === "done" && (
              <div className="flex items-center justify-between border-t border-border px-4 py-2 text-[11px] text-muted-foreground">
                <span>{transcript.seconds.toFixed(1)} s round trip</span>
                {transcript.text && (
                  <button type="button" onClick={copyTranscript} className="flex items-center gap-1 hover:text-primary">
                    {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                )}
              </div>
            )}
            <button
              type="button"
              onClick={onContinue}
              className="m-3 mt-0 flex items-center justify-center gap-1.5 rounded-lg bg-primary px-4 py-2.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary-hover"
            >
              Choose a task to go deeper
              <ArrowDown className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
        </div>
      )}
    </div>
  );
};

export default QuickTranscribe;
