import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Flame } from "lucide-react";
import type { DeepfakeSaliency } from "../types";
import { readSaliencyVerdict } from "../SaliencyPanel";
import { audioUrlFor, errorMessage, postDeepfake } from "./api";
import { safePlay, useWaveform } from "./audio";
import { Disclosure, ErrorNote, FeatureImage, Finding, PrimaryButton } from "./ui";
import { usePalette } from "./theme";

const W = 600;
const H = 150;

/**
 * Feature 3 for everyone: "where did it listen?" — the saliency drawn as
 * heat over the real waveform, revealed with a left-to-right sweep. Blue
 * bands mark where the voice is, so heat on silence stands out.
 */
export const ListeningHeatmap = ({ model, recordingId }: { model: string; recordingId: string }) => {
  const { INK, ink, REAL, FAKE, WARN } = usePalette();
  const [result, setResult] = useState<DeepfakeSaliency | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [position, setPosition] = useState(0);
  const audio = useRef<HTMLAudioElement>(null);
  const url = audioUrlFor(recordingId);
  const waveform = useWaveform(result ? url : undefined, 200);

  const latestRequest = useRef(0);

  useEffect(() => {
    latestRequest.current += 1;
    setResult(null);
    setError(null);
    setPosition(0);
    setRunning(false);
  }, [recordingId, model]);

  const run = async () => {
    const request = ++latestRequest.current;
    setRunning(true);
    setError(null);
    try {
      const payload = await postDeepfake<DeepfakeSaliency>("saliency", { model, recording_id: recordingId });
      if (request === latestRequest.current) setResult(payload);
    } catch (caught) {
      if (request === latestRequest.current) setError(errorMessage(caught, "The heatmap failed."));
    } finally {
      if (request === latestRequest.current) setRunning(false);
    }
  };

  const verdict = result ? readSaliencyVerdict(result) : null;
  const duration = result?.total_duration || 1;
  const x = (seconds: number) => (seconds / duration) * W;

  const seek = (seconds: number) => {
    const clamped = Math.min(Math.max(seconds, 0), duration);
    setPosition(clamped);
    if (audio.current) {
      audio.current.currentTime = clamped;
      safePlay(audio.current);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Flame className="h-4 w-4 text-rose-300" />
        <h4 className="font-display text-base font-semibold text-white">Where did it listen?</h4>
      </div>

      {!result && (
        <>
          <FeatureImage
            src="/deepfake/saliency.png"
            alt="A spectrogram of speech, showing energy over time and pitch"
            caption="A heatmap over the sound shows which moments pushed the detector towards synthetic."
            className="h-32"
          />
          <PrimaryButton onClick={run} busy={running} className="w-full">
            <Flame className="h-4 w-4" /> {running ? "Tracing the attention" : "Show the heatmap"}
          </PrimaryButton>
        </>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}

      {result && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
          <div
            role="slider"
            tabIndex={0}
            aria-label="Seek within the clip"
            aria-valuemin={0}
            aria-valuemax={Number(duration.toFixed(2))}
            aria-valuenow={Number(position.toFixed(2))}
            onKeyDown={(event) => {
              const step = event.shiftKey ? 1 : 0.25;
              if (event.key === "ArrowRight") seek(position + step);
              else if (event.key === "ArrowLeft") seek(position - step);
              else return;
              event.preventDefault();
            }}
            className="overflow-hidden df-well rounded-2xl ring-1 ring-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
          >
            <svg
              viewBox={`0 0 ${W} ${H}`}
              className="block h-auto w-full cursor-pointer"
              role="img"
              aria-label="Attention heatmap over the waveform"
              onClick={(event) => {
                const box = event.currentTarget.getBoundingClientRect();
                seek(((event.clientX - box.left) / box.width) * duration);
              }}
            >
              <defs>
                <linearGradient id="df-heat" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={FAKE} />
                  <stop offset="55%" stopColor={WARN} />
                  <stop offset="100%" stopColor={FAKE} />
                </linearGradient>
                <clipPath id="df-heat-reveal">
                  <motion.rect
                    x={0}
                    y={0}
                    height={H}
                    initial={{ width: 0 }}
                    animate={{ width: W }}
                    transition={{ duration: 1.4, ease: [0.4, 0, 0.2, 1] }}
                  />
                </clipPath>
              </defs>

              <g clipPath="url(#df-heat-reveal)">
                {result.speech_intervals.map(([start, end]) => (
                  <rect key={`s-${start}`} x={x(start)} y={H - 8} width={Math.max(1, x(end) - x(start))} height={8} fill={REAL} opacity={0.8} />
                ))}
                {result.segments.map((segment) => (
                  <rect
                    key={`h-${segment.start_time}`}
                    x={x(segment.start_time)}
                    y={0}
                    width={Math.max(1, x(segment.end_time) - x(segment.start_time))}
                    height={H - 10}
                    fill="url(#df-heat)"
                    opacity={0.05 + segment.saliency * 0.75}
                  />
                ))}
                {waveform && (
                  <path
                    d={waveform.peaks
                      .map((peak, index) => {
                        // The peaks span the whole file but the heat only the
                        // analysed window, so place them by time and drop
                        // whatever falls past the window.
                        const px = x((index / waveform.peaks.length) * waveform.duration);
                        if (px > W) return "";
                        const half = (peak * (H - 14)) / 2.3;
                        const mid = (H - 10) / 2;
                        return `M ${px.toFixed(1)} ${(mid - half).toFixed(1)} L ${px.toFixed(1)} ${(mid + half).toFixed(1)}`;
                      })
                      .join(" ")}
                    stroke={ink(0.8)}
                    strokeWidth={1.2}
                  />
                )}
              </g>
              <line x1={x(position)} x2={x(position)} y1={0} y2={H} stroke={INK} strokeWidth={1.5} />
            </svg>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm df-heat-swatch" /> attention
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-1.5 w-3 rounded-sm bg-cyan-400" /> voice present
            </span>
            <span className="ml-auto italic">click to play from a moment</span>
          </div>
          <audio
            ref={audio}
            src={url}
            preload="none"
            onTimeUpdate={(event) => {
              if (!event.currentTarget.paused) setPosition(event.currentTarget.currentTime);
            }}
          />

          {verdict && <Finding {...verdict} />}

          <Disclosure>
            <p>
              {result.method_label}, taken on the {result.target}. Attribution is normalised within this clip: it
              ranks moments against each other and never compares across clips or models.
            </p>
            <p className="text-xs text-slate-400">
              Analysis capped at {result.max_saliency_seconds}s (the shared saliency service&apos;s cap)
              {result.truncated ? " (this clip was truncated to fit)" : ""}. Voice regions found with an energy
              threshold {result.silence_top_db} dB below the clip&apos;s own peak.
              {result.saliency_in_speech_fraction !== null &&
                ` Share of attention on voice: ${(result.saliency_in_speech_fraction * 100).toFixed(1)}%.`}
              {result.cached ? " Cached." : ""}
            </p>
          </Disclosure>
        </motion.div>
      )}
    </div>
  );
};
