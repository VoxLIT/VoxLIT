import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Pause, Play } from "lucide-react";
import { preview, safePlay, useWaveform } from "./audio";
import { usePalette } from "./theme";

const BARS = 96;

/**
 * A player that shows the clip's real waveform: bars grow in once it is
 * decoded, the played part lights up in the page gradient, and clicking (or
 * arrow keys) seeks.
 */
export const WaveformPlayer = ({ url, label }: { url: string; label: string }) => {
  const { REAL, MID, FAKE, ink } = usePalette();
  const audioRef = useRef<HTMLAudioElement>(null);
  const waveform = useWaveform(url, BARS);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setTime(0);
    setDuration(0);
  }, [url]);

  // Follow playback smoothly rather than at timeupdate's ~4 Hz.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      if (audioRef.current) setTime(audioRef.current.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const total = duration || waveform?.duration || 0;
  const progress = total > 0 ? Math.min(1, time / total) : 0;

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      preview.stop();
      safePlay(audio);
    } else {
      audio.pause();
    }
  };

  const seek = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio || !total) return;
    const clamped = Math.min(Math.max(seconds, 0), total);
    audio.currentTime = clamped;
    setTime(clamped);
  };

  const peaks = waveform?.peaks ?? Array.from({ length: BARS }, () => 0.06);

  return (
    <div className="flex items-center gap-3">
      <motion.button
        type="button"
        onClick={toggle}
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        aria-label={playing ? `Pause ${label}` : `Play ${label}`}
        className="relative grid h-12 w-12 shrink-0 place-items-center df-play rounded-full"
      >
        {playing && <span className="df-ripple absolute inset-0 rounded-full border-2 border-cyan-300" aria-hidden />}
        {playing ? <Pause className="h-5 w-5" /> : <Play className="ml-0.5 h-5 w-5" />}
      </motion.button>

      <div className="min-w-0 flex-1">
        <div
          role="slider"
          tabIndex={0}
          aria-label={`Seek within ${label}`}
          aria-valuemin={0}
          aria-valuemax={Number(total.toFixed(2))}
          aria-valuenow={Number(time.toFixed(2))}
          aria-valuetext={`${time.toFixed(1)} of ${total.toFixed(1)} seconds`}
          onKeyDown={(event) => {
            const step = event.shiftKey ? 1 : 0.25;
            if (event.key === "ArrowRight") seek(time + step);
            else if (event.key === "ArrowLeft") seek(time - step);
            else if (event.key === " " || event.key === "Enter") toggle();
            else return;
            event.preventDefault();
          }}
          onClick={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            seek(((event.clientX - box.left) / box.width) * total);
          }}
          className="relative h-14 cursor-pointer rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
        >
          <svg viewBox={`0 0 ${BARS * 4} 56`} preserveAspectRatio="none" className="h-full w-full" aria-hidden>
            <defs>
              <linearGradient id="df-wave-played" x1="0" x2="1">
                <stop offset="0%" stopColor={REAL} />
                <stop offset="50%" stopColor={MID} />
                <stop offset="100%" stopColor={FAKE} />
              </linearGradient>
              <clipPath id={`df-wave-clip-${label.replace(/\W/g, "")}`}>
                <rect x={0} y={0} width={progress * BARS * 4} height={56} />
              </clipPath>
            </defs>
            {[false, true].map((played) => (
              <g
                key={String(played)}
                clipPath={played ? `url(#df-wave-clip-${label.replace(/\W/g, "")})` : undefined}
              >
                {peaks.map((peak, index) => {
                  const height = Math.max(2, peak * 50);
                  return (
                    <motion.rect
                      key={index}
                      x={index * 4 + 0.6}
                      width={2.6}
                      rx={1.3}
                      initial={{ height: 2, y: 27 }}
                      animate={{ height, y: 28 - height / 2 }}
                      transition={{ delay: waveform ? index * 0.006 : 0, type: "spring", stiffness: 160, damping: 18 }}
                      fill={played ? "url(#df-wave-played)" : ink(0.18)}
                    />
                  );
                })}
              </g>
            ))}
          </svg>
          {/* playhead */}
          <div
            className="pointer-events-none absolute inset-y-0 df-playhead w-0.5 rounded-full"
            style={{ left: `${progress * 100}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400">
          <span>{time.toFixed(1)}s</span>
          <span>{total ? `${total.toFixed(1)}s` : "0.0s"}</span>
        </div>
      </div>

      <audio
        ref={audioRef}
        src={url}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)}
        onTimeUpdate={(event) => {
          if (event.currentTarget.paused) return;
          setTime(event.currentTarget.currentTime);
        }}
      />
    </div>
  );
};
