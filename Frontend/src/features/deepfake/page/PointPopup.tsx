import { motion } from "motion/react";
import { MousePointerClick, Pause, Play } from "lucide-react";
import type { EmbeddingRecording, RecordingInfo } from "../types";
import { audioUrlFor, formatBytes, formatSeconds } from "./api";
import { preview, usePreviewUrl } from "./audio";
import { leanWords, formatScore } from "./palette";
import { usePalette } from "./theme";
import { VerdictChip } from "./ui";

interface PointPopupProps {
  point: EmbeddingRecording;
  info: RecordingInfo | undefined;
  threshold: number;
  selected: boolean;
  /** Position inside the map container, in px. */
  left: number;
  top: number;
  /** Which side of the point the card opens on, so the spring grows out of it. */
  originX: "left" | "right";
  onSelect: () => void;
  onEnter: () => void;
  onLeave: () => void;
}

/**
 * The Datapoint Editor, reborn as a hover card: everything about one point,
 * opening out of the point itself on a spring, with a quick preview and a
 * button to make it the clip under study.
 */
export const PointPopup = ({ point, info, threshold, selected, left, top, originX, onSelect, onEnter, onLeave }: PointPopupProps) => {
  const { scoreColor, REAL } = usePalette();
  const url = audioUrlFor(point.recording_id);
  const playing = usePreviewUrl() === url;
  const colour = scoreColor(point.spoof_probability);
  const spoof = point.decision === "spoof";

  return (
    <motion.div
      role="dialog"
      aria-label={`Details for ${point.display_filename}`}
      initial={{ opacity: 0, scale: 0.6, filter: "blur(10px)", y: 10 }}
      animate={{ opacity: 1, scale: 1, filter: "blur(0px)", y: 0 }}
      exit={{ opacity: 0, scale: 0.85, filter: "blur(6px)", transition: { duration: 0.15 } }}
      transition={{ type: "spring", stiffness: 420, damping: 28 }}
      style={{ left, top, transformOrigin: `${originX === "left" ? "0%" : "100%"} 50%` }}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      className="df-popup-border absolute z-30 w-[270px] rounded-2xl"
    >
      <div className="df-panel rounded-2xl p-4 shadow-2xl backdrop-blur-xl">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: colour, }} />
          <span className="truncate font-mono text-sm font-semibold text-white">{point.display_filename}</span>
        </div>

        <div className="mt-3 flex items-center justify-between">
          <VerdictChip spoof={spoof} size="sm" />
          <span className="font-mono text-lg font-semibold tabular-nums" style={{ color: colour }}>
            {formatScore(point.spoof_probability)}
          </span>
        </div>

        {/* score bar with the threshold notch */}
        <div className="relative mt-2 h-2 overflow-hidden rounded-full bg-white/10">
          <motion.div
            className="h-full rounded-full"
            style={{ background: `linear-gradient(90deg, ${REAL}, ${colour})` }}
            initial={{ width: 0 }}
            animate={{ width: `${point.spoof_probability * 100}%` }}
            transition={{ type: "spring", stiffness: 120, damping: 18, delay: 0.05 }}
          />
          <div className="df-marker absolute inset-y-0 w-0.5" style={{ left: `${threshold * 100}%` }} />
        </div>
        <p className="mt-1.5 text-[11px] text-slate-400">
          The detector {leanWords(point.spoof_probability, threshold)} {spoof ? "synthetic" : "real"}.
        </p>

        <dl className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
          <div>
            <dt className="text-slate-500">length</dt>
            <dd className="font-mono text-slate-200">{formatSeconds(info?.duration_seconds)}</dd>
          </div>
          <div>
            <dt className="text-slate-500">size</dt>
            <dd className="font-mono text-slate-200">{info ? formatBytes(info.size_bytes) : "n/a"}</dd>
          </div>
          <div>
            <dt className="text-slate-500">format</dt>
            <dd className="font-mono text-slate-200">{info?.extension ?? "n/a"}</dd>
          </div>
        </dl>

        <div className="mt-3 flex gap-2">
          <button
            type="button"
            onClick={() => preview.toggle(url)}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-white/10 px-3 py-2 text-xs font-semibold text-white transition hover:bg-white/20"
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {playing ? "Stop" : "Preview"}
          </button>
          <button
            type="button"
            onClick={onSelect}
            disabled={selected}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg df-primary-btn px-3 py-2 text-xs font-semibold transition hover:brightness-110 disabled:opacity-60"
          >
            <MousePointerClick className="h-3.5 w-3.5" />
            {selected ? "Selected" : "Study this"}
          </button>
        </div>
      </div>
    </motion.div>
  );
};
