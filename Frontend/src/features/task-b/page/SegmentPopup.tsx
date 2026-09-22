import { motion } from "motion/react";
import { MousePointerClick, Play } from "lucide-react";
import type { DiarizationSegment } from "../types";
import { confidenceWords, formatClock } from "./segmentWords";

interface SegmentPopupProps {
  segment: DiarizationSegment;
  colour: string;
  selected: boolean;
  /** Position inside the frame that holds the view, in px. */
  left: number;
  top: number;
  /** Which side of the point the card opens on, so the spring grows out of it. */
  originX: "left" | "right";
  onPlay: () => void;
  onSelect: () => void;
  onEnter: () => void;
  onLeave: () => void;
}

export const POPUP_WIDTH = 240;
export const POPUP_HEIGHT = 170;

/**
 * Everything about one segment, opening out of the point on a spring: who,
 * when, and how sure the model was — in words. The raw score stays in the
 * technical details.
 */
export const SegmentPopup = ({
  segment,
  colour,
  selected,
  left,
  top,
  originX,
  onPlay,
  onSelect,
  onEnter,
  onLeave,
}: SegmentPopupProps) => (
  <motion.div
    role="dialog"
    aria-label={`Details for ${segment.speaker}, ${formatClock(segment.start)}`}
    initial={{ opacity: 0, scale: 0.6, filter: "blur(10px)", y: 10 }}
    animate={{ opacity: 1, scale: 1, filter: "blur(0px)", y: 0 }}
    exit={{ opacity: 0, scale: 0.85, filter: "blur(6px)", transition: { duration: 0.15 } }}
    transition={{ type: "spring", stiffness: 420, damping: 28 }}
    style={{ left, top, width: POPUP_WIDTH, transformOrigin: `${originX === "left" ? "0%" : "100%"} 50%` }}
    onMouseEnter={onEnter}
    onMouseLeave={onLeave}
    className="dz-popup-border absolute z-30 rounded-2xl"
  >
    <div className="dz-panel rounded-2xl p-4 shadow-2xl backdrop-blur-xl">
      <div className="flex items-center gap-2">
        <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: colour }} />
        <span className="truncate text-sm font-semibold text-white">{segment.speaker}</span>
      </div>
      <div className="mt-2 font-mono text-sm text-slate-200">
        {formatClock(segment.start)} – {formatClock(segment.end)}
      </div>
      <div className="mt-1 text-xs text-slate-400">The model was {confidenceWords(segment.confidence_bucket)}.</div>

      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={onPlay}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-white/10 px-3 py-2 text-xs font-semibold text-white transition hover:bg-white/20"
        >
          <Play className="h-3.5 w-3.5" /> Play
        </button>
        <button
          type="button"
          onClick={onSelect}
          disabled={selected}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-lg dz-primary-btn px-3 py-2 text-xs font-semibold transition hover:brightness-110 disabled:opacity-60"
        >
          <MousePointerClick className="h-3.5 w-3.5" />
          {selected ? "Selected" : "Select"}
        </button>
      </div>
    </div>
  </motion.div>
);
