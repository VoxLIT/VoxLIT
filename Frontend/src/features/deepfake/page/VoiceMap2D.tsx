import { type SyntheticEvent, useMemo } from "react";
import { motion } from "motion/react";
import type { EmbeddingRecording } from "../types";
import { nearest, normalise } from "./mapGeometry";
import { usePalette } from "./theme";

const W = 1000;
const H = 640;
const PAD = 56;

export interface MapPointerEvent {
  recordingId: string;
  /** Client coordinates of the point's centre. */
  clientX: number;
  clientY: number;
}

interface VoiceMap2DProps {
  recordings: EmbeddingRecording[];
  coordinates: number[][];
  selectedId: string;
  hoveredId: string | null;
  onHover: (event: MapPointerEvent | null) => void;
  onSelect: (recordingId: string) => void;
}

// Deterministic background dust, so it does not reshuffle on every render.
const DUST = Array.from({ length: 90 }, (_, index) => {
  const a = Math.sin(index * 12.9898) * 43758.5453;
  const b = Math.sin(index * 78.233) * 12345.678;
  return { x: (a - Math.floor(a)) * W, y: (b - Math.floor(b)) * H, r: 0.6 + ((a * 7) % 1) * 1.2, delay: (index % 9) * 0.45 };
});

/** What a screen reader hears for a point: the clip and the detector's reading — never a dataset label. */
const pointLabel = (recording: EmbeddingRecording) =>
  `${recording.display_filename}${recording.uploaded ? " (your clip)" : ""}, ` +
  `detector score ${Math.round(recording.spoof_probability * 100)} out of 100 synthetic`;

/**
 * The 2D voice map: each recording a glowing point, coloured by the
 * detector's score. The selection is shown by SIZE and ripples — never by a
 * different colour, because the colour is the reading. Points fly in from the
 * centre, glide when the projection changes, and drift gently at rest.
 */
export const VoiceMap2D = ({ recordings, coordinates, selectedId, hoveredId, onHover, onSelect }: VoiceMap2DProps) => {
  const { INK, CANVAS, scoreColor } = usePalette();
  const points = useMemo(() => {
    const unit = normalise(coordinates, 2);
    // One scale for both axes, as large as the frame allows.
    const spanX = Math.max(1e-6, ...unit.map(([x]) => Math.abs(x)));
    const spanY = Math.max(1e-6, ...unit.map(([, y]) => Math.abs(y)));
    const scale = Math.min((W / 2 - PAD) / spanX, (H / 2 - PAD) / spanY);
    return unit.map(([x, y]) => [W / 2 + x * scale, H / 2 - y * scale]);
  }, [coordinates]);

  // Hover and keyboard focus open the same detail card, anchored on the point.
  const announce = (event: SyntheticEvent<SVGCircleElement>, recordingId: string) => {
    const box = event.currentTarget.getBoundingClientRect();
    onHover({ recordingId, clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 });
  };

  const selectedIndex = recordings.findIndex((recording) => recording.recording_id === selectedId);
  const neighbours = useMemo(
    () => (selectedIndex >= 0 ? nearest(points, selectedIndex, 5) : []),
    [points, selectedIndex],
  );

  // Draw the selected and hovered points last so they sit on top.
  const order = useMemo(() => {
    const indices = recordings.map((_, index) => index);
    const rank = (index: number) =>
      recordings[index].recording_id === selectedId
        ? 3
        : recordings[index].recording_id === hoveredId
          ? 2
          : recordings[index].uploaded
            ? 1
            : 0;
    return indices.sort((a, b) => rank(a) - rank(b));
  }, [recordings, selectedId, hoveredId]);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="group" aria-label="Voice map: one point per recording. Tab to a point, Enter or Space to select it.">
      <defs>
        <radialGradient id="df-map-bg" cx="50%" cy="45%" r="65%">
          <stop offset="0%" stopColor="rgba(91,156,246,0.06)" />
          <stop offset="100%" stopColor="rgba(11,15,23,0)" />
        </radialGradient>
        <filter id="df-point-glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect width={W} height={H} fill="url(#df-map-bg)" />
      {DUST.map((dust, index) => (
        <circle
          key={index}
          cx={dust.x}
          cy={dust.y}
          r={dust.r}
          fill="white"
          className="df-twinkle"
          style={{ animationDelay: `${dust.delay}s` }}
        />
      ))}

      {/* the selected point's nearest neighbours on this map */}
      {selectedIndex >= 0 &&
        neighbours.map((index) => (
          <motion.line
            key={`${selectedId}-${index}`}
            x1={points[selectedIndex][0]}
            y1={points[selectedIndex][1]}
            x2={points[index][0]}
            y2={points[index][1]}
            stroke={scoreColor(recordings[index].spoof_probability)}
            strokeWidth={1.2}
            strokeDasharray="3 5"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.7 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          />
        ))}

      {order.map((index) => {
        const recording = recordings[index];
        const [x, y] = points[index] ?? [W / 2, H / 2];
        const colour = scoreColor(recording.spoof_probability);
        const selected = recording.recording_id === selectedId;
        const hovered = recording.recording_id === hoveredId;
        const radius = selected ? 13 : hovered ? 9.5 : 5.5;

        return (
          <motion.g
            key={recording.recording_id}
            data-testid="map-point"
            data-recording-id={recording.recording_id}
            data-selected={selected ? "true" : "false"}
            initial={{ x: W / 2, y: H / 2, opacity: 0, scale: 0 }}
            animate={{ x, y, opacity: 1, scale: 1 }}
            transition={{ type: "spring", stiffness: 70, damping: 14, delay: Math.min(index * 0.006, 1.2) }}
          >
            <g className="df-float" style={{ animationDelay: `${(index % 17) * 0.3}s` }}>
              {selected && (
                <>
                  <circle r={radius} fill="none" stroke={colour} strokeWidth={2} className="df-ripple" />
                  <circle r={radius} fill="none" stroke={colour} strokeWidth={2} className="df-ripple df-ripple-delay" />
                  <circle r={radius + 6} fill="none" stroke={INK} strokeOpacity={0.9} strokeWidth={2} />
                </>
              )}
              {recording.uploaded && (
                <>
                  <circle r={radius + 5} fill="none" stroke={INK} strokeWidth={1.5} strokeDasharray="3 3" data-testid="user-clip-marker" />
                  <text y={-(radius + 10)} textAnchor="middle" className="fill-white text-[13px] font-semibold" style={{ pointerEvents: "none" }}>
                    you
                  </text>
                </>
              )}
              <motion.circle
                animate={{ r: radius }}
                transition={{ type: "spring", stiffness: 320, damping: 16 }}
                fill={colour}
                stroke={selected ? INK : CANVAS}
                strokeWidth={selected ? 2 : 1.6}
                filter={selected || hovered ? "url(#df-point-glow)" : undefined}
              />
              {/* generous invisible hit area — also the keyboard and screen-reader target */}
              <circle
                r={14}
                fill="transparent"
                tabIndex={0}
                role="button"
                aria-label={pointLabel(recording)}
                aria-pressed={selected}
                className="cursor-pointer outline-none focus-visible:stroke-white focus-visible:[stroke-width:2]"
                onMouseEnter={(event) => announce(event, recording.recording_id)}
                onFocus={(event) => announce(event, recording.recording_id)}
                onMouseLeave={() => onHover(null)}
                onBlur={() => onHover(null)}
                onClick={() => onSelect(recording.recording_id)}
                onKeyDown={(event) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  onSelect(recording.recording_id);
                }}
              />
            </g>
          </motion.g>
        );
      })}
    </svg>
  );
};
