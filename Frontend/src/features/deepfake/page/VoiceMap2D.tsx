import { type PointerEvent as ReactPointerEvent, type SyntheticEvent, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import type { EmbeddingRecording } from "../types";
import { declutter, nearest, normalise } from "./mapGeometry";
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
  /** Nudge overlapping points apart so every clip can be seen. */
  spread?: boolean;
}

/** Gridlines per axis: a quiet frame of reference, left unlabelled because
 *  the projected axes have no units. */
const GRID = 10;
/** Point radii in viewBox units. Small, rimmed discs keep crowded clusters
 *  countable, as in published embedding plots; hover and selection grow them. */
const RADIUS = { rest: 4.5, hovered: 7.5, selected: 10 };
/** Hit area: a little larger than a resting point, but small enough that
 *  neighbours in a dense cluster do not steal each other's hover. */
const HIT = 7;
const MAX_ZOOM = 40;

interface View {
  k: number;
  tx: number;
  ty: number;
}
const IDENTITY: View = { k: 1, tx: 0, ty: 0 };

/** What a screen reader hears for a point: the clip and the detector's reading — never a dataset label. */
const pointLabel = (recording: EmbeddingRecording) =>
  `${recording.display_filename}${recording.uploaded ? " (your clip)" : ""}, ` +
  `detector score ${Math.round(recording.spoof_probability * 100)} out of 100 synthetic`;

/**
 * The 2D voice map: each recording a small rimmed point, coloured by the
 * detector's score. The selection is shown by SIZE and ripples — never by a
 * different colour, because the colour is the reading. Points fly in from the
 * centre and glide when the projection changes, then hold still so each one
 * can be found again.
 */
export const VoiceMap2D = ({ recordings, coordinates, selectedId, hoveredId, onHover, onSelect, spread = true }: VoiceMap2DProps) => {
  const { INK, scoreColor, ink } = usePalette();
  const projected = useMemo(() => {
    const unit = normalise(coordinates, 2);
    // One scale for both axes, as large as the frame allows.
    const spanX = Math.max(1e-6, ...unit.map(([x]) => Math.abs(x)));
    const spanY = Math.max(1e-6, ...unit.map(([, y]) => Math.abs(y)));
    const scale = Math.min((W / 2 - PAD) / spanX, (H / 2 - PAD) / spanY);
    return unit.map(([x, y]) => [W / 2 + x * scale, H / 2 - y * scale]);
  }, [coordinates]);
  // Drawn positions: overlaps removed (a rim's width of air between discs).
  const points = useMemo(
    () => {
      if (!spread) return projected;
      const gap = RADIUS.rest * 2 + 2.5;
      let spaced = declutter(projected, gap);
      // Opening a cluster can push its rim past the frame: shrink the whole
      // map uniformly to fit, then open it again, until it settles inside.
      for (let round = 0; round < 4; round += 1) {
        const xs = spaced.map(([x]) => Math.abs(x - W / 2));
        const ys = spaced.map(([, y]) => Math.abs(y - H / 2));
        const fit = Math.min(1, (W / 2 - PAD) / Math.max(1e-6, ...xs), (H / 2 - PAD) / Math.max(1e-6, ...ys));
        if (fit > 0.995) break;
        spaced = declutter(
          spaced.map(([x, y]) => [W / 2 + (x - W / 2) * fit, H / 2 + (y - H / 2) * fit]),
          gap,
        );
      }
      return spaced;
    },
    [projected, spread],
  );

  // Hover and keyboard focus open the same detail card, anchored on the point.
  const announce = (event: SyntheticEvent<SVGCircleElement>, recordingId: string) => {
    const box = event.currentTarget.getBoundingClientRect();
    onHover({ recordingId, clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 });
  };

  const selectedIndex = recordings.findIndex((recording) => recording.recording_id === selectedId);
  const neighbours = useMemo(
    // Neighbours come from the projection itself, not the nudged display.
    () => (selectedIndex >= 0 ? nearest(projected, selectedIndex, 5) : []),
    [projected, selectedIndex],
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

  // Scroll to zoom at the cursor, drag the background to pan. Points and their
  // rings are counter-scaled, so zooming spreads a cluster apart instead of
  // just magnifying the overlap.
  const svg = useRef<SVGSVGElement>(null);
  const [view, setView] = useState<View>(IDENTITY);
  const drag = useRef<{ x: number; y: number; view: View } | null>(null);
  // A new projection starts from the whole map.
  useEffect(() => setView(IDENTITY), [coordinates]);

  /** Client pixels → viewBox units. */
  const toViewBox = (clientX: number, clientY: number) => {
    const matrix = svg.current?.getScreenCTM();
    if (!matrix) return null;
    return new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse());
  };

  useEffect(() => {
    const element = svg.current;
    if (!element) return;
    // Registered natively: React's wheel listener is passive and cannot stop the page scrolling.
    const wheel = (event: WheelEvent) => {
      const matrix = element.getScreenCTM();
      if (!matrix) return;
      event.preventDefault();
      const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
      setView((previous) => {
        const k = Math.min(MAX_ZOOM, Math.max(1, previous.k * Math.exp(-event.deltaY * 0.0015)));
        if (k === 1) return IDENTITY;
        const ratio = k / previous.k;
        return { k, tx: at.x - (at.x - previous.tx) * ratio, ty: at.y - (at.y - previous.ty) * ratio };
      });
    };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, []);

  const startPan = (event: ReactPointerEvent<SVGSVGElement>) => {
    if ((event.target as Element).getAttribute("role") === "button" || view.k === 1) return;
    drag.current = { x: event.clientX, y: event.clientY, view };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const pan = (event: ReactPointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    if (!start) return;
    const from = toViewBox(start.x, start.y);
    const to = toViewBox(event.clientX, event.clientY);
    if (!from || !to) return;
    setView({ ...start.view, tx: start.view.tx + to.x - from.x, ty: start.view.ty + to.y - from.y });
  };
  const endPan = () => {
    drag.current = null;
  };
  const zoomed = view.k > 1.001;

  return (
    <div className="relative h-full w-full">
    <svg
      ref={svg}
      viewBox={`0 0 ${W} ${H}`}
      className={`h-full w-full touch-none ${zoomed ? "cursor-grab active:cursor-grabbing" : ""}`}
      role="group"
      aria-label="Voice map: one point per recording. Tab to a point, Enter or Space to select it. Scroll to zoom, drag to pan."
      onPointerDown={startPan}
      onPointerMove={pan}
      onPointerUp={endPan}
      onPointerCancel={endPan}
      onDoubleClick={(event) => {
        if ((event.target as Element).getAttribute("role") !== "button") setView(IDENTITY);
      }}
    >
      {/* plot frame and a light grid */}
      <rect x={PAD / 2} y={PAD / 2} width={W - PAD} height={H - PAD} rx={6} fill="none" stroke={ink(0.12)} />
      {Array.from({ length: GRID - 1 }, (_, step) => {
        const fx = PAD / 2 + ((W - PAD) * (step + 1)) / GRID;
        const fy = PAD / 2 + ((H - PAD) * (step + 1)) / GRID;
        return (
          <g key={step} stroke={ink(step === GRID / 2 - 1 ? 0.1 : 0.05)} strokeWidth={1}>
            <line x1={fx} x2={fx} y1={PAD / 2} y2={H - PAD / 2} />
            <line x1={PAD / 2} x2={W - PAD / 2} y1={fy} y2={fy} />
          </g>
        );
      })}

      <defs>
        <clipPath id="df-map-clip">
          <rect x={PAD / 2} y={PAD / 2} width={W - PAD} height={H - PAD} rx={6} />
        </clipPath>
      </defs>
      <g clipPath="url(#df-map-clip)">
      <g transform={`translate(${view.tx} ${view.ty}) scale(${view.k})`}>
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
            strokeDasharray={`${3 / view.k} ${5 / view.k}`}
            vectorEffect="non-scaling-stroke"
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
        const radius = selected ? RADIUS.selected : hovered ? RADIUS.hovered : RADIUS.rest;

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
            <g transform={`scale(${1 / view.k})`}>
              {selected && (
                <>
                  <circle r={radius} fill="none" stroke={colour} strokeWidth={2} className="df-ripple" />
                  <circle r={radius} fill="none" stroke={colour} strokeWidth={2} className="df-ripple df-ripple-delay" />
                  {/* a white gap and an ink ring lift the point off whatever lies beneath */}
                  <circle r={radius + 4.5} fill="#ffffff" stroke={INK} strokeWidth={2.2} style={{ filter: "drop-shadow(0 1px 4px rgba(15,23,42,0.35))" }} />
                </>
              )}
              {recording.uploaded && (
                <>
                  <circle r={radius + 4} fill="none" stroke={INK} strokeWidth={1.5} strokeDasharray="3 3" data-testid="user-clip-marker" />
                  <text y={-(radius + 10)} textAnchor="middle" className="fill-white text-[13px] font-semibold" style={{ pointerEvents: "none" }}>
                    you
                  </text>
                </>
              )}
              <motion.circle
                animate={{ r: radius }}
                transition={{ type: "spring", stiffness: 320, damping: 16 }}
                fill={colour}
                fillOpacity={selected || hovered ? 1 : 0.9}
                stroke={selected || hovered ? INK : "#ffffff"}
                strokeWidth={selected || hovered ? 1.6 : 1.1}
              />
              {/* generous invisible hit area — also the keyboard and screen-reader target */}
              <circle
                r={HIT}
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
      </g>
      </g>
    </svg>
    {zoomed && (
      <button
        type="button"
        onClick={() => setView(IDENTITY)}
        className="df-overlay-chip absolute right-3 top-3 rounded-full px-3 py-1 text-xs font-semibold text-slate-200 backdrop-blur"
      >
        Reset zoom ({view.k.toFixed(1)}x)
      </button>
    )}
    </div>
  );
};
