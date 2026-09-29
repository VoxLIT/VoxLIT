import { type SyntheticEvent, useMemo } from "react";
import { motion } from "motion/react";
import type { DiarizationSegment, ProjectionPoint } from "../types";
import { segmentLabel } from "./segmentWords";
import { usePalette } from "./theme";

const W = 1000;
const H = 620;
const PAD = 48;

export interface MapPointerEvent {
  segmentId: string;
  /** Client coordinates of the point's centre. */
  clientX: number;
  clientY: number;
}

interface SegmentMap2DProps {
  points: ProjectionPoint[];
  segmentsById: Map<string, DiarizationSegment>;
  speakers: string[];
  selectedId: string | null;
  hoveredId: string | null;
  /** The selected segment's nearest segments by cosine in embedding space. */
  neighbourIds: string[];
  onHover: (event: MapPointerEvent | null) => void;
  onSelect: (segmentId: string) => void;
}

/** Centre the cloud and scale it with ONE factor for both axes, so it keeps
 *  its shape (a per-axis stretch would invent structure). */
function layout(points: ProjectionPoint[]): [number, number][] {
  if (points.length === 0) return [];
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const spanX = Math.max(1e-6, ...xs.map((x) => Math.abs(x - cx)));
  const spanY = Math.max(1e-6, ...ys.map((y) => Math.abs(y - cy)));
  const scale = Math.min((W / 2 - PAD) / spanX, (H / 2 - PAD) / spanY);
  return points.map((p) => [W / 2 + (p.x - cx) * scale, H / 2 - (p.y - cy) * scale]);
}

/**
 * The 2D segment map: one dot per embedded segment, in its speaker's colour.
 * Selection is shown by SIZE and ripples, never by a new colour — colour is
 * speaker identity. Uncertain segments are faded with a dashed outline, the
 * same language as the timeline's hatching.
 */
export const SegmentMap2D = ({
  points,
  segmentsById,
  speakers,
  selectedId,
  hoveredId,
  neighbourIds,
  onHover,
  onSelect,
}: SegmentMap2DProps) => {
  const { INK, CANVAS, speakerColor } = usePalette();
  const xy = useMemo(() => layout(points), [points]);
  const indexById = useMemo(() => new Map(points.map((p, i) => [p.id, i])), [points]);
  const selectedIndex = selectedId !== null ? indexById.get(selectedId) : undefined;

  // Selected and hovered points are drawn last so they sit on top.
  const order = useMemo(() => {
    const rank = (id: string) => (id === selectedId ? 2 : id === hoveredId ? 1 : 0);
    return points.map((_, i) => i).sort((a, b) => rank(points[a].id) - rank(points[b].id));
  }, [points, selectedId, hoveredId]);

  const announce = (event: SyntheticEvent<SVGCircleElement>, segmentId: string) => {
    const box = event.currentTarget.getBoundingClientRect();
    onHover({ segmentId, clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 });
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-full w-full"
      role="group"
      aria-label="Segment map: one dot per segment, coloured by speaker. Tab to a dot, Enter or Space to select it."
    >
      {/* dashed lines to the nearest segments — measured in the model's space, not on this map */}
      {selectedIndex !== undefined &&
        neighbourIds.map((id) => {
          const index = indexById.get(id);
          if (index === undefined) return null;
          return (
            <motion.line
              key={`${selectedId}-${id}`}
              data-testid="neighbour-line"
              x1={xy[selectedIndex][0]}
              y1={xy[selectedIndex][1]}
              x2={xy[index][0]}
              y2={xy[index][1]}
              stroke={speakerColor(speakers, points[index].speaker)}
              strokeWidth={1.4}
              strokeDasharray="4 5"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.8 }}
              transition={{ duration: 0.7, delay: 0.15 }}
            />
          );
        })}

      {order.map((index) => {
        const point = points[index];
        const segment = segmentsById.get(point.id);
        const [x, y] = xy[index];
        const colour = speakerColor(speakers, point.speaker);
        const selected = point.id === selectedId;
        const hovered = point.id === hoveredId;
        const uncertain = segment?.confidence_bucket === "uncertain";
        const radius = selected ? 12 : hovered ? 9 : 5.5;

        return (
          <motion.g
            key={point.id}
            data-testid="map-point"
            data-segment-id={point.id}
            data-selected={selected ? "true" : "false"}
            initial={{ x: W / 2, y: H / 2, opacity: 0, scale: 0 }}
            animate={{ x, y, opacity: 1, scale: 1 }}
            transition={{ type: "spring", stiffness: 70, damping: 14, delay: Math.min(index * 0.006, 1.2) }}
          >
            {selected && (
              <>
                <circle r={radius} fill="none" stroke={colour} strokeWidth={2} className="dz-ripple" />
                <circle r={radius + 5} fill="none" stroke={INK} strokeOpacity={0.9} strokeWidth={2} />
              </>
            )}
            <motion.circle
              data-testid="map-dot"
              // Mount at the target size (the group already flies in); only later size changes spring.
              initial={false}
              animate={{ r: radius }}
              transition={{ type: "spring", stiffness: 320, damping: 16 }}
              fill={colour}
              fillOpacity={uncertain ? 0.45 : 1}
              stroke={uncertain ? colour : CANVAS}
              strokeDasharray={uncertain ? "2 2" : undefined}
              strokeWidth={1.6}
            />
            {/* generous invisible hit area — also the keyboard and screen-reader target */}
            <circle
              r={14}
              fill="transparent"
              tabIndex={0}
              role="button"
              aria-label={segment ? segmentLabel(segment) : point.speaker}
              aria-pressed={selected}
              className="cursor-pointer outline-none"
              onMouseEnter={(event) => announce(event, point.id)}
              onFocus={(event) => announce(event, point.id)}
              onMouseLeave={() => onHover(null)}
              onBlur={() => onHover(null)}
              onClick={() => onSelect(point.id)}
              onKeyDown={(event) => {
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                onSelect(point.id);
              }}
            />
          </motion.g>
        );
      })}
    </svg>
  );
};
