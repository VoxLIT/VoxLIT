import { motion } from "motion/react";
import { usePalette } from "./theme";

const W = 520;
const H = 320;
const LANE_X = 96;
const LANE_W = W - LANE_X - 24;
const LANES = [
  { name: "Speaker A", initial: "A", y: 70 },
  { name: "Speaker B", initial: "B", y: 160 },
  { name: "Speaker C", initial: "C", y: 250 },
];
const BAR_H = 26;

/** The conversation, as fractions of the lane width, in the order it is
 *  spoken. One turn is "unsure" and drawn hatched. */
const TURNS: { lane: number; start: number; end: number; unsure?: boolean }[] = [
  { lane: 0, start: 0.0, end: 0.2 },
  { lane: 1, start: 0.21, end: 0.36 },
  { lane: 0, start: 0.37, end: 0.44 },
  { lane: 2, start: 0.45, end: 0.62 },
  { lane: 1, start: 0.6, end: 0.7, unsure: true },
  { lane: 0, start: 0.71, end: 0.83 },
  { lane: 2, start: 0.84, end: 1.0 },
];

/**
 * The hero picture: three voices, three lanes, and a conversation that builds
 * itself bar by bar while a playhead sweeps across it. Speaker colours come
 * from the active theme, so it reads the same in light and dark.
 */
export const HeroConversation = () => {
  const { SPEAKERS, INK, ink, CANVAS } = usePalette();
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label="Three speakers on a timeline: each voice gets its own colour and lane, and one uncertain turn is hatched."
    >
      <defs>
        {TURNS.filter((turn) => turn.unsure).map((turn, index) => (
          <pattern
            key={index}
            id={`dz-hero-hatch-${turn.lane}`}
            width="6"
            height="6"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width="3" height="6" fill={SPEAKERS[turn.lane]} />
          </pattern>
        ))}
        <clipPath id="dz-hero-track">
          <rect x={LANE_X} y={0} width={LANE_W} height={H} />
        </clipPath>
      </defs>

      {LANES.map((lane, index) => (
        <g key={lane.name}>
          {/* avatar */}
          <motion.g
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 220, damping: 14, delay: 0.1 + index * 0.12 }}
            style={{ transformOrigin: `36px ${lane.y}px` }}
          >
            <circle cx={36} cy={lane.y} r={22} fill={SPEAKERS[index]} fillOpacity={0.18} />
            <circle cx={36} cy={lane.y} r={22} fill="none" stroke={SPEAKERS[index]} strokeWidth={1.5} />
            <text
              x={36}
              y={lane.y + 5}
              textAnchor="middle"
              fontSize={15}
              fontWeight={700}
              fill={SPEAKERS[index]}
              style={{ fontFamily: '"Space Grotesk", Inter, sans-serif' }}
            >
              {lane.initial}
            </text>
          </motion.g>
          {/* lane */}
          <rect x={LANE_X} y={lane.y - BAR_H / 2} width={LANE_W} height={BAR_H} rx={6} fill={ink(0.05)} />
        </g>
      ))}

      {TURNS.map((turn, index) => {
        const colour = SPEAKERS[turn.lane];
        const y = LANES[turn.lane].y - BAR_H / 2;
        const width = (turn.end - turn.start) * LANE_W;
        return (
          <motion.rect
            key={index}
            x={LANE_X + turn.start * LANE_W}
            y={y}
            height={BAR_H}
            rx={6}
            initial={{ width: 0 }}
            animate={{ width }}
            transition={{ duration: 0.45, ease: [0.2, 0.7, 0.3, 1], delay: 0.5 + index * 0.28 }}
            fill={turn.unsure ? `url(#dz-hero-hatch-${turn.lane})` : colour}
            fillOpacity={turn.unsure ? 0.7 : 1}
            stroke={turn.unsure ? colour : "none"}
            strokeWidth={turn.unsure ? 1.5 : 0}
            strokeDasharray={turn.unsure ? "4 3" : undefined}
          />
        );
      })}

      {/* labels */}
      <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.6 }}>
        <text
          x={LANE_X + 0.21 * LANE_W}
          y={LANES[1].y - BAR_H / 2 - 8}
          fontSize={11}
          fill={ink(0.65)}
          style={{ fontFamily: '"JetBrains Mono", ui-monospace, monospace' }}
        >
          00:12 · Speaker B
        </text>
        <text
          x={LANE_X + 0.6 * LANE_W}
          y={LANES[1].y + BAR_H / 2 + 16}
          fontSize={11}
          fill={ink(0.65)}
          style={{ fontFamily: '"JetBrains Mono", ui-monospace, monospace' }}
        >
          00:36 · unsure
        </text>
      </motion.g>

      {/* playhead */}
      <g clipPath="url(#dz-hero-track)">
        <g
          className="dz-sweep"
          style={{ ["--dz-sweep-distance" as string]: `${LANE_W}px` }}
        >
          <line x1={LANE_X} x2={LANE_X} y1={24} y2={H - 24} stroke={INK} strokeWidth={1.5} />
          <circle cx={LANE_X} cy={24} r={4} fill={INK} stroke={CANVAS} strokeWidth={1.5} />
        </g>
      </g>
    </svg>
  );
};
