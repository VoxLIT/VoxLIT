import { useEffect, useMemo } from "react";
import { motion, useSpring, useTransform } from "motion/react";
import type { AttackSummary, DetPoint, ScoreBin } from "../types";
import { usePalette } from "./theme";

/** A number that counts to its value. */
export const CountUp = ({ value, format }: { value: number; format: (value: number) => string }) => {
  const spring = useSpring(0, { stiffness: 50, damping: 16 });
  useEffect(() => {
    spring.set(value);
  }, [value, spring]);
  const text = useTransform(spring, format);
  return <motion.span>{text}</motion.span>;
};

const HW = 760;
const HH = 260;
const HP = { top: 16, right: 16, bottom: 36, left: 16 };

/**
 * The two score populations on one axis, with the user's threshold. Bars on
 * the wrong side of the cut glow amber — those are the mistakes.
 */
export const Histogram = ({
  bonafide,
  spoof,
  threshold,
  eerThreshold,
  shippedThreshold,
}: {
  bonafide: ScoreBin[];
  spoof: ScoreBin[];
  threshold: number;
  eerThreshold: number;
  shippedThreshold: number;
}) => {
  const { REAL, MID, FAKE, WARN, INK, ink } = usePalette();
  const plotW = HW - HP.left - HP.right;
  const plotH = HH - HP.top - HP.bottom;
  const peak = Math.max(1, ...bonafide.map((bin) => bin.count), ...spoof.map((bin) => bin.count));
  const x = (score: number) => HP.left + score * plotW;
  // square-root height so a bin of 2 clips is still visible next to one of 100
  const h = (count: number) => (Math.sqrt(count) / Math.sqrt(peak)) * plotH;

  const bars = (bins: ScoreBin[], kind: "real" | "fake") =>
    bins
      .filter((bin) => bin.count > 0)
      .map((bin, index) => {
        const mid = (bin.bin_start + bin.bin_end) / 2;
        const wrong = kind === "real" ? mid >= threshold : mid < threshold;
        const width = Math.max(2, x(bin.bin_end) - x(bin.bin_start) - 3);
        const height = h(bin.count);
        return (
          <motion.rect
            key={`${kind}-${bin.bin_start}`}
            x={x(bin.bin_start) + 1.5}
            y={HP.top + plotH - height}
            width={width}
            height={height}
            rx={4}
            fill={`url(#df-hist-${kind})`}
            stroke={wrong ? WARN : "none"}
            strokeWidth={wrong ? 2 : 0}
            filter={wrong ? "url(#df-hist-glow)" : undefined}
            opacity={kind === "real" ? 0.9 : 0.85}
            style={{ transformBox: "fill-box", originY: 1 }}
            initial={{ scaleY: 0 }}
            whileInView={{ scaleY: 1 }}
            viewport={{ once: true }}
            transition={{ type: "spring", stiffness: 80, damping: 14, delay: index * 0.04 + (kind === "fake" ? 0.3 : 0) }}
          >
            <title>
              {bin.count} {kind === "real" ? "genuine" : "synthetic"} clips scored {bin.bin_start.toFixed(2)}–
              {bin.bin_end.toFixed(2)}
            </title>
          </motion.rect>
        );
      });

  return (
    <svg viewBox={`0 0 ${HW} ${HH}`} className="h-auto w-full" role="img" aria-label="Score distributions for genuine and synthetic clips">
      <defs>
        <linearGradient id="df-hist-real" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={REAL} />
          <stop offset="100%" stopColor={REAL} stopOpacity={0.15} />
        </linearGradient>
        <linearGradient id="df-hist-fake" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={FAKE} />
          <stop offset="100%" stopColor={FAKE} stopOpacity={0.15} />
        </linearGradient>
        <filter id="df-hist-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* the two zones the threshold creates */}
      <motion.rect
        y={HP.top}
        height={plotH}
        x={HP.left}
        animate={{ width: Math.max(0, x(threshold) - HP.left) }}
        transition={{ type: "spring", stiffness: 200, damping: 26 }}
        fill={REAL}
        opacity={0.06}
      />
      <motion.rect
        y={HP.top}
        height={plotH}
        animate={{ x: x(threshold), width: Math.max(0, HP.left + plotW - x(threshold)) }}
        transition={{ type: "spring", stiffness: 200, damping: 26 }}
        fill={FAKE}
        opacity={0.06}
      />
      <text x={HP.left + 8} y={HP.top + 14} className="text-[12px] font-semibold" fill={REAL}>
        called real
      </text>
      <text x={HP.left + plotW - 8} y={HP.top + 14} textAnchor="end" className="text-[12px] font-semibold" fill={FAKE}>
        called synthetic
      </text>

      {bars(spoof, "fake")}
      {bars(bonafide, "real")}

      <line x1={HP.left} x2={HP.left + plotW} y1={HP.top + plotH} y2={HP.top + plotH} stroke={ink(0.2)} />

      {/* reference cuts */}
      <line x1={x(shippedThreshold)} x2={x(shippedThreshold)} y1={HP.top + 20} y2={HP.top + plotH} stroke={ink(0.35)} strokeDasharray="4 4" />
      <line x1={x(eerThreshold)} x2={x(eerThreshold)} y1={HP.top + 20} y2={HP.top + plotH} stroke={MID} strokeDasharray="2 3" strokeWidth={1.5} />

      {/* the live cut */}
      <motion.g animate={{ x: x(threshold) }} transition={{ type: "spring", stiffness: 260, damping: 26 }}>
        <line x1={0} x2={0} y1={HP.top} y2={HP.top + plotH} stroke={INK} strokeWidth={2.5} />
        <circle cx={0} cy={HP.top} r={5} fill={INK} />
      </motion.g>

      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <text key={tick} x={x(tick)} y={HH - 14} textAnchor="middle" className="fill-slate-400 text-[11px]">
          {tick.toFixed(2)}
        </text>
      ))}
    </svg>
  );
};

const DS = 300;
const DP = { top: 14, right: 14, bottom: 40, left: 44 };

/** DET curve drawn in with a stroke animation; the live cut rides along it. */
export const DetChart = ({
  points,
  live,
  eerPercent,
}: {
  points: DetPoint[];
  live: DetPoint;
  eerPercent: number;
}) => {
  const { REAL, MID, FAKE, INK, ink, CANVAS } = usePalette();
  const plot = DS - DP.left - DP.right;
  const plotH = DS - DP.top - DP.bottom;
  const x = (rate: number) => DP.left + rate * plot;
  const y = (rate: number) => DP.top + (1 - rate) * plotH;

  const ordered = useMemo(() => [...points].sort((a, b) => a.threshold - b.threshold), [points]);
  const path = ordered
    .map((point, index) => `${index ? "L" : "M"} ${x(point.false_rejection_rate).toFixed(1)} ${y(point.false_acceptance_rate).toFixed(1)}`)
    .join(" ");
  const eer = ordered.reduce((best, point) =>
    Math.abs(point.false_acceptance_rate - point.false_rejection_rate) <
    Math.abs(best.false_acceptance_rate - best.false_rejection_rate)
      ? point
      : best,
  );

  return (
    <svg viewBox={`0 0 ${DS} ${DS}`} className="mx-auto h-auto w-full max-w-[360px]" role="img" aria-label="Detection error tradeoff curve">
      <defs>
        <linearGradient id="df-det" x1="0" x2="1" y1="1" y2="0">
          <stop offset="0%" stopColor={REAL} />
          <stop offset="50%" stopColor={MID} />
          <stop offset="100%" stopColor={FAKE} />
        </linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={DP.top} y2={DP.top + plotH} stroke={ink(0.05)} />
          <line x1={DP.left} x2={DP.left + plot} y1={y(tick)} y2={y(tick)} stroke={ink(0.05)} />
        </g>
      ))}
      {/* where both errors are equal */}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={ink(0.25)} strokeDasharray="3 4" />
      <motion.path
        d={path}
        fill="none"
        stroke="url(#df-det)"
        strokeWidth={3}
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        whileInView={{ pathLength: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.6, ease: "easeInOut" }}
      />
      <circle cx={x(eer.false_rejection_rate)} cy={y(eer.false_acceptance_rate)} r={6} fill={CANVAS} stroke={MID} strokeWidth={2.5} />
      <circle cx={x(eer.false_rejection_rate)} cy={y(eer.false_acceptance_rate)} r={6} fill="none" stroke={MID} strokeWidth={2} className="df-ripple" />
      <motion.circle
        r={6}
        fill={INK}
        animate={{ cx: x(live.false_rejection_rate), cy: y(live.false_acceptance_rate) }}
        transition={{ type: "spring", stiffness: 220, damping: 22 }}
        style={{ filter: "drop-shadow(0 0 8px white)" }}
      />
      {[0, 0.5, 1].map((tick) => (
        <g key={tick}>
          <text x={x(tick)} y={DS - 22} textAnchor="middle" className="fill-slate-400 text-[10px]">
            {(tick * 100).toFixed(0)}%
          </text>
          <text x={DP.left - 6} y={y(tick) + 3} textAnchor="end" className="fill-slate-400 text-[10px]">
            {(tick * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      <text x={DP.left + plot / 2} y={DS - 6} textAnchor="middle" className="fill-slate-300 text-[10px]">
        real voices flagged →
      </text>
      <text x={12} y={DP.top + plotH / 2} textAnchor="middle" transform={`rotate(-90 12 ${DP.top + plotH / 2})`} className="fill-slate-300 text-[10px]">
        fakes let through →
      </text>
      <text x={x(eer.false_rejection_rate) + 10} y={y(eer.false_acceptance_rate) - 8} className="text-[10px]" fill={MID}>
        EER {eerPercent.toFixed(1)}%
      </text>
    </svg>
  );
};

/** Mean score per fake-voice generator, as bars sliding in. */
export const AttackBars = ({ rows }: { rows: AttackSummary[] }) => {
  const { REAL, scoreColor } = usePalette();
  return (
    <div className="space-y-2">
      {rows.map((row, index) => {
        const colour = scoreColor(row.mean_score);
        return (
          <div key={row.attack} className="grid grid-cols-[88px_1fr_52px] items-center gap-3 text-xs">
            <span className="truncate font-mono text-slate-200">{row.attack === "bonafide" ? "genuine" : row.attack}</span>
            <div className="relative h-3 overflow-hidden rounded-full bg-white/5">
              <motion.div
                className="h-full rounded-full"
                style={{ background: `linear-gradient(90deg, ${REAL}55, ${colour})`, boxShadow: `0 0 16px -4px ${colour}` }}
                initial={{ width: 0 }}
                whileInView={{ width: `${row.mean_score * 100}%` }}
                viewport={{ once: true }}
                transition={{ type: "spring", stiffness: 60, damping: 15, delay: index * 0.07 }}
              />
            </div>
            <span className="text-right font-mono tabular-nums text-slate-300">{row.mean_score.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
};
