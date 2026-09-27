import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
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

const FONT = "ui-sans-serif, system-ui, -apple-system, sans-serif";
const NUMERIC = { fontVariantNumeric: "tabular-nums" } as const;
// keeps a label legible where it crosses a curve or gridline
const HALO = { stroke: "#ffffff", strokeWidth: 3, strokeLinejoin: "round", paintOrder: "stroke" } as const;
const NICE_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000];

/**
 * Width of the chart's container, so the SVG is drawn 1:1 in CSS pixels.
 * A fixed viewBox scales its text with the card; at full page width that
 * made 11px tick labels render near 28px.
 */
const useContainerWidth = (fallback: number) => {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    // measure now too: the observer's first callback waits for a rendered frame
    if (element.clientWidth > 0) setWidth(element.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const next = Math.round(entry.contentRect.width);
      if (next > 0) setWidth(next);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
};

interface Marker {
  key: string;
  value: number;
  subs: string[];
  colour: string;
  dash?: string;
  strong?: boolean;
}

/**
 * Class-conditional score histograms on a shared axis, with the decision
 * threshold τ and the two reference thresholds. Where both classes share a
 * bin the bars sit side by side rather than on top of each other; bins on the
 * wrong side of τ are hatched — those are the misclassifications.
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
  const { REAL, FAKE, MID, INK, ink } = usePalette();
  const [ref, width] = useContainerWidth(760);
  const W = Math.max(320, width);
  const H = 320;
  const M = { top: 50, right: 18, bottom: 50, left: 58 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const base = M.top + plotH;
  const x = (score: number) => M.left + score * plotW;

  const peak = Math.max(1, ...bonafide.map((bin) => bin.count), ...spoof.map((bin) => bin.count));
  // the axis ends on a labelled round number at or above the tallest bin
  const axisMax = NICE_COUNTS.find((value) => value >= peak) ?? peak;
  // square-root count axis so a bin of 2 clips is still visible next to one of 100
  const h = (count: number) => (Math.sqrt(count) / Math.sqrt(axisMax)) * plotH;
  const yTicks = [0];
  for (const value of NICE_COUNTS) {
    if (value > axisMax) break;
    if (h(value) - h(yTicks[yTicks.length - 1]) >= 26) yTicks.push(value);
  }
  if (yTicks[yTicks.length - 1] !== axisMax) yTicks[yTicks.length - 1] = axisMax;
  const xTicks = Array.from({ length: 11 }, (_, index) => index / 10);

  const occupied = (bins: ScoreBin[]) => new Set(bins.filter((bin) => bin.count > 0).map((bin) => bin.bin_start));
  const realBins = occupied(bonafide);
  const fakeBins = occupied(spoof);

  const bars = (bins: ScoreBin[], kind: "real" | "fake") =>
    bins
      .filter((bin) => bin.count > 0)
      .map((bin) => {
        const mid = (bin.bin_start + bin.bin_end) / 2;
        const wrong = kind === "real" ? mid >= threshold : mid < threshold;
        const colour = kind === "real" ? REAL : FAKE;
        const shared = realBins.has(bin.bin_start) && fakeBins.has(bin.bin_start);
        const binLeft = x(bin.bin_start) + 1;
        const binWidth = Math.max(2, x(bin.bin_end) - x(bin.bin_start) - 2);
        const left = shared && kind === "fake" ? binLeft + binWidth / 2 : binLeft;
        const barWidth = shared ? binWidth / 2 : binWidth;
        const height = Math.max(1, h(bin.count));
        return (
          <motion.g
            key={`${kind}-${bin.bin_start}`}
            style={{ transformBox: "fill-box", originY: 1 }}
            initial={{ scaleY: 0 }}
            whileInView={{ scaleY: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.45, ease: "easeOut" }}
          >
            <rect x={left} y={base - height} width={barWidth} height={height} fill={colour} fillOpacity={0.4} stroke={colour} strokeWidth={1} />
            {wrong && (
              <rect x={left} y={base - height} width={barWidth} height={height} fill={`url(#df-hatch-${kind})`} stroke={INK} strokeWidth={1.25} />
            )}
            <title>
              {`${kind === "real" ? "Bona fide" : "Spoof"}: n = ${bin.count}, s ∈ [${bin.bin_start.toFixed(2)}, ${bin.bin_end.toFixed(2)})${
                wrong ? `, misclassified at threshold ${threshold.toFixed(3)}` : ""
              }`}
            </title>
          </motion.g>
        );
      });

  // Thresholds that coincide share one line and one label ("τ = τ_op = 0.500").
  const near = (a: number, b: number) => Math.abs(a - b) < 0.0025;
  const markers: Marker[] = [{ key: "live", value: threshold, subs: [""], colour: INK, strong: true }];
  const addReference = (value: number, sub: string, colour: string, dash: string) => {
    const same = markers.find((marker) => near(marker.value, value));
    if (same) same.subs.push(sub);
    else markers.push({ key: sub, value, subs: [sub], colour, dash });
  };
  addReference(eerThreshold, "EER", MID, "5 3");
  addReference(shippedThreshold, "op", ink(0.55), "1.5 2.5");

  // Labels sit in a strip above the plot; a label that would overlap the one
  // before it on its row drops to the second row.
  const labelWidth = (marker: Marker) => (marker.subs.length * 4 + marker.subs.join("").length * 0.75 + 5) * 6.6;
  const rowsEnd = [-Infinity, -Infinity];
  const placed = [...markers]
    .sort((a, b) => a.value - b.value)
    .map((marker) => {
      const w = labelWidth(marker);
      const centre = Math.min(M.left + plotW - w / 2, Math.max(M.left + w / 2, x(marker.value)));
      const row = centre - w / 2 >= rowsEnd[0] + 8 ? 0 : 1;
      rowsEnd[row] = centre + w / 2;
      return { marker, centre, row };
    });
  const rowY = (row: number) => M.top - 30 + row * 16;

  return (
    <div ref={ref} className="w-full">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block max-w-full" role="img" aria-label="Class-conditional score histograms for bona fide and spoof clips" fontFamily={FONT}>
        <defs>
          {(["real", "fake"] as const).map((kind) => (
            <pattern key={kind} id={`df-hatch-${kind}`} width={5} height={5} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width={5} height={5} fill={kind === "real" ? REAL : FAKE} fillOpacity={0.4} />
              <line x1={0} y1={0} x2={0} y2={5} stroke={INK} strokeWidth={1} strokeOpacity={0.75} />
            </pattern>
          ))}
          <clipPath id="df-hist-plot">
            <rect x={M.left} y={M.top} width={plotW} height={plotH} />
          </clipPath>
        </defs>

        {/* spoof decision region */}
        <rect x={x(threshold)} y={M.top} width={Math.max(0, M.left + plotW - x(threshold))} height={plotH} fill={ink(0.035)} />

        {/* gridlines and y axis */}
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            {tick > 0 && <line x1={M.left} x2={M.left + plotW} y1={base - h(tick)} y2={base - h(tick)} stroke={ink(0.08)} />}
            <line x1={M.left - 4} x2={M.left} y1={base - h(tick)} y2={base - h(tick)} stroke={ink(0.55)} />
            <text x={M.left - 8} y={base - h(tick) + 4} textAnchor="end" fontSize={11} fill={ink(0.7)} style={NUMERIC}>
              {tick}
            </text>
          </g>
        ))}

        <g clipPath="url(#df-hist-plot)">
          {bars(spoof, "fake")}
          {bars(bonafide, "real")}
        </g>

        <rect x={M.left} y={M.top} width={plotW} height={plotH} fill="none" stroke={ink(0.3)} />

        {/* thresholds: one line each, labels in the strip above */}
        {placed.map(({ marker, centre, row }) => (
          <g key={marker.key}>
            <line
              x1={x(marker.value)}
              x2={x(marker.value)}
              y1={rowY(row) + 5}
              y2={base}
              stroke={marker.colour}
              strokeWidth={marker.strong ? 1.75 : 1.25}
              strokeDasharray={marker.dash}
            />
            <text x={centre} y={rowY(row)} textAnchor="middle" fontSize={11} fontWeight={marker.strong ? 600 : 500} fill={marker.colour} style={NUMERIC}>
              {marker.subs.map((sub, index) => (
                <tspan key={sub || "live"}>
                  τ
                  {sub && (
                    <tspan dy={3} fontSize={8}>
                      {sub}
                    </tspan>
                  )}
                  <tspan dy={sub ? -3 : 0}>{index < marker.subs.length - 1 ? " = " : ` = ${marker.value.toFixed(3)}`}</tspan>
                </tspan>
              ))}
            </text>
          </g>
        ))}

        {/* x axis */}
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line x1={x(tick)} x2={x(tick)} y1={base} y2={base + 4} stroke={ink(0.55)} />
            <text x={x(tick)} y={base + 17} textAnchor="middle" fontSize={11} fill={ink(0.7)} style={NUMERIC}>
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={M.left + plotW / 2} y={H - 8} textAnchor="middle" fontSize={12} fill={ink(0.85)}>
          Detector score s
        </text>
        <text x={16} y={M.top + plotH / 2} textAnchor="middle" transform={`rotate(-90 16 ${M.top + plotH / 2})`} fontSize={12} fill={ink(0.85)}>
          Clips per bin (√ scale)
        </text>
      </svg>
    </div>
  );
};

/** Inverse standard-normal CDF (Acklam's rational approximation, |ε| < 1.2e-9). */
const probit = (p: number): number => {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const tail = (q: number) =>
    (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  if (p < 0.02425) return tail(Math.sqrt(-2 * Math.log(p)));
  if (p > 1 - 0.02425) return -tail(Math.sqrt(-2 * Math.log(1 - p)));
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
};

const PROBIT_TICKS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4];
const PROBIT_MAX = 0.6;
const LINEAR_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];

type DetScale = "probit" | "linear";

/**
 * Detection error trade-off curve. Defaults to normal-deviate axes, the
 * ASVspoof / NIST convention, where a good detector's curve is not crushed
 * into the corner. Zero error rates cannot be placed on a probit axis, so
 * they are drawn on the axis floor, half the smallest non-zero rate.
 */
export const DetChart = ({
  points,
  live,
  threshold,
  eerPercent,
}: {
  points: DetPoint[];
  live: DetPoint;
  threshold: number;
  eerPercent: number;
}) => {
  const { REAL, INK, ink, CANVAS } = usePalette();
  const [scale, setScale] = useState<DetScale>("probit");
  const [ref, width] = useContainerWidth(380);
  const S = Math.min(460, Math.max(280, width));
  const M = { top: 14, right: 16, bottom: 52, left: 58 };
  const plot = S - M.left - M.right;
  const plotH = S - M.top - M.bottom;

  const ordered = useMemo(() => [...points].sort((a, b) => a.threshold - b.threshold), [points]);

  const floor = useMemo(() => {
    const rates = ordered.flatMap((point) => [point.false_acceptance_rate, point.false_rejection_rate]).filter((rate) => rate > 0);
    const smallest = rates.length ? Math.min(...rates) : 0.01;
    return Math.max(0.001, smallest / 2);
  }, [ordered]);

  const lo = scale === "probit" ? probit(floor) : 0;
  const hi = scale === "probit" ? probit(PROBIT_MAX) : 1;
  const unit = (rate: number) => {
    if (scale === "linear") return Math.min(1, Math.max(0, rate));
    const clamped = Math.min(PROBIT_MAX, Math.max(floor, rate));
    return (probit(clamped) - lo) / (hi - lo);
  };
  const x = (rate: number) => M.left + unit(rate) * plot;
  const y = (rate: number) => M.top + (1 - unit(rate)) * plotH;
  const ticks = scale === "probit" ? PROBIT_TICKS.filter((tick) => tick >= floor * 0.999) : LINEAR_TICKS;
  const tickLabel = (tick: number) => {
    const percent = tick * 100;
    return percent < 1 ? percent.toFixed(1) : percent.toFixed(0);
  };

  const path = ordered
    .map((point, index) => `${index ? "L" : "M"} ${x(point.false_rejection_rate).toFixed(1)} ${y(point.false_acceptance_rate).toFixed(1)}`)
    .join(" ");

  // By definition the EER point lies on the FAR = FRR diagonal.
  const eerRate = eerPercent / 100;
  const eerX = x(eerRate);
  const eerY = y(eerRate);
  const liveX = x(live.false_rejection_rate);
  const liveY = y(live.false_acceptance_rate);

  // Labels go up-right of their marker, flipped inward near an edge. When the
  // two markers are close the τ label moves below, or failing that, left.
  const right = M.left + plot;
  const bottom = M.top + plotH;
  const place = (px: number, py: number, chars: number, prefer: "above" | "below" | "left") => {
    const w = chars * 6.4;
    const fitsRight = px + 10 + w <= right;
    const fitsLeft = px - 10 - w >= M.left;
    const fitsAbove = py - 10 >= M.top + 11;
    const fitsBelow = py + 17 <= bottom - 3;
    if (prefer === "left" && fitsLeft) return { x: px - 10, y: fitsAbove ? py - 10 : py + 17, anchor: "end" } as const;
    const side = fitsRight ? { x: px + 10, anchor: "start" as const } : { x: px - 10, anchor: "end" as const };
    if (prefer === "below") return { ...side, y: fitsBelow ? py + 17 : py - 10 };
    return { ...side, y: fitsAbove ? py - 10 : py + 17 };
  };
  const eerText = `EER = ${eerPercent.toFixed(1)}%`;
  const liveText = `τ = ${threshold.toFixed(3)}`;
  const eerLabel = place(eerX, eerY, eerText.length, "above");
  const crowded = Math.hypot(eerX - liveX, eerY - liveY) < 40;
  const canGoBelow = liveY + 17 <= bottom - 3 && eerLabel.y < eerY;
  const liveLabel = place(liveX, liveY, liveText.length, crowded ? (canGoBelow ? "below" : "left") : "above");

  return (
    <div ref={ref} className="w-full">
      <div className="mb-2 flex justify-end">
        <div className="inline-flex rounded-md p-0.5 text-[11px] ring-1 ring-white/15" role="group" aria-label="DET axis scale">
          {(["probit", "linear"] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setScale(option)}
              aria-pressed={scale === option}
              className={`rounded px-2 py-0.5 font-medium transition ${scale === option ? "bg-white/10 text-white" : "text-slate-400 hover:text-slate-200"}`}
            >
              {option === "probit" ? "Normal deviate" : "Linear"}
            </button>
          ))}
        </div>
      </div>
      <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`} className="mx-auto block max-w-full" role="img" aria-label="Detection error trade-off curve" fontFamily={FONT}>
        <defs>
          <clipPath id="df-det-plot">
            <rect x={M.left} y={M.top} width={plot} height={plotH} />
          </clipPath>
        </defs>

        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={M.top} y2={M.top + plotH} stroke={ink(0.08)} />
            <line x1={M.left} x2={M.left + plot} y1={y(tick)} y2={y(tick)} stroke={ink(0.08)} />
            <line x1={x(tick)} x2={x(tick)} y1={M.top + plotH} y2={M.top + plotH + 4} stroke={ink(0.55)} />
            <line x1={M.left - 4} x2={M.left} y1={y(tick)} y2={y(tick)} stroke={ink(0.55)} />
            <text x={x(tick)} y={M.top + plotH + 17} textAnchor="middle" fontSize={11} fill={ink(0.7)} style={NUMERIC}>
              {tickLabel(tick)}
            </text>
            <text x={M.left - 8} y={y(tick) + 4} textAnchor="end" fontSize={11} fill={ink(0.7)} style={NUMERIC}>
              {tickLabel(tick)}
            </text>
          </g>
        ))}

        <g clipPath="url(#df-det-plot)">
          {/* FAR = FRR */}
          <line x1={M.left} y1={M.top + plotH} x2={M.left + plot} y2={M.top} stroke={ink(0.3)} strokeDasharray="3 4" />
          <motion.path
            key={scale}
            d={path}
            fill="none"
            stroke={REAL}
            strokeWidth={2}
            strokeLinejoin="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.9, ease: "easeInOut" }}
          />
        </g>
        <rect x={M.left} y={M.top} width={plot} height={plotH} fill="none" stroke={ink(0.3)} />

        <circle cx={eerX} cy={eerY} r={5} fill={CANVAS} stroke={INK} strokeWidth={1.75} />
        <text x={eerLabel.x} y={eerLabel.y} textAnchor={eerLabel.anchor} fontSize={11} fontWeight={500} fill={INK} style={NUMERIC} {...HALO}>
          {eerText}
        </text>

        <motion.rect
          width={8}
          height={8}
          fill={INK}
          initial={false}
          animate={{ x: liveX - 4, y: liveY - 4 }}
          transition={{ duration: 0.15 }}
        />
        <text x={liveLabel.x} y={liveLabel.y} textAnchor={liveLabel.anchor} fontSize={11} fill={ink(0.75)} style={NUMERIC} {...HALO}>
          {liveText}
        </text>

        <text x={M.left + plot / 2} y={S - 12} textAnchor="middle" fontSize={12} fill={ink(0.85)}>
          False rejection rate, bona fide (%)
        </text>
        <text x={16} y={M.top + plotH / 2} textAnchor="middle" transform={`rotate(-90 16 ${M.top + plotH / 2})`} fontSize={12} fill={ink(0.85)}>
          False acceptance rate, spoof (%)
        </text>
      </svg>
    </div>
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
