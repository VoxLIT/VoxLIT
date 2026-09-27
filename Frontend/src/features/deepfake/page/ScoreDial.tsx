import { useEffect } from "react";
import { motion, useSpring, useTransform } from "motion/react";
import { usePalette } from "./theme";
import { formatScore } from "./palette";

const R = 110;
const CX = 140;
const CY = 140;

const polar = (fraction: number, radius = R) => {
  const angle = Math.PI * (1 - fraction);
  return [CX + radius * Math.cos(angle), CY - radius * Math.sin(angle)] as const;
};

/**
 * The spoof score as a half-circle dial: real on the left, synthetic on the
 * right, the threshold as a notch. The needle springs to the score so the
 * reading feels like a measurement, not a label.
 */
export const ScoreDial = ({ score, threshold }: { score: number; threshold: number }) => {
  const { REAL, MID, FAKE, INK, ink, CANVAS } = usePalette();
  const spring = useSpring(0, { stiffness: 60, damping: 14, mass: 0.9 });
  useEffect(() => {
    spring.set(Math.max(0, Math.min(1, score)));
  }, [score, spring]);

  // The needle's tip, moved directly: an SVG rotate origin would be measured
  // from the element's own box rather than from the dial's hub.
  const tipX = useTransform(spring, (value) => polar(value, R - 8)[0]);
  const tipY = useTransform(spring, (value) => polar(value, R - 8)[1]);
  const text = useTransform(spring, (value) => formatScore(value));
  const arcLength = useTransform(spring, (value) => value);

  const [tx1, ty1] = polar(threshold, R - 18);
  const [tx2, ty2] = polar(threshold, R + 14);

  return (
    <div className="relative mx-auto w-full max-w-[300px]">
      <svg viewBox="0 0 280 170" className="w-full" role="img" aria-label={`Spoof score ${formatScore(score)} on a 0 to 1 dial`}>
        <defs>
          <linearGradient id="df-dial" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor={REAL} />
            <stop offset="50%" stopColor={MID} />
            <stop offset="100%" stopColor={FAKE} />
          </linearGradient>
          <filter id="df-dial-glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* track */}
        <path
          d={`M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`}
          fill="none"
          stroke={ink(0.08)}
          strokeWidth={18}
          strokeLinecap="round"
        />
        {/* filled arc, grows with the needle */}
        <motion.path
          d={`M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`}
          fill="none"
          stroke="url(#df-dial)"
          strokeWidth={18}
          strokeLinecap="round"
          filter="url(#df-dial-glow)"
          style={{ pathLength: arcLength }}
        />
        {/* threshold notch */}
        <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke={INK} strokeWidth={2.5} strokeLinecap="round" />
        <text x={tx2} y={ty2 - 6} textAnchor="middle" className="fill-slate-300 text-[10px]">
          cut {threshold.toFixed(2)}
        </text>

        {/* needle */}
        <motion.line x1={CX} y1={CY} x2={tipX} y2={tipY} stroke={INK} strokeWidth={3} strokeLinecap="round" />
        <circle cx={CX} cy={CY} r={9} fill={CANVAS} stroke={INK} strokeWidth={3} />

        <text x={CX - R} y={CY + 24} textAnchor="middle" className="text-[11px] font-semibold" fill={REAL}>
          real
        </text>
        <text x={CX + R} y={CY + 24} textAnchor="middle" className="text-[11px] font-semibold" fill={FAKE}>
          synthetic
        </text>
      </svg>
      <div className="-mt-1 text-center">
        <motion.div className="font-mono text-3xl font-semibold tabular-nums text-white">{text}</motion.div>
        <div className="text-[11px] uppercase tracking-widest text-slate-400">spoof score</div>
      </div>
    </div>
  );
};
