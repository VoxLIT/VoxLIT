import { useId, useMemo, type ReactNode } from "react";
import { usePalette } from "./theme";

/**
 * Vector artwork for the hero: the same head in profile, drawn once as a
 * person and once as a machine, framed like an analyser readout.
 *
 * - Person: a smooth silhouette with soft contour lines, a flowing waveform
 *   whose amplitude drifts, and an irregular spectrum.
 * - Machine: a faceted low-poly shell with glowing nodes, a stepped
 *   (quantised) waveform, a scan line, and a regular, gridded spectrum.
 *
 * Colours come from the active palette, so both follow the theme. Motion is
 * CSS-driven (see deepfake-page.css) and switches off for reduced motion.
 */

const W = 300;
const H = 400;
/** The heads are authored on a 400×400 grid; this fits them into the portrait frame. */
const CONTENT = "translate(-22 40) scale(0.8)";
const MOUTH_Y = 226;
const WAVE_START = 250;
const WAVE_END = 392;

const HUMAN_HEAD =
  "M 80 340 C 70 290, 55 240, 62 190 C 70 120, 120 80, 170 88 C 205 94, 222 125, 218 160 L 236 196 C 238 202, 232 205, 226 206 L 224 218 C 228 222, 226 228, 220 230 C 224 236, 222 242, 216 244 L 214 262 C 212 276, 200 282, 184 282 L 176 284 L 178 340 Z";

const MACHINE_OUTLINE: [number, number][] = [
  [80, 340], [62, 250], [66, 180], [92, 118], [140, 88], [190, 94], [216, 130], [218, 160], [238, 198], [226, 208],
  [226, 228], [218, 232], [218, 246], [214, 264], [196, 282], [176, 284], [178, 340],
];
const MACHINE_INNER: [number, number][] = [
  [112, 150], [160, 128], [192, 176], [100, 214], [148, 196], [196, 232], [124, 268], [160, 300], [104, 312],
];

/** Low-poly mesh: every inner node joins its nearest neighbours. */
const meshEdges = () => {
  const all = [...MACHINE_OUTLINE, ...MACHINE_INNER];
  const edges = new Set<string>();
  MACHINE_INNER.forEach(([x, y]) => {
    all
      .map(([ox, oy]) => ({ ox, oy, d: Math.hypot(ox - x, oy - y) }))
      .filter(({ d }) => d > 0)
      .sort((a, b) => a.d - b.d)
      .slice(0, 4)
      .forEach(({ ox, oy }) => {
        if (!edges.has([ox, oy, x, y].join(","))) edges.add([x, y, ox, oy].join(","));
      });
  });
  return [...edges].map((key) => key.split(",").map(Number) as [number, number, number, number]);
};

const humanWave = (phase: number, scale: number) => {
  const points: string[] = [];
  for (let x = WAVE_START; x <= WAVE_END; x += 2) {
    const t = (x - WAVE_START) / (WAVE_END - WAVE_START);
    const envelope = 42 * scale * Math.pow(1 - t, 0.8) * (0.6 + 0.4 * Math.sin(t * 6 + phase));
    const y = MOUTH_Y + envelope * Math.sin(t * 22 + phase * 1.7 + Math.sin(t * 5) * 1.3);
    points.push(`${x === WAVE_START ? "M" : "L"} ${x} ${y.toFixed(1)}`);
  }
  return points.join(" ");
};

const machineWave = (phase: number, scale: number) => {
  const step = 9;
  const parts = [`M ${WAVE_START} ${MOUTH_Y}`];
  for (let x = WAVE_START; x < WAVE_END; x += step) {
    const t = (x - WAVE_START) / (WAVE_END - WAVE_START);
    const raw = 42 * scale * (1 - t * 0.55) * Math.sin(t * 19 + phase);
    parts.push(`V ${MOUTH_Y + Math.round(raw / 10) * 10} H ${x + step}`);
  }
  return parts.join(" ");
};

/** Spectrum strip along the bottom: organic for the person, quantised and periodic for the machine. */
const BARS = 28;
const humanSpectrum = Array.from({ length: BARS }, (_, index) => {
  const t = index / (BARS - 1);
  return 0.25 + 0.55 * Math.abs(Math.sin(t * 5.3 + 0.4) * Math.cos(t * 2.1)) + 0.2 * Math.abs(Math.sin(index * 1.7));
});
const machineSpectrum = Array.from({ length: BARS }, (_, index) => [0.35, 0.7, 0.5, 0.9][index % 4] * (1 - (index / BARS) * 0.35));

const Frame = ({ label, children }: { label: string; children: ReactNode }) => (
  <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
    {children}
  </svg>
);

/** Corner brackets, a header tag and a footer caption: the "analyser" chrome. */
const Hud = ({ colour, ink, tag, caption }: { colour: string; ink: (alpha: number) => string; tag: string; caption: string }) => (
  <g>
    <g fill="none" stroke={colour} strokeWidth={1.5} strokeOpacity={0.8}>
      <path d="M 16 34 V 16 H 34" />
      <path d={`M ${W - 34} 16 H ${W - 16} V 34`} />
      <path d={`M 16 ${H - 34} V ${H - 16} H 34`} />
      <path d={`M ${W - 34} ${H - 16} H ${W - 16} V ${H - 34}`} />
    </g>
    <g className="font-mono" fontSize={9} letterSpacing={1.2}>
      <circle cx={30} cy={36} r={3} fill={colour} className="df-art-blink" />
      <text x={40} y={39} fill={colour} fontWeight={600}>
        {tag}
      </text>
      <text x={W - 28} y={39} fill={ink(0.45)} textAnchor="end">
        16 kHz · mono
      </text>
      <text x={W / 2} y={H - 26} fill={ink(0.5)} textAnchor="middle">
        {caption}
      </text>
    </g>
  </g>
);

const Spectrum = ({ values, colour, stepped }: { values: number[]; colour: string; stepped?: boolean }) => {
  const left = 34;
  const width = W - left * 2;
  const base = H - 44;
  const gap = width / values.length;
  return (
    <g>
      {values.map((value, index) => {
        const height = stepped ? Math.round(value * 4) * 7 : value * 28;
        return (
          <rect
            key={index}
            x={left + index * gap}
            y={base - height}
            width={gap * 0.55}
            height={height}
            rx={stepped ? 0 : gap * 0.27}
            fill={colour}
            fillOpacity={0.35 + value * 0.45}
          />
        );
      })}
      <line x1={left} x2={left + width} y1={base + 3} y2={base + 3} stroke={colour} strokeOpacity={0.35} />
    </g>
  );
};

const Glow = ({ id }: { id: string }) => (
  <filter id={id} x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur stdDeviation="3" result="blur" />
    <feMerge>
      <feMergeNode in="blur" />
      <feMergeNode in="SourceGraphic" />
    </feMerge>
  </filter>
);

export const HumanVoiceArt = () => {
  const { REAL, ink } = usePalette();
  const id = useId().replace(/:/g, "");
  const waves = useMemo(() => [humanWave(0, 1), humanWave(1.4, 0.7), humanWave(2.6, 0.45)], []);
  return (
    <Frame label="Illustration: a human head in profile speaking a smooth, uneven natural waveform">
      <defs>
        <radialGradient id={`${id}-bg`} gradientUnits="userSpaceOnUse" cx={110} cy={190} r={260}>
          <stop offset="0%" stopColor={REAL} stopOpacity={0.18} />
          <stop offset="100%" stopColor={REAL} stopOpacity={0} />
        </radialGradient>
        <linearGradient id={`${id}-head`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={REAL} stopOpacity={0.34} />
          <stop offset="100%" stopColor={REAL} stopOpacity={0.06} />
        </linearGradient>
        <radialGradient id={`${id}-voice`} gradientUnits="userSpaceOnUse" cx={206} cy={236} r={70}>
          <stop offset="0%" stopColor={REAL} stopOpacity={0.55} />
          <stop offset="100%" stopColor={REAL} stopOpacity={0} />
        </radialGradient>
        <clipPath id={`${id}-clip`}>
          <path d={HUMAN_HEAD} />
        </clipPath>
        <pattern id={`${id}-dots`} width="14" height="14" patternUnits="userSpaceOnUse">
          <circle cx="2" cy="2" r="0.9" fill={ink(0.14)} />
        </pattern>
        <Glow id={`${id}-glow`} />
      </defs>
      <rect x={-W} y={-H} width={W * 3} height={H * 3} fill={`url(#${id}-dots)`} />
      <rect x={-W} y={-H} width={W * 3} height={H * 3} fill={`url(#${id}-bg)`} />

      <g transform={CONTENT}>
        <path d={HUMAN_HEAD} fill={`url(#${id}-head)`} />
        <g clipPath={`url(#${id}-clip)`}>
          {/* soft contours, and warmth where the voice is made */}
          <g fill="none" stroke={REAL} strokeOpacity={0.22} strokeWidth={1.1}>
            {[30, 52, 74, 96, 118, 140].map((r, index) => (
              <ellipse key={r} cx={136 + index * 3} cy={194 - index * 2} rx={r * 1.12} ry={r} transform={`rotate(${-14 + index * 5} 136 194)`} />
            ))}
          </g>
          <circle cx={206} cy={236} r={70} fill={`url(#${id}-voice)`} className="df-art-breathe" />
        </g>
        <path d={HUMAN_HEAD} fill="none" stroke={REAL} strokeWidth={2.4} strokeLinejoin="round" filter={`url(#${id}-glow)`} />
        <path d="M 186 152 q 8 -5 16 0" fill="none" stroke={REAL} strokeWidth={2.4} strokeLinecap="round" />
        <path d="M 112 176 c -14 4 -14 34 2 38" fill="none" stroke={REAL} strokeWidth={2.4} strokeLinecap="round" />

        {waves.map((d, index) => (
          <path key={index} d={d} fill="none" stroke={REAL} strokeWidth={index === 0 ? 2.8 : 1.6} strokeOpacity={1 - index * 0.32} strokeLinecap="round" />
        ))}
        {/* a bright pulse travelling out along the main wave */}
        <path d={waves[0]} fill="none" stroke={ink(0.95)} strokeWidth={3} strokeLinecap="round" pathLength={100} className="df-art-pulse" filter={`url(#${id}-glow)`} />
      </g>

      <Spectrum values={humanSpectrum} colour={REAL} />
      <Hud colour={REAL} ink={ink} tag="HUMAN" caption="breath · pitch drift · micro-pauses" />
    </Frame>
  );
};

export const MachineVoiceArt = () => {
  const { FAKE, ink } = usePalette();
  const id = useId().replace(/:/g, "");
  const waves = useMemo(() => [machineWave(0, 1), machineWave(1.3, 0.6)], []);
  const edges = useMemo(meshEdges, []);
  const outline = MACHINE_OUTLINE.map((point) => point.join(",")).join(" ");
  return (
    <Frame label="Illustration: a faceted machine head in profile speaking a stepped, quantised waveform">
      <defs>
        <radialGradient id={`${id}-bg`} gradientUnits="userSpaceOnUse" cx={110} cy={190} r={260}>
          <stop offset="0%" stopColor={FAKE} stopOpacity={0.16} />
          <stop offset="100%" stopColor={FAKE} stopOpacity={0} />
        </radialGradient>
        <linearGradient id={`${id}-head`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={FAKE} stopOpacity={0.3} />
          <stop offset="100%" stopColor={FAKE} stopOpacity={0.05} />
        </linearGradient>
        <linearGradient id={`${id}-scan`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={FAKE} stopOpacity={0} />
          <stop offset="85%" stopColor={FAKE} stopOpacity={0.35} />
          <stop offset="100%" stopColor={FAKE} stopOpacity={0.9} />
        </linearGradient>
        <clipPath id={`${id}-clip`}>
          <polygon points={outline} />
        </clipPath>
        <pattern id={`${id}-grid`} width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 H 0 V 20" fill="none" stroke={ink(0.07)} strokeWidth={1} />
        </pattern>
        <Glow id={`${id}-glow`} />
      </defs>
      <rect x={-W} y={-H} width={W * 3} height={H * 3} fill={`url(#${id}-grid)`} />
      <rect x={-W} y={-H} width={W * 3} height={H * 3} fill={`url(#${id}-bg)`} />

      <g transform={CONTENT}>
        <polygon points={outline} fill={`url(#${id}-head)`} />
        <g clipPath={`url(#${id}-clip)`}>
          <g stroke={FAKE} strokeOpacity={0.45} strokeWidth={1}>
            {edges.map(([x1, y1, x2, y2]) => (
              <line key={`${x1},${y1},${x2},${y2}`} x1={x1} y1={y1} x2={x2} y2={y2} />
            ))}
          </g>
          {/* scan line sweeping down the shell */}
          <rect x={40} y={60} width={220} height={40} fill={`url(#${id}-scan)`} className="df-art-scan" />
        </g>
        <polygon points={outline} fill="none" stroke={FAKE} strokeWidth={2.4} strokeLinejoin="miter" filter={`url(#${id}-glow)`} />
        {MACHINE_INNER.map(([x, y], index) => (
          <rect key={`${x},${y}`} x={x - 3} y={y - 3} width={6} height={6} fill={FAKE} className="df-art-node" style={{ animationDelay: `${index * 0.35}s` }} />
        ))}
        {/* eye: a sensor, not a glance */}
        <rect x={182} y={144} width={20} height={10} rx={1} fill="none" stroke={FAKE} strokeWidth={2.2} />
        <rect x={188} y={147} width={8} height={4} fill={ink(0.95)} className="df-art-blink" />

        {waves.map((d, index) => (
          <path key={index} d={d} fill="none" stroke={FAKE} strokeWidth={index === 0 ? 2.8 : 1.6} strokeOpacity={1 - index * 0.45} strokeLinejoin="miter" />
        ))}
        <path d={waves[0]} fill="none" stroke={ink(0.95)} strokeWidth={3} pathLength={100} className="df-art-pulse" filter={`url(#${id}-glow)`} />
      </g>

      <Spectrum values={machineSpectrum} colour={FAKE} stepped />
      <Hud colour={FAKE} ink={ink} tag="SYNTHETIC" caption="vocoder · fixed frames · clean spectrum" />
    </Frame>
  );
};
