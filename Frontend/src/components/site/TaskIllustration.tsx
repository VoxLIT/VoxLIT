import { TaskId } from "@/tasks/types";

/**
 * Vector figures for the home-page task tiles. Each one is a schematic of what
 * the task's workbench actually shows, drawn in the brand palette so the
 * tiles read like figures from a paper rather than stock photography.
 */

const BLUE = "#0073E6";
const BLUE_LIGHT = "#7CC4FF";
const ORANGE = "#F06638";
const INK = "#334155";
const MUTED = "#94A3B8";
const RULE = "#E2E8F0";

/** Deterministic pseudo-random sequence so figures are stable across renders. */
const seq = (n: number, seed: number) =>
  Array.from({ length: n }, (_, i) => {
    const x = Math.sin((i + 1) * 12.9898 + seed * 78.233) * 43758.5453;
    return x - Math.floor(x);
  });

/** Speech-like envelope: a few syllable bumps with jitter. */
const envelope = (n: number, seed: number, bumps = 5) => {
  const r = seq(n, seed);
  return r.map((v, i) => {
    const t = i / (n - 1);
    const syll = Math.abs(Math.sin(t * Math.PI * bumps + seed));
    const edge = Math.min(1, t * 8, (1 - t) * 8);
    return Math.max(0.06, (0.25 + 0.75 * syll) * (0.55 + 0.45 * v) * edge);
  });
};

const Wave = ({
  x, y, w, h, n, seed, color = BLUE, highlight,
}: {
  x: number; y: number; w: number; h: number; n: number; seed: number; color?: string;
  highlight?: (i: number) => boolean;
}) => {
  const env = envelope(n, seed);
  const step = w / n;
  return (
    <g>
      {env.map((v, i) => {
        const bh = v * h;
        return (
          <rect
            key={i}
            x={x + i * step}
            y={y - bh / 2}
            width={Math.max(1.2, step * 0.55)}
            height={bh}
            rx={Math.min(1.5, step * 0.27)}
            fill={highlight?.(i) ? ORANGE : color}
          />
        );
      })}
    </g>
  );
};

const Label = ({ x, y, children, anchor = "start", size = 9, color = MUTED, weight = 500 }: {
  x: number; y: number; children: string; anchor?: "start" | "middle" | "end"; size?: number; color?: string; weight?: number;
}) => (
  <text x={x} y={y} textAnchor={anchor} fontSize={size} fill={color} fontWeight={weight}
    fontFamily="Inter, system-ui, sans-serif" letterSpacing="0.02em">
    {children}
  </text>
);

const Transcription = () => {
  const tokens = [
    { t: "the", from: 4, to: 12, s: 0.25 },
    { t: "model", from: 14, to: 27, s: 0.9 },
    { t: "heard", from: 29, to: 40, s: 0.55 },
    { t: "this", from: 42, to: 52, s: 0.35 },
  ];
  const n = 56;
  const x0 = 28, w = 344, step = w / n;
  const hot = (i: number) => (i >= 17 && i <= 24) || (i >= 32 && i <= 35);
  return (
    <>
      <Label x={28} y={30}>WAVEFORM · GRADIENT SALIENCY</Label>
      <Wave x={x0} y={82} w={w} h={70} n={n} seed={1} color={BLUE_LIGHT} highlight={hot} />
      {tokens.map(({ t, from, to, s }) => {
        const cx = x0 + ((from + to) / 2) * step;
        const tw = Math.max(36, t.length * 8 + 16);
        return (
          <g key={t}>
            <path d={`M${x0 + from * step},124 L${x0 + from * step},130 L${x0 + to * step},130 L${x0 + to * step},124`}
              fill="none" stroke={MUTED} strokeWidth={1} />
            <line x1={cx} y1={130} x2={cx} y2={146} stroke={MUTED} strokeWidth={1} strokeDasharray="2 2" />
            <rect x={cx - tw / 2} y={148} width={tw} height={24} rx={6}
              fill={ORANGE} fillOpacity={0.1 + s * 0.75} stroke={ORANGE} strokeOpacity={0.5} />
            <Label x={cx} y={164} anchor="middle" size={11} color={s > 0.6 ? "#fff" : INK} weight={600}>{t}</Label>
          </g>
        );
      })}
      <Label x={28} y={196}>DECODED TOKENS · ATTRIBUTION</Label>
    </>
  );
};

const Emotion = () => {
  const classes = [
    ["neutral", 0.12], ["happy", 0.71], ["sad", 0.05], ["angry", 0.08], ["fearful", 0.04],
  ] as const;
  const n = 34;
  return (
    <>
      <Label x={28} y={30}>INPUT</Label>
      <Wave x={28} y={110} w={150} h={90} n={n} seed={4} color={BLUE_LIGHT}
        highlight={(i) => i >= 11 && i <= 19} />
      <path d="M190,110 L206,110" stroke={MUTED} strokeWidth={1.2} />
      <path d="M204,106 L211,110 L204,114 Z" fill={MUTED} />
      <Label x={224} y={30}>CLASS PROBABILITY</Label>
      {classes.map(([name, p], i) => {
        const y = 48 + i * 30;
        const top = p > 0.5;
        return (
          <g key={name}>
            <Label x={224} y={y + 13} size={10} color={INK} weight={top ? 600 : 500}>{name}</Label>
            <rect x={278} y={y + 3} width={68} height={14} rx={3} fill={RULE} />
            <rect x={278} y={y + 3} width={68 * p} height={14} rx={3} fill={top ? ORANGE : BLUE_LIGHT} />
            <Label x={374} y={y + 14} anchor="end" size={9} color={top ? INK : MUTED}>{p.toFixed(2)}</Label>
          </g>
        );
      })}
    </>
  );
};

const Verification = () => {
  const a = seq(18, 7), b = seq(18, 9), c = seq(18, 11), d = seq(18, 13);
  const clusterA = a.map((v, i) => [262 + (v - 0.5) * 50, 88 + (b[i] - 0.5) * 42]);
  const clusterB = c.map((v, i) => [318 + (v - 0.5) * 50, 138 + (d[i] - 0.5) * 42]);
  return (
    <>
      <Label x={28} y={30}>ENROLMENT</Label>
      <Wave x={28} y={66} w={150} h={40} n={32} seed={6} color={BLUE} />
      <Label x={28} y={124}>PROBE</Label>
      <Wave x={28} y={160} w={150} h={40} n={32} seed={8} color={ORANGE} />
      <path d="M186,66 C206,66 206,100 222,100" fill="none" stroke={MUTED} strokeWidth={1.2} />
      <path d="M186,160 C206,160 206,128 222,128" fill="none" stroke={MUTED} strokeWidth={1.2} />
      <rect x={222} y={40} width={150} height={150} rx={8} fill="#F8FAFC" stroke={RULE} />
      {[0, 1, 2].map((i) => (
        <line key={i} x1={222 + 37.5 * (i + 1)} y1={40} x2={222 + 37.5 * (i + 1)} y2={190} stroke={RULE} />
      ))}
      {clusterA.map(([x, y], i) => <circle key={`a${i}`} cx={x} cy={y} r={2.6} fill={BLUE} fillOpacity={0.8} />)}
      {clusterB.map(([x, y], i) => <circle key={`b${i}`} cx={x} cy={y} r={2.6} fill={ORANGE} fillOpacity={0.8} />)}
      <line x1={262} y1={88} x2={318} y2={138} stroke={INK} strokeWidth={1.2} strokeDasharray="3 3" />
      <rect x={230} y={160} width={80} height={22} rx={11} fill="#fff" stroke={RULE} />
      <Label x={270} y={175} anchor="middle" size={10} color={INK} weight={600}>cos = 0.82</Label>
      <Label x={366} y={184} anchor="end" size={8}>EMBEDDING SPACE</Label>
    </>
  );
};

const Diarization = () => {
  const speakers = [
    { name: "SPK 1", color: BLUE, segs: [[0, 0.22], [0.47, 0.62], [0.86, 1]] },
    { name: "SPK 2", color: BLUE_LIGHT, segs: [[0.2, 0.44], [0.7, 0.84]] },
    { name: "SPK 3", color: ORANGE, segs: [[0.6, 0.73]] },
  ];
  const x0 = 76, w = 296;
  return (
    <>
      <Label x={28} y={30}>WHO SPOKE WHEN · CONFIDENCE</Label>
      {speakers.map((s, row) => {
        const y = 52 + row * 42;
        return (
          <g key={s.name}>
            <Label x={28} y={y + 17} size={10} color={INK} weight={600}>{s.name}</Label>
            <line x1={x0} y1={y + 13} x2={x0 + w} y2={y + 13} stroke={RULE} />
            {s.segs.map(([a, b], i) => {
              const sx = x0 + a * w, sw = (b - a) * w;
              return (
                <g key={i}>
                  {/* uncertainty margins at segment edges */}
                  <rect x={sx - 6} y={y + 2} width={sw + 12} height={22} rx={5} fill={s.color} fillOpacity={0.18} />
                  <rect x={sx} y={y + 2} width={sw} height={22} rx={4} fill={s.color} />
                </g>
              );
            })}
          </g>
        );
      })}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x0 + t * w} y1={180} x2={x0 + t * w} y2={184} stroke={MUTED} />
          <Label x={x0 + t * w} y={196} anchor="middle" size={8}>{`${(t * 20).toFixed(0)}s`}</Label>
        </g>
      ))}
      <line x1={x0} y1={180} x2={x0 + w} y2={180} stroke={MUTED} />
    </>
  );
};

const Deepfake = () => {
  const cols = 26, rows = 12, cw = 7, ch = 10.5;
  const r = seq(cols * rows, 21);
  const env = envelope(cols, 5, 4);
  const models = [["A", 0.18], ["B", 0.64], ["C", 0.81]] as const;
  return (
    <>
      <Label x={28} y={30}>SPECTROGRAM · ARTEFACT SALIENCY</Label>
      <g>
        {Array.from({ length: rows }).map((_, row) =>
          Array.from({ length: cols }).map((__, col) => {
            const low = 1 - row / rows;
            const v = Math.min(1, env[col] * (0.35 + 0.75 * low) * (0.6 + 0.5 * r[row * cols + col]));
            return (
              <rect key={`${row}-${col}`} x={28 + col * cw} y={44 + row * ch} width={cw - 1} height={ch - 1}
                fill={BLUE} fillOpacity={0.06 + v * 0.8} />
            );
          })
        )}
        <rect x={28 + 15 * cw - 2} y={44 - 2} width={6 * cw + 3} height={4 * ch + 3} rx={3}
          fill="none" stroke={ORANGE} strokeWidth={2} />
        <Label x={28 + 15 * cw - 2} y={44 + rows * ch + 12} size={8} color={ORANGE} weight={600}>▲ high-band artefact</Label>
      </g>
      <Label x={244} y={30}>SPOOF SCORE</Label>
      {models.map(([m, p], i) => {
        const y = 52 + i * 38;
        const spoof = p > 0.5;
        return (
          <g key={m}>
            <Label x={244} y={y + 14} size={10} color={INK} weight={600}>{`Model ${m}`}</Label>
            <rect x={244} y={y + 20} width={128} height={8} rx={4} fill={RULE} />
            <rect x={244} y={y + 20} width={128 * p} height={8} rx={4} fill={spoof ? ORANGE : BLUE_LIGHT} />
          </g>
        );
      })}
      <line x1={244 + 64} y1={60} x2={244 + 64} y2={172} stroke={INK} strokeDasharray="2 3" strokeWidth={1} />
      <Label x={244 + 64} y={186} anchor="middle" size={8}>threshold</Label>
    </>
  );
};

const FIGURES: Record<TaskId, () => JSX.Element> = {
  transcription: Transcription,
  emotion: Emotion,
  verification: Verification,
  "task-b": Diarization,
  deepfake: Deepfake,
};

const TaskIllustration = ({ taskId, className = "" }: { taskId: TaskId; className?: string }) => {
  const Figure = FIGURES[taskId];
  return (
    <svg viewBox="0 0 400 216" className={className} role="img" aria-hidden="true" preserveAspectRatio="xMidYMid meet">
      <Figure />
    </svg>
  );
};

export default TaskIllustration;
