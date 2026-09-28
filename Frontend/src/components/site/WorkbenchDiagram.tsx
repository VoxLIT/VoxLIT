/** Schematic of the task workbench: three linked panels, with one selected
 *  datapoint driving the explanation and editor panels. */

const BLUE = "#0073E6";
const BLUE_LIGHT = "#7CC4FF";
const ORANGE = "#F06638";
const INK = "#334155";
const MUTED = "#94A3B8";
const RULE = "#E2E8F0";
const PANEL = "#F8FAFC";

const seq = (n: number, seed: number) =>
  Array.from({ length: n }, (_, i) => {
    const x = Math.sin((i + 1) * 12.9898 + seed * 78.233) * 43758.5453;
    return x - Math.floor(x);
  });

const T = ({ x, y, children, size = 9, color = MUTED, weight = 500, anchor = "start" }: {
  x: number; y: number; children: string; size?: number; color?: string; weight?: number; anchor?: "start" | "middle" | "end";
}) => (
  <text x={x} y={y} fontSize={size} fill={color} fontWeight={weight} textAnchor={anchor}
    fontFamily="Inter, system-ui, sans-serif" letterSpacing="0.02em">
    {children}
  </text>
);

const Panel = ({ x, title }: { x: number; title: string }) => (
  <g>
    <rect x={x} y={62} width={172} height={238} rx={8} fill="#fff" stroke={RULE} />
    <rect x={x} y={62} width={172} height={26} rx={8} fill={PANEL} />
    <rect x={x} y={80} width={172} height={8} fill={PANEL} />
    <line x1={x} y1={88} x2={x + 172} y2={88} stroke={RULE} />
    <T x={x + 12} y={79} size={9.5} color={INK} weight={600}>{title}</T>
  </g>
);

const WorkbenchDiagram = ({ className = "" }: { className?: string }) => {
  const r1 = seq(40, 3), r2 = seq(40, 5);
  const points = r1.map((v, i) => [30 + v * 140, 104 + r2[i] * 120, i % 3]);
  const wave = seq(44, 9).map((v, i) => {
    const t = i / 43;
    return (0.25 + 0.75 * Math.abs(Math.sin(t * Math.PI * 4))) * (0.5 + 0.5 * v) * Math.min(1, t * 8, (1 - t) * 8);
  });
  const attn = seq(36, 17);

  return (
    <svg viewBox="0 0 580 316" className={className} role="img" aria-label="Workbench layout: embeddings, explanations and datapoint editor panels">
      {/* Window chrome */}
      <rect x={4} y={4} width={572} height={308} rx={12} fill="#fff" stroke={RULE} />
      <circle cx={22} cy={22} r={4} fill="#FCA5A5" />
      <circle cx={36} cy={22} r={4} fill="#FCD34D" />
      <circle cx={50} cy={22} r={4} fill="#86EFAC" />
      <rect x={16} y={36} width={150} height={18} rx={5} fill={PANEL} stroke={RULE} />
      <T x={24} y={48.5} size={8.5} color={INK}>Task · Speech Transcription</T>
      <rect x={172} y={36} width={112} height={18} rx={5} fill={PANEL} stroke={RULE} />
      <T x={180} y={48.5} size={8.5} color={INK}>Model · Whisper Base</T>

      {/* Panel 1: embeddings */}
      <Panel x={16} title="Embeddings" />
      {points.map(([x, y, c], i) => (
        <circle key={i} cx={x} cy={y} r={2.6} fill={c === 0 ? BLUE : c === 1 ? BLUE_LIGHT : "#CBD5E1"} />
      ))}
      <circle cx={112} cy={170} r={9} fill="none" stroke={ORANGE} strokeWidth={1.5} />
      <circle cx={112} cy={170} r={3.6} fill={ORANGE} />
      <T x={28} y={284} size={8}>UMAP · 2D</T>

      {/* Panel 2: explanations */}
      <Panel x={204} title="Explanations" />
      <T x={216} y={106} size={8}>SALIENCY</T>
      {wave.map((v, i) => {
        const h = v * 44;
        return <rect key={i} x={216 + i * 3.4} y={138 - h / 2} width={2} height={h} rx={1}
          fill={i >= 14 && i <= 22 ? ORANGE : BLUE_LIGHT} />;
      })}
      <T x={216} y={180} size={8}>ATTENTION</T>
      {attn.map((v, i) => {
        const col = i % 6, row = Math.floor(i / 6);
        const diag = Math.abs(col - row) <= 1 ? 0.55 : 0;
        return <rect key={i} x={216 + col * 24} y={188 + row * 15} width={22} height={13} rx={2}
          fill={BLUE} fillOpacity={0.08 + Math.min(0.85, diag + v * 0.3)} />;
      })}

      {/* Panel 3: datapoint editor */}
      <Panel x={392} title="Datapoint editor" />
      {[
        ["file", "sample-000037.mp3"],
        ["duration", "3.86 s"],
        ["prediction", "minds in the door"],
        ["WER", "0.00"],
      ].map(([k, v], i) => (
        <g key={k}>
          <T x={404} y={110 + i * 30} size={8}>{k.toUpperCase()}</T>
          <T x={404} y={122 + i * 30} size={9.5} color={INK} weight={k === "prediction" ? 600 : 500}>{v}</T>
        </g>
      ))}
      <rect x={404} y={236} width={148} height={40} rx={6} fill={PANEL} stroke={RULE} />
      <circle cx={420} cy={256} r={8} fill={BLUE} />
      <path d="M418,252 L423,256 L418,260 Z" fill="#fff" />
      <rect x={434} y={253} width={108} height={6} rx={3} fill={RULE} />
      <rect x={434} y={253} width={44} height={6} rx={3} fill={ORANGE} />

      {/* Links from the selected point */}
      <path d="M121,166 C160,150 176,138 204,138" fill="none" stroke={ORANGE} strokeWidth={1.3} strokeDasharray="3 3" />
      <path d="M112,161 C120,40 380,40 404,98" fill="none" stroke={ORANGE} strokeWidth={1.3} strokeDasharray="3 3" />
    </svg>
  );
};

export default WorkbenchDiagram;
