/** Colours for the deepfake page. A score is always coloured by the detector's
 *  own output — blue reads as a real voice, red as a synthetic one, neutral
 *  slate in between — never by the dataset's hidden labels.
 *
 *  A restrained, diverging blue/slate/red ramp that sits on the console's
 *  white panels next to the shared AWS-blue accent. */
export interface Palette {
  REAL: string;
  MID: string;
  FAKE: string;
  WARN: string;
  /** Solid "ink" for markers, needles and playheads. */
  INK: string;
  /** Ink at an opacity, for gridlines, tracks and unplayed bars. */
  ink: (alpha: number) => string;
  /** The page background, for fills that must read as "cut out" of a chart. */
  CANVAS: string;
  scoreColor: (score: number) => string;
}

const hex = (value: string) => [1, 3, 5].map((offset) => parseInt(value.slice(offset, offset + 2), 16));

const rampFor = (real: string, mid: string, fake: string) => {
  const stops = [hex(real), hex(mid), hex(fake)];
  return (score: number): string => {
    const t = Math.min(1, Math.max(0, Number.isFinite(score) ? score : 0));
    const [from, to, local] = t < 0.5 ? [stops[0], stops[1], t / 0.5] : [stops[1], stops[2], (t - 0.5) / 0.5];
    const mixed = from.map((channel, index) => Math.round(channel + (to[index] - channel) * local));
    return `#${mixed.map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
  };
};

export const CONSOLE_PALETTE: Palette = {
  REAL: "#2159c4",
  MID: "#64748b",
  FAKE: "#c53030",
  WARN: "#b7791f",
  INK: "#0f172a",
  ink: (alpha) => `rgba(15,23,42,${alpha})`,
  CANVAS: "#f6f7f9",
  scoreColor: rampFor("#2159c4", "#64748b", "#c53030"),
};

export const { REAL, MID, FAKE, WARN, scoreColor } = CONSOLE_PALETTE;

/** Plain-language strength of a score relative to the threshold. The score is
 *  a ranking, not a probability, so this deliberately avoids percentages. */
export const leanWords = (score: number, threshold: number): string => {
  const distance = Math.abs(score - threshold);
  if (distance >= 0.35) return "leans strongly";
  if (distance >= 0.15) return "leans";
  return "only just leans";
};

/** The backend rounds probabilities to 6 decimals, so that is the most any
 *  score can show. */
const MAX_DECIMALS = 6;

/** A probability with `digits` decimals, or more when that would round a real
 *  value to 0 or 1: enough to show two significant figures of the distance
 *  from the nearer end. Models B and C saturate on in-domain clips
 *  (0.000027, 0.999997), and "0.000" there reads as a failed run. */
export const formatScore = (score: number, digits = 3): string => {
  if (!Number.isFinite(score)) return "n/a";
  if (score <= 0) return `<0.${"0".repeat(MAX_DECIMALS - 1)}1`;
  if (score >= 1) return `>0.${"9".repeat(MAX_DECIMALS)}`;
  const gap = Math.min(score, 1 - score);
  const needed = Math.ceil(-Math.log10(gap)) + 1;
  const text = score.toFixed(Math.min(MAX_DECIMALS, Math.max(digits, needed)));
  // closer to an end than 6 decimals can show: bound it rather than round to 0 or 1
  if (Number(text) <= 0) return `<0.${"0".repeat(MAX_DECIMALS - 1)}1`;
  if (Number(text) >= 1) return `>0.${"9".repeat(MAX_DECIMALS)}`;
  return text;
};
