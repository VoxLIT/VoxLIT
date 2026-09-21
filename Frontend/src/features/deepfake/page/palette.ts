/** Colours for the deepfake page. A score is always coloured by the detector's
 *  own output — blue reads as a real voice, red as a synthetic one, neutral
 *  slate in between — never by the dataset's hidden labels.
 *
 *  A restrained, diverging blue/slate/red scheme. Two sets: softer tints for
 *  the dark theme, and deeper shades of the same hues for the light theme. */
export interface Palette {
  REAL: string;
  MID: string;
  FAKE: string;
  WARN: string;
  /** Solid "ink" for markers, needles and playheads: white on dark, near-black on light. */
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

export const DARK_PALETTE: Palette = {
  REAL: "#5b9cf6",
  MID: "#8b97ab",
  FAKE: "#e5484d",
  WARN: "#d4a72c",
  INK: "#e6eaf2",
  ink: (alpha) => `rgba(230,234,242,${alpha})`,
  CANVAS: "#0b0f17",
  scoreColor: rampFor("#5b9cf6", "#8b97ab", "#e5484d"),
};

export const LIGHT_PALETTE: Palette = {
  REAL: "#2159c4",
  MID: "#64748b",
  FAKE: "#c53030",
  WARN: "#b7791f",
  INK: "#0f172a",
  ink: (alpha) => `rgba(15,23,42,${alpha})`,
  CANVAS: "#f6f7f9",
  scoreColor: rampFor("#2159c4", "#64748b", "#c53030"),
};

// The dark set stays the module default, so code outside the page's theme
// provider (and the existing tests) keeps its behaviour.
export const { REAL, MID, FAKE, WARN, scoreColor } = DARK_PALETTE;

/** Plain-language strength of a score relative to the threshold. The score is
 *  a ranking, not a probability, so this deliberately avoids percentages. */
export const leanWords = (score: number, threshold: number): string => {
  const distance = Math.abs(score - threshold);
  if (distance >= 0.35) return "leans strongly";
  if (distance >= 0.15) return "leans";
  return "only just leans";
};
