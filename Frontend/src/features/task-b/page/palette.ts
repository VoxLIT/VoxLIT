/** Colours for the diarization page. Colour means speaker identity and
 *  nothing else: the same speaker has the same colour in every view.
 *  Uncertainty is shown with hatching and opacity, never with a new colour,
 *  and selection with size and a pulse, never by recolouring.
 *
 *  Two sets of the same eight hues: lighter tints for the dark theme, deeper
 *  shades for the light theme. The indigo accent is not a speaker colour. */
export const SPEAKERS_DARK = ["#5b9cf6", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#22d3ee", "#fb923c", "#a3e635"];
export const SPEAKERS_LIGHT = ["#2159c4", "#b45309", "#047857", "#be185d", "#6d28d9", "#0e7490", "#c2410c", "#4d7c0f"];

export interface Palette {
  SPEAKERS: string[];
  ACCENT: string;
  WARN: string;
  /** Solid "ink" for markers and playheads: near-white on dark, near-black on light. */
  INK: string;
  /** Ink at an opacity, for gridlines, tracks and empty lanes. */
  ink: (alpha: number) => string;
  /** The page background, for fills that must read as "cut out" of a chart. */
  CANVAS: string;
  /** Top of the similarity matrix's sequential ramp (bottom is CANVAS): an
   *  indigo far from the page background, and not a speaker colour. */
  HEAT: string;
  /** A speaker's colour, by their position in the run's `speakers` list. */
  speakerColor: (speakers: string[], speaker: string) => string;
}

const colourFor = (set: string[]) => (speakers: string[], speaker: string) =>
  set[Math.max(0, speakers.indexOf(speaker)) % set.length];

export const DARK_PALETTE: Palette = {
  SPEAKERS: SPEAKERS_DARK,
  ACCENT: "#6366f1",
  WARN: "#d4a72c",
  INK: "#e6e9f2",
  ink: (alpha) => `rgba(230,233,242,${alpha})`,
  CANVAS: "#0b0d16",
  HEAT: "#a5b4fc",
  speakerColor: colourFor(SPEAKERS_DARK),
};

export const LIGHT_PALETTE: Palette = {
  SPEAKERS: SPEAKERS_LIGHT,
  ACCENT: "#4f46e5",
  WARN: "#b7791f",
  INK: "#111827",
  ink: (alpha) => `rgba(17,24,39,${alpha})`,
  CANVAS: "#f6f7fb",
  HEAT: "#312e81",
  speakerColor: colourFor(SPEAKERS_LIGHT),
};
