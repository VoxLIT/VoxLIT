/** Colour ramp for the embedding view: the detector's spoof score, blue (0,
 *  reads as bona fide) to red (1, reads as spoof). Colouring by the model's own
 *  score never reveals the dataset's ground truth. Kept as plain "#rrggbb" so
 *  EmbeddingPlot's dimming/blending, which parses hex, works on it unchanged. */
export const BONAFIDE_COLOR = "#2563eb";
export const SPOOF_COLOR = "#dc2626";

const channel = (hex: string, offset: number): number =>
  parseInt(hex.slice(offset, offset + 2), 16);

export const spoofScoreColor = (score: number): string => {
  const t = Math.min(1, Math.max(0, Number.isFinite(score) ? score : 0));
  const mix = (offset: number) =>
    Math.round(channel(BONAFIDE_COLOR, offset) + (channel(SPOOF_COLOR, offset) - channel(BONAFIDE_COLOR, offset)) * t)
      .toString(16)
      .padStart(2, "0");
  return `#${mix(1)}${mix(3)}${mix(5)}`;
};
