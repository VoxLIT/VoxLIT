import type { ConfidenceBucket, DiarizationSegment } from "../types";

/** The confidence bucket in plain words. `null` means the segment was too
 *  short to embed, so it has no score at all — not a low one. */
export const confidenceWords = (bucket: ConfidenceBucket): string =>
  bucket === "high" ? "confident" : bucket === "medium" ? "fairly sure" : bucket === "uncertain" ? "unsure" : "too short to score";

/** `mm:ss.s`, e.g. 72.4 → "01:12.4". */
export const formatClock = (seconds: number): string => {
  // Round to tenths first, so 59.96 becomes "01:00.0" rather than "00:60.0".
  const tenths = Math.round(Math.max(0, seconds) * 10);
  const minutes = Math.floor(tenths / 600);
  const rest = (tenths - minutes * 600) / 10;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
};

/** What a screen reader hears for a segment. */
export const segmentLabel = (segment: DiarizationSegment): string =>
  `${segment.speaker}, ${formatClock(segment.start)} to ${formatClock(segment.end)}, ${confidenceWords(segment.confidence_bucket)}`;
