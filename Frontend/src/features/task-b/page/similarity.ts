/** The segment-by-segment similarity matrix behind Step 4, kept free of UI so
 *  it can be tested. The ordering and the matrix itself are the pre-redesign
 *  `SimilarityMatrix` logic, unchanged. */
import type { DiarizationSegment } from "../types";

/** Only segments long enough to have an embedding take part. Time order by
 *  default (segments arrive in time order); grouped puts each speaker's
 *  segments together, in the run's speaker order, then by start time. */
export function orderSegments(
  segments: DiarizationSegment[],
  embeddings: Record<string, number[]>,
  speakers: string[],
  groupBySpeaker: boolean,
): DiarizationSegment[] {
  const embeddable = segments.filter((s) => embeddings[s.id]);
  if (!groupBySpeaker) return embeddable;
  return [...embeddable].sort(
    (a, b) => speakers.indexOf(a.speaker) - speakers.indexOf(b.speaker) || a.start - b.start,
  );
}

/** Row-major n×n cosine similarities. Cosine is the dot product here because
 *  the backend L2-normalises embeddings. */
export function similarityMatrix(ordered: DiarizationSegment[], embeddings: Record<string, number[]>): Float32Array {
  const n = ordered.length;
  const vectors = ordered.map((s) => embeddings[s.id]);
  const values = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    values[i * n + i] = 1;
    for (let j = i + 1; j < n; j++) {
      let dot = 0;
      const a = vectors[i];
      const b = vectors[j];
      for (let k = 0; k < a.length; k++) dot += a[k] * b[k];
      values[i * n + j] = dot;
      values[j * n + i] = dot;
    }
  }
  return values;
}

export interface SpeakerPair {
  a: string;
  b: string;
  /** Mean similarity over every cross pair of their segments. */
  similarity: number;
}

/** The two speakers whose segments are, on average, most alike — the pair the
 *  clustering is most likely to confuse. `null` with fewer than two speakers
 *  that have embedded segments. */
export function mostAlikeSpeakers(
  ordered: DiarizationSegment[],
  matrix: Float32Array,
  speakers: string[],
): SpeakerPair | null {
  const n = ordered.length;
  const sums = new Map<string, { total: number; count: number }>();
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const si = ordered[i].speaker;
      const sj = ordered[j].speaker;
      if (si === sj) continue;
      // Key in the run's speaker order, so A|B and B|A are one pair.
      const [a, b] = speakers.indexOf(si) <= speakers.indexOf(sj) ? [si, sj] : [sj, si];
      const key = `${a}\u0000${b}`;
      const entry = sums.get(key) ?? { total: 0, count: 0 };
      entry.total += matrix[i * n + j];
      entry.count += 1;
      sums.set(key, entry);
    }
  }
  let best: SpeakerPair | null = null;
  for (const [key, { total, count }] of sums) {
    const similarity = total / count;
    if (!best || similarity > best.similarity) {
      const [a, b] = key.split("\u0000");
      best = { a, b, similarity };
    }
  }
  return best;
}

/** Cut-offs for putting a WeSpeaker cosine similarity into words. A heuristic,
 *  stated as such in the technical details — not a calibrated threshold. */
export const VERY_ALIKE = 0.6;
export const SOMEWHAT_ALIKE = 0.3;

export const similarityWords = (value: number): string =>
  value >= VERY_ALIKE ? "sound very alike" : value >= SOMEWHAT_ALIKE ? "sound somewhat alike" : "sound different";

/** Uncertain segments, least confident first. Segments too short to score
 *  (`confidence: null`) are not uncertain — they have no score at all. */
export const uncertainSegments = (segments: DiarizationSegment[]): DiarizationSegment[] =>
  segments
    .filter((s) => s.confidence_bucket === "uncertain" && s.confidence !== null)
    .sort((a, b) => (a.confidence ?? 0) - (b.confidence ?? 0));
