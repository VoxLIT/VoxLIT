import type { BatchAnalysisResponse } from "./batchTypes";

/** One unique recording pair (i < j) and its cosine similarity. */
export interface ScoredPair {
  i: number;
  j: number;
  similarity: number;
  /** Ground-truth relation; null when the batch has no known speaker groups. */
  sameSpeaker: boolean | null;
}

export interface ThresholdMetrics {
  threshold: number;
  totalPairs: number;
  accepted: number;
  /** Everything below is null when there is no ground truth (or, for a rate,
   *  when the class it divides by is empty). */
  samePairs: number | null;
  differentPairs: number | null;
  falseAccepts: number | null;
  falseRejects: number | null;
  far: number | null;
  frr: number | null;
  accuracy: number | null;
}

export interface ErrorCurvePoint {
  threshold: number;
  far: number | null;
  frr: number | null;
}

export interface EqualErrorRate {
  threshold: number;
  eer: number;
  far: number;
  frr: number;
}

export interface ChangedPair extends ScoredPair {
  acceptedAtCalibrated: boolean;
  acceptedAtWhatIf: boolean;
}

type PairSource = Pick<BatchAnalysisResponse, "similarity_matrix" | "ground_truth_groups" | "ground_truth_available">;

export const isAccepted = (similarity: number, threshold: number) => similarity >= threshold;

export function buildPairs(result: PairSource): ScoredPair[] {
  const matrix = result.similarity_matrix;
  const groups =
    result.ground_truth_available && result.ground_truth_groups?.length === matrix.length
      ? result.ground_truth_groups
      : null;
  const pairs: ScoredPair[] = [];
  for (let i = 0; i < matrix.length; i += 1) {
    for (let j = i + 1; j < matrix.length; j += 1) {
      const similarity = matrix[i]?.[j];
      if (typeof similarity !== "number" || !Number.isFinite(similarity)) continue;
      pairs.push({ i, j, similarity, sameSpeaker: groups ? groups[i] === groups[j] : null });
    }
  }
  return pairs;
}

export function metricsAt(pairs: ScoredPair[], threshold: number): ThresholdMetrics {
  let accepted = 0;
  let samePairs = 0;
  let differentPairs = 0;
  let falseAccepts = 0;
  let falseRejects = 0;
  for (const pair of pairs) {
    const accept = isAccepted(pair.similarity, threshold);
    if (accept) accepted += 1;
    if (pair.sameSpeaker === true) {
      samePairs += 1;
      if (!accept) falseRejects += 1;
    } else if (pair.sameSpeaker === false) {
      differentPairs += 1;
      if (accept) falseAccepts += 1;
    }
  }
  const labelled = samePairs + differentPairs;
  if (labelled === 0) {
    return {
      threshold,
      totalPairs: pairs.length,
      accepted,
      samePairs: null,
      differentPairs: null,
      falseAccepts: null,
      falseRejects: null,
      far: null,
      frr: null,
      accuracy: null,
    };
  }
  return {
    threshold,
    totalPairs: pairs.length,
    accepted,
    samePairs,
    differentPairs,
    falseAccepts,
    falseRejects,
    far: differentPairs > 0 ? falseAccepts / differentPairs : null,
    frr: samePairs > 0 ? falseRejects / samePairs : null,
    accuracy: (labelled - falseAccepts - falseRejects) / labelled,
  };
}

/** Same- and different-speaker similarities, each sorted ascending. */
function sortedClasses(pairs: ScoredPair[]) {
  const same: number[] = [];
  const different: number[] = [];
  for (const pair of pairs) {
    if (pair.sameSpeaker === true) same.push(pair.similarity);
    else if (pair.sameSpeaker === false) different.push(pair.similarity);
  }
  same.sort((a, b) => a - b);
  different.sort((a, b) => a - b);
  return { same, different };
}

/** Number of values in an ascending array that are strictly below `value`. */
function countBelow(sorted: number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (sorted[mid] < value) low = mid + 1;
    else high = mid;
  }
  return low;
}

const farAt = (different: number[], threshold: number) =>
  different.length > 0 ? (different.length - countBelow(different, threshold)) / different.length : null;

const frrAt = (same: number[], threshold: number) =>
  same.length > 0 ? countBelow(same, threshold) / same.length : null;

export function similarityRange(pairs: ScoredPair[]): { min: number; max: number } | null {
  if (pairs.length === 0) return null;
  let min = Infinity;
  let max = -Infinity;
  for (const pair of pairs) {
    if (pair.similarity < min) min = pair.similarity;
    if (pair.similarity > max) max = pair.similarity;
  }
  return { min, max };
}

/** FAR and FRR at `steps` evenly spaced thresholds from the lowest to the
 *  highest pair similarity (inclusive). */
export function errorCurve(pairs: ScoredPair[], steps: number): ErrorCurvePoint[] {
  const range = similarityRange(pairs);
  if (!range || steps < 1) return [];
  const { same, different } = sortedClasses(pairs);
  const count = range.max === range.min ? 1 : Math.max(2, Math.floor(steps));
  const points: ErrorCurvePoint[] = [];
  for (let k = 0; k < count; k += 1) {
    const threshold =
      k === count - 1 ? range.max : range.min + ((range.max - range.min) * k) / (count - 1);
    points.push({ threshold, far: farAt(different, threshold), frr: frrAt(same, threshold) });
  }
  return points;
}

/** The observed similarity at which FAR and FRR are closest, and their
 *  average there. Null when either class is empty. Ties keep the lowest
 *  threshold. */
export function equalErrorRate(pairs: ScoredPair[]): EqualErrorRate | null {
  const { same, different } = sortedClasses(pairs);
  if (same.length === 0 || different.length === 0) return null;
  const candidates = [...same, ...different].sort((a, b) => a - b);
  let best: EqualErrorRate | null = null;
  let bestGap = Infinity;
  let previous: number | null = null;
  for (const threshold of candidates) {
    if (threshold === previous) continue;
    previous = threshold;
    const far = (different.length - countBelow(different, threshold)) / different.length;
    const frr = countBelow(same, threshold) / same.length;
    const gap = Math.abs(far - frr);
    if (gap < bestGap) {
      bestGap = gap;
      best = { threshold, eer: (far + frr) / 2, far, frr };
    }
  }
  return best;
}

/** Pairs whose accept/reject decision differs between the two thresholds,
 *  closest to the calibrated threshold first. */
export function changedPairs(pairs: ScoredPair[], calibrated: number, whatIf: number): ChangedPair[] {
  const changed: ChangedPair[] = [];
  for (const pair of pairs) {
    const acceptedAtCalibrated = isAccepted(pair.similarity, calibrated);
    const acceptedAtWhatIf = isAccepted(pair.similarity, whatIf);
    if (acceptedAtCalibrated !== acceptedAtWhatIf) {
      changed.push({ ...pair, acceptedAtCalibrated, acceptedAtWhatIf });
    }
  }
  changed.sort(
    (a, b) => Math.abs(a.similarity - calibrated) - Math.abs(b.similarity - calibrated) || a.i - b.i || a.j - b.j
  );
  return changed;
}
