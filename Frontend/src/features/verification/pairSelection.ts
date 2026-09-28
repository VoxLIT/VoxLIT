/**
 * Computes the next verification pair selection when an embedding point is clicked.
 *
 * Rules:
 * 1. If clicking a clip that is already part of the active selection:
 *    - In a 2-clip pair [A, B], clicking B breaks pair comparison and retains ONLY [B].
 *      This ensures clip A's circle is removed, leaving only the actively clicked clip B.
 *    - In a 2-clip pair [A, B], clicking A breaks pair comparison and retains ONLY [A].
 *    - In a 1-clip selection [B], clicking B retains [B].
 * 2. If clicking a new clip when fewer than 2 clips are selected:
 *    - Appends the new clip, activating pair comparison once length reaches 2 (e.g. [A] -> [A, B]).
 * 3. If clicking a new clip when 2 clips are already selected:
 *    - Sliding window: drops the oldest clip and pairs the previous clip with the new one (e.g. [A, B] + C -> [B, C]).
 */
export function getNextVerificationPairSelection(
  prevSelection: string[] | undefined,
  clickedFilename: string
): string[] {
  const prev = prevSelection ?? [];
  if (prev.includes(clickedFilename)) {
    return [clickedFilename];
  }
  if (prev.length < 2) {
    return [...prev, clickedFilename];
  }
  return [prev[1], clickedFilename];
}
