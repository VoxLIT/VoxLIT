/** Nearest segments in the model's own embedding space.
 *
 *  Always cosine on the full speaker embeddings from `/run`, never distance on
 *  the 2D map: PCA to two dimensions throws most of the geometry away, so two
 *  dots can sit side by side on the map while sounding nothing alike. */

export interface Neighbour {
  id: string;
  similarity: number;
}

/** Cosine similarity. The backend L2-normalises embeddings, but this does not
 *  rely on it. Zero-length vectors compare as 0 rather than NaN. */
export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator > 0 ? dot / denominator : 0;
}

/** The `k` segments most similar to `id`, most similar first. Empty when `id`
 *  has no embedding — segments under 0.4 s are never embedded. */
export function nearestByCosine(embeddings: Record<string, number[]>, id: string, k = 5): Neighbour[] {
  const origin = embeddings[id];
  if (!origin) return [];
  return Object.entries(embeddings)
    .filter(([candidate]) => candidate !== id)
    .map(([candidate, vector]) => ({ id: candidate, similarity: cosine(origin, vector) }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);
}
