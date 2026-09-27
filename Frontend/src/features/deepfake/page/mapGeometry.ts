/** Scale projected coordinates into [-1, 1] with ONE factor for every axis,
 *  so the cloud keeps its shape (a per-axis stretch would invent structure). */
export function normalise(coordinates: number[][], dims: number): number[][] {
  if (coordinates.length === 0) return [];
  const centre: number[] = [];
  let range = 0;
  for (let axis = 0; axis < dims; axis += 1) {
    const values = coordinates.map((row) => row[axis] ?? 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    centre.push((min + max) / 2);
    range = Math.max(range, max - min);
  }
  const half = range / 2 || 1;
  return coordinates.map((row) => centre.map((c, axis) => ((row[axis] ?? 0) - c) / half));
}

/** Indices of the k closest points to `index` on the map (not in the model). */
export function nearest(points: number[][], index: number, k: number): number[] {
  const origin = points[index];
  if (!origin) return [];
  return points
    .map((point, candidate) => ({
      candidate,
      distance: point.reduce((sum, value, axis) => sum + (value - origin[axis]) ** 2, 0),
    }))
    .filter(({ candidate }) => candidate !== index)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k)
    .map(({ candidate }) => candidate);
}

/**
 * Overlap removal for display: nudges points apart until no two are closer
 * than `minDistance`, moving each only as far as it must. Clusters open into
 * countable discs while keeping their place, shape and neighbours — the
 * trick embedding viewers use so a dense group is not one blob. Works in any
 * number of dimensions; `points` are already in screen/world units.
 */
export function declutter(points: number[][], minDistance: number, iterations = 80): number[][] {
  const out = points.map((point) => [...point]);
  const n = out.length;
  const dims = out[0]?.length ?? 0;
  const minSq = minDistance * minDistance;
  for (let pass = 0; pass < iterations; pass += 1) {
    let moved = false;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        let distSq = 0;
        for (let axis = 0; axis < dims; axis += 1) distSq += (out[j][axis] - out[i][axis]) ** 2;
        if (distSq >= minSq) continue;
        moved = true;
        let dist = Math.sqrt(distSq);
        const direction: number[] = [];
        if (dist < 1e-9) {
          // Identical points: separate along a deterministic direction.
          const angle = (i * 2.399963 + j) % (Math.PI * 2);
          for (let axis = 0; axis < dims; axis += 1) direction.push(axis === 0 ? Math.cos(angle) : axis === 1 ? Math.sin(angle) : Math.cos(angle * 1.7));
          dist = 0;
        } else {
          for (let axis = 0; axis < dims; axis += 1) direction.push((out[j][axis] - out[i][axis]) / dist);
        }
        const push = (minDistance - dist) / 2;
        for (let axis = 0; axis < dims; axis += 1) {
          out[i][axis] -= direction[axis] * push;
          out[j][axis] += direction[axis] * push;
        }
      }
    }
    if (!moved) break;
  }
  return out;
}
