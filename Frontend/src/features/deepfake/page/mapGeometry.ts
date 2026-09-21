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
