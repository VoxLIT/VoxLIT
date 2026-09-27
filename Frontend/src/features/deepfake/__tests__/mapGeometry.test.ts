/**
 * Overlap removal on the voice map: after declutter no two points may sit
 * closer than the requested gap, and points already apart must not move.
 */
import { describe, expect, it } from "vitest";
import { declutter } from "../page/mapGeometry";

const closest = (points: number[][]) => {
  let min = Infinity;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      min = Math.min(min, Math.hypot(...points[i].map((value, axis) => value - points[j][axis])));
    }
  }
  return min;
};

describe("declutter", () => {
  it("opens a dense cluster until no pair is closer than the gap, in 2D and 3D", () => {
    for (const dims of [2, 3]) {
      const cluster = Array.from({ length: 300 }, (_, index) =>
        Array.from({ length: dims }, (_, axis) => Math.sin(index * (axis + 1)) * 0.02),
      );
      expect(closest(declutter(cluster, 0.05, 200))).toBeGreaterThan(0.05 * 0.97);
    }
  });

  it("separates identical points and leaves well-spaced ones where they are", () => {
    const stacked = declutter([[0, 0], [0, 0]], 1);
    expect(closest(stacked)).toBeGreaterThan(0.99);
    const apart = [[0, 0], [5, 5]];
    expect(declutter(apart, 1)).toEqual(apart);
  });
});
